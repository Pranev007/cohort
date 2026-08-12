"""Injected posture anomalies — the ground truth the evaluator scores against.

Eligibility is derived from each category's own policy rather than hard-coded.
If a category legitimately publishes anyone-with-link URLs (marketing briefs do),
then "anyone-with-link" is *not* an anomaly for it and will not be injected there.
Getting this wrong is the classic way to manufacture a flattering benchmark: you
inject "anomalies" that are actually normal, the detector cannot find them, and
your recall looks bad — or worse, you inject them only where they are trivially
detectable and your precision looks fake.
"""

from __future__ import annotations

import numpy as np

from cohort.schema import AnomalyType, LabelTier, LinkScope, RepoType
from cohort.synthorg.acl import AccessState
from cohort.synthorg.categories import Category
from cohort.synthorg.org import Organisation

#: A downgrade is only an anomaly when the downgraded label is genuinely rare
#: for the category. Above this share of public+internal, it is just normal.
_MAX_DOWNGRADE_MASS = 0.10


def eligible_anomalies(cat: Category) -> list[str]:
    """Which anomaly types are genuinely abnormal for this category.

    Returns the enum *values*. `StrEnum` members compare equal to their values,
    so `AnomalyType.EXTERNAL_LINK in eligible_anomalies(cat)` still reads
    naturally, and callers that need to weight or sample them get plain strings
    without threading `.value` through every call site.
    """
    pol = cat.policy
    out: list[str] = []

    if pol.link_scope.get(LinkScope.ANYONE_WITH_LINK, 0.0) < 0.02:
        out.append(AnomalyType.EXTERNAL_LINK)

    if pol.repo_type.get(RepoType.ONEDRIVE_PERSONAL, 0.0) < 0.02:
        out.append(AnomalyType.WRONG_LOCATION)

    # Eligibility here is about the *destination* label being rare, not about the
    # source distribution being sensitive. An earlier version used
    # "confidential + restricted > 0.60", which admitted incident postmortems —
    # a category that is 33% internal natively. Relabelling one of those
    # "internal" is not an anomaly, it is the second most common state, and
    # injecting it produced unfindable-by-construction labels that made the
    # detector look worse than it is. Measure what is actually abnormal.
    downgrade_mass = pol.label_tier.get(LabelTier.PUBLIC, 0.0) + pol.label_tier.get(
        LabelTier.INTERNAL, 0.0
    )
    if downgrade_mass < _MAX_DOWNGRADE_MASS:
        out.append(AnomalyType.MISLABELED_DOWN)

    if pol.external_prob < 0.15:
        out.append(AnomalyType.THIRD_PARTY_ACCESS)

    broad = sum(p for sel, p in pol.group_grants if sel == "all_employees")
    if broad < 0.15:
        out.append(AnomalyType.OVERSHARED)

    # Dormant external access is abnormal even where external access itself is
    # routine: outside counsel should not still hold a live grant on a contract
    # nobody has touched in three years.
    if pol.external_prob < 0.45:
        out.append(AnomalyType.STALE_EXTERNAL)

    return out


def inject(
    state: AccessState,
    cat: Category,
    org: Organisation,
    rng: np.random.Generator,
    anomaly: AnomalyType,
) -> AccessState:
    """Mutate a normal access state into an anomalous one, in place-ish."""
    if anomaly == AnomalyType.EXTERNAL_LINK:
        state.link_scope = LinkScope.ANYONE_WITH_LINK
        # A public link usually comes with an explicit grant, not inheritance.
        if rng.random() < 0.6:
            state.acl_origin = "explicit"

    elif anomaly == AnomalyType.WRONG_LOCATION:
        state.repo_type = RepoType.ONEDRIVE_PERSONAL
        # Personal-drive copies sit shallower and break inheritance.
        state.path_depth = int(max(1, state.path_depth - rng.integers(1, 4)))
        state.acl_origin = "explicit"

    elif anomaly == AnomalyType.MISLABELED_DOWN:
        # Downgrade toward whichever tier the category uses *least*, so the
        # injected state is genuinely off-baseline rather than merely uncommon.
        p_public = cat.policy.label_tier.get(LabelTier.PUBLIC, 0.0)
        p_internal = cat.policy.label_tier.get(LabelTier.INTERNAL, 0.0)
        target = LabelTier.PUBLIC if p_public <= p_internal else LabelTier.INTERNAL
        state.label_tier = str(target)

    elif anomaly == AnomalyType.THIRD_PARTY_ACCESS:
        # Pick a partner org this category has no business relationship with.
        candidates = [o for o in org.external_groups if o != cat.policy.external_org]
        if candidates:
            gid = org.external_groups[str(rng.choice(candidates))]
            members = org.groups[gid].member_ids
            k = int(rng.integers(1, min(5, len(members)) + 1))
            state.direct_ids.extend(str(m) for m in rng.choice(members, size=k, replace=False))

    elif anomaly == AnomalyType.OVERSHARED:
        if org.all_company_group not in state.group_ids:
            state.group_ids.append(org.all_company_group)
        if rng.random() < 0.35:
            state.group_ids.append(str(rng.choice(org.project_groups)))

    elif anomaly == AnomalyType.STALE_EXTERNAL:
        partner = cat.policy.external_org
        if partner is None:
            partner = next(iter(org.external_groups))
        stale_gid = org.external_groups.get(partner)
        if stale_gid:
            members = org.groups[stale_gid].member_ids
            k = int(rng.integers(1, min(3, len(members)) + 1))
            state.direct_ids.extend(str(m) for m in rng.choice(members, size=k, replace=False))
        # Dormant for one to three years.
        state.staleness_days = int(rng.integers(760, 1400))
        state.age_days = max(state.age_days, state.staleness_days + int(rng.integers(30, 400)))

    return state
