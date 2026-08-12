"""Sampling access-control state, and turning it into posture features.

Normal ACLs are drawn from each category's policy. Anomalous ACLs are produced
by mutating a normal one (see `anomalies.py`). Both paths converge here, so the
feature vector for an injected anomaly is computed by exactly the same code as
for a clean document — no leakage of the label into the features.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from cohort.schema import AclOrigin, LinkScope, RepoType
from cohort.synthorg.categories import AclPolicy, Category
from cohort.synthorg.org import Organisation


@dataclass(slots=True)
class AccessState:
    group_ids: list[str] = field(default_factory=list)
    direct_ids: list[str] = field(default_factory=list)
    link_scope: str = LinkScope.NONE
    repo_type: str = RepoType.SHAREPOINT
    label_tier: str = "internal"
    acl_origin: str = AclOrigin.INHERITED
    owner_id: str = ""
    path_depth: int = 4
    age_days: int = 100
    staleness_days: int = 50
    pii_density: float = 1.0


def _sample_cat(rng: np.random.Generator, dist: dict[str, float]) -> str:
    keys = list(dist.keys())
    probs = np.array([dist[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(keys, p=probs))


def _resolve_selector(sel: str, org: Organisation, rng: np.random.Generator) -> str | None:
    if sel == "leadership":
        return org.leadership_group
    if sel == "all_employees":
        return org.all_company_group
    if sel == "project:random":
        return str(rng.choice(org.project_groups))
    if sel.startswith("dept:"):
        key = sel[len("dept:") :]
        return org.dept_groups.get(key)
    return None


def sample_access(
    cat: Category,
    org: Organisation,
    rng: np.random.Generator,
    owner_id: str,
    internal_pool: list[str] | None = None,
) -> AccessState:
    """Draw a *normal* access state for one document of this category.

    `internal_pool` is accepted so the caller can hoist it out of the per-document
    loop; materialising it here costs an O(n_employees) pass per document.
    """
    pol: AclPolicy = cat.policy

    group_ids: list[str] = []
    for sel, prob in pol.group_grants:
        if rng.random() < prob:
            gid = _resolve_selector(sel, org, rng)
            if gid:
                group_ids.append(gid)

    if internal_pool is None:
        internal_pool = [p.person_id for p in org.internal_people()]
    n_direct = int(rng.integers(pol.n_direct[0], pol.n_direct[1] + 1))
    direct_ids = (
        [
            str(x)
            for x in rng.choice(
                internal_pool, size=min(n_direct, len(internal_pool)), replace=False
            )
        ]
        if n_direct
        else []
    )
    if owner_id not in direct_ids:
        direct_ids.append(owner_id)

    # Legitimate external access for categories where that is normal practice.
    if pol.external_org and rng.random() < pol.external_prob:
        ext_gid = org.external_groups.get(pol.external_org)
        if ext_gid:
            members = org.groups[ext_gid].member_ids
            k = int(rng.integers(1, min(4, len(members)) + 1))
            direct_ids.extend(str(m) for m in rng.choice(members, size=k, replace=False))

    mu, sigma = pol.pii_density
    return AccessState(
        group_ids=group_ids,
        direct_ids=direct_ids,
        link_scope=_sample_cat(rng, pol.link_scope),
        repo_type=_sample_cat(rng, pol.repo_type),
        label_tier=_sample_cat(rng, pol.label_tier),
        acl_origin=_sample_cat(rng, pol.acl_origin),
        owner_id=owner_id,
        path_depth=int(rng.integers(pol.path_depth[0], pol.path_depth[1] + 1)),
        age_days=int(rng.integers(pol.age_days[0], pol.age_days[1] + 1)),
        staleness_days=int(rng.integers(pol.staleness_days[0], pol.staleness_days[1] + 1)),
        pii_density=float(max(0.0, rng.normal(mu, sigma))),
    )


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    # Summed in sorted order: float addition is not associative, so an unsorted
    # input would make the result depend on iteration order.
    h = 0.0
    for c in sorted(counts):
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def posture_features(state: AccessState, org: Organisation) -> dict[str, float | str | bool]:
    """Derive the scored feature vector from raw access state.

    Group expansion happens here: `n_principals` is the transitive count, not the
    number of ACL entries. A single grant to `All Employees` therefore looks very
    different from a single grant to a nine-person contracts team.
    """
    # Sorted, not raw set order. Python randomises string hashing per process
    # (PYTHONHASHSEED), so iterating the principal set directly makes the Counter's
    # insertion order vary between runs — and two features silently inherit that:
    #
    #   * `accessor_dept_entropy` sums floats, and float addition is not
    #     associative, so a different order gives a different last bit;
    #   * `Counter.most_common` breaks ties by insertion order, so a cohort with
    #     two equally-common departments could flip `owner_dept_is_modal`.
    #
    # The first is cosmetic, the second changes a scored categorical value. Both
    # made the corpus non-reproducible and were caught by the CI digest check.
    principals = org.resolve(state.group_ids, state.direct_ids)
    ordered = sorted(principals)

    depts = Counter(org.dept_of(p) for p in ordered)
    domains = org.domains_of(principals)
    external_domains = (
        {d for d in domains if d != org.people[state.owner_id].domain}
        if state.owner_id in org.people
        else set()
    )

    owner_dept = org.dept_of(state.owner_id)
    # Explicit tie-break on the department name so the result never depends on
    # how the Counter happened to be built.
    modal_dept = min(depts.items(), key=lambda kv: (-kv[1], kv[0]))[0] if depts else "Unknown"

    return {
        # categorical
        "link_scope": state.link_scope,
        "repo_type": state.repo_type,
        "label_tier": state.label_tier,
        "acl_origin": state.acl_origin,
        "has_external_principal": bool(external_domains),
        "owner_dept_is_modal": owner_dept == modal_dept,
        # continuous
        "n_principals": float(len(principals)),
        "n_groups": float(len(state.group_ids)),
        "n_external_domains": float(len(external_domains)),
        "accessor_dept_entropy": _entropy(list(depts.values())),
        "path_depth": float(state.path_depth),
        "age_days": float(state.age_days),
        "staleness_days": float(state.staleness_days),
        "pii_density": float(state.pii_density),
        "dup_count": 0.0,  # filled by the lineage stage
    }
