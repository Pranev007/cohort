"""The data contract shared by the generator, the scorer, and the explainer.

Keeping the feature schema in one place is what makes attributions, counterfactual
remediation, and the evaluation harness agree with each other. If you add a
posture feature, you add it here and every downstream stage picks it up.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class AnomalyType(StrEnum):
    """Injected posture anomalies. These are the labels the evaluator scores against."""

    EXTERNAL_LINK = "external_link_on_sensitive"
    WRONG_LOCATION = "wrong_location"
    MISLABELED_DOWN = "mislabeled_down"
    THIRD_PARTY_ACCESS = "third_party_access"
    OVERSHARED = "overshared_principal_count"
    STALE_EXTERNAL = "stale_external_access"

    @classmethod
    def all(cls) -> list[str]:
        return [m.value for m in cls]


class LabelTier(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


LABEL_ORDER: Final[list[str]] = [
    LabelTier.PUBLIC,
    LabelTier.INTERNAL,
    LabelTier.CONFIDENTIAL,
    LabelTier.RESTRICTED,
]


class LinkScope(StrEnum):
    NONE = "none"
    INTERNAL_LINK = "internal_link"
    DOMAIN_LINK = "domain_link"
    ANYONE_WITH_LINK = "anyone_with_link"


class RepoType(StrEnum):
    SHAREPOINT = "sharepoint"
    ONEDRIVE_PERSONAL = "onedrive_personal"
    GDRIVE_SHARED = "gdrive_shared"
    S3_BUCKET = "s3_bucket"
    SMB_SHARE = "smb_share"
    CONFLUENCE = "confluence"
    SLACK = "slack"


class AclOrigin(StrEnum):
    INHERITED = "inherited"
    EXPLICIT = "explicit"
    MIXED = "mixed"


# --------------------------------------------------------------------------
# Posture feature schema
# --------------------------------------------------------------------------
# Categorical features are scored by Dirichlet-smoothed surprisal against the
# cohort's empirical distribution. Continuous features are scored by a robust
# two-sided marginal surprisal. Both are in nats and simply add up, which is
# what makes the total score exactly decomposable.

CATEGORICAL_FEATURES: Final[list[str]] = [
    "link_scope",
    "repo_type",
    "label_tier",
    "acl_origin",
    "has_external_principal",
    "owner_dept_is_modal",
]

CONTINUOUS_FEATURES: Final[list[str]] = [
    "n_principals",
    "n_groups",
    "n_external_domains",
    "accessor_dept_entropy",
    "path_depth",
    "age_days",
    "staleness_days",
    "pii_density",
    "dup_count",
]

POSTURE_FEATURES: Final[list[str]] = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES

#: Which tail of a continuous feature is actually risky.
#:
#: This matters more than it looks. A symmetric anomaly score flags a contract
#: shared with *fewer* people than its peers just as loudly as one shared with
#: more — which is noise, and noise is what gets a security product uninstalled.
#: Only `path_depth` is genuinely two-sided: an unusually shallow path suggests a
#: file dragged to a drive root, an unusually deep one a copy buried out of view.
FEATURE_DIRECTION: Final[dict[str, str]] = {
    "n_principals": "high",
    "n_groups": "high",
    "n_external_domains": "high",
    "accessor_dept_entropy": "high",
    "path_depth": "both",
    "age_days": "high",
    "staleness_days": "high",
    "pii_density": "high",
    "dup_count": "high",
}

#: Features a remediation action can actually change. `age_days` is not one of
#: them — you cannot un-age a document — so the counterfactual search skips it.
MUTABLE_FEATURES: Final[list[str]] = [
    "link_scope",
    "repo_type",
    "label_tier",
    "acl_origin",
    "has_external_principal",
    "n_principals",
    "n_groups",
    "n_external_domains",
    "accessor_dept_entropy",
]

#: Rough operational cost of each remediation, used to rank counterfactual edits.
#: Revoking a public link is cheap and safe; relocating a file may break links.
EDIT_COST: Final[dict[str, float]] = {
    "link_scope": 1.0,
    "has_external_principal": 1.5,
    "n_external_domains": 1.5,
    "label_tier": 2.0,
    "n_principals": 2.5,
    "n_groups": 2.5,
    "acl_origin": 3.0,
    "accessor_dept_entropy": 3.5,
    "repo_type": 4.0,
}

#: Maps a feature edit onto the concrete platform action an operator would take.
REMEDIATION_VERB: Final[dict[str, str]] = {
    "link_scope": "Restrict sharing link scope",
    "repo_type": "Relocate document",
    "label_tier": "Reapply sensitivity label",
    "acl_origin": "Restore permission inheritance",
    "has_external_principal": "Remove external principals",
    "n_principals": "Reduce the principals with access",
    "n_groups": "Remove broad group grants",
    "n_external_domains": "Remove external domains",
    "accessor_dept_entropy": "Narrow access to owning department",
}

FINDING_COLUMNS: Final[list[str]] = [
    "doc_id",
    "cohort_id",
    "cohort_name",
    "risk_score",
    "conformal_p",
    "is_flagged",
    "top_attributions",
    "narrative",
]
