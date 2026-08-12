"""Document categories: what they say, and what "normal" access looks like for each.

Two things are defined per category and they are deliberately independent:

1. **Content** — the text, which is all the semantic layer ever sees. The pipeline
   is never told which category a document belongs to; it has to recover the
   grouping from language alone.
2. **Access policy** — the distribution over posture features that constitutes
   normal handling for that category. This is what the scorer has to learn, also
   without being told.

Categories share boilerplate and overlap in vocabulary on purpose (an NDA and an
MSA are both contracts; an offer letter and a performance review are both about
one employee). A corpus where every category is lexically disjoint would make the
clustering result meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class AclPolicy:
    """The distribution of *normal* posture for a category."""

    label_tier: dict[str, float]
    link_scope: dict[str, float]
    repo_type: dict[str, float]
    acl_origin: dict[str, float]
    #: Group selectors granted access, each with an independent probability.
    group_grants: list[tuple[str, float]]
    n_direct: tuple[int, int]
    #: Probability this category legitimately involves an external partner.
    external_prob: float
    external_org: str | None
    age_days: tuple[int, int]
    staleness_days: tuple[int, int]
    pii_density: tuple[float, float]
    path_depth: tuple[int, int]


@dataclass(slots=True)
class Category:
    key: str
    display: str
    owning_depts: list[str]
    base_rate: float
    policy: AclPolicy
    title_patterns: list[str]
    templates: list[str]
    lexicon: list[str]
    multilingual: bool = False
    de_templates: list[str] = field(default_factory=list)
    fr_templates: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Shared boilerplate — appears across categories so clusters are not trivially
# separable by the presence of any single phrase.
# --------------------------------------------------------------------------
BOILERPLATE = [
    "This document is the property of {company} and is intended solely for the recipients named herein.",
    "Distribution outside the organisation requires written authorisation from the document owner.",
    "Retention of this record is governed by the {company} information lifecycle standard.",
    "Questions regarding this document should be directed to the owning team via the internal service desk.",
    "Printed copies of this document are uncontrolled and may not reflect the current revision.",
    "Reviewed under the {company} records management schedule; supersedes all prior revisions.",
]

CLOSERS = [
    "Prepared by {person} on {date}.",
    "Owner: {person}. Last reviewed {date}.",
    "Filed {date} by {person} under reference {ref}.",
    "Approved {date}. Point of contact: {person}.",
]


def _p(**kw: float) -> dict[str, float]:
    total = sum(kw.values())
    return {k: v / total for k, v in kw.items()}


CATEGORIES: list[Category] = [
    # ---------------------------------------------------------------- People
    Category(
        key="employment_offer",
        display="Employment offer letters",
        owning_depts=["People"],
        base_rate=0.09,
        policy=AclPolicy(
            label_tier=_p(confidential=0.82, restricted=0.16, internal=0.02),
            link_scope=_p(none=0.93, internal_link=0.07),
            repo_type=_p(sharepoint=0.86, gdrive_shared=0.14),
            acl_origin=_p(inherited=0.74, explicit=0.24, mixed=0.02),
            group_grants=[("dept:People:comp", 0.95), ("leadership", 0.18)],
            n_direct=(1, 4),
            external_prob=0.0,
            external_org=None,
            age_days=(10, 900),
            staleness_days=(5, 700),
            pii_density=(9.0, 3.0),
            path_depth=(4, 7),
        ),
        title_patterns=[
            "Offer of Employment - {person}",
            "Employment Agreement - {person} ({dept})",
        ],
        lexicon=[
            "base salary",
            "equity grant",
            "vesting schedule",
            "signing bonus",
            "probation period",
            "notice period",
            "annual leave entitlement",
            "start date",
            "reporting line",
            "restricted stock units",
            "at-will employment",
            "relocation allowance",
        ],
        templates=[
            "We are pleased to offer you the position of {title} within the {dept} organisation at {company}, based in {city}.",
            "Your annual {term} will be {money}, payable monthly in arrears and reviewed each performance cycle.",
            "You will be granted {n} {term} subject to the standard four-year {term} with a one-year cliff.",
            "This offer is contingent on satisfactory completion of background verification and reference checks.",
            "Your anticipated start date is {date}, reporting to {person}, {title}.",
            "The role carries a {term} of three months during which either party may terminate with two weeks' notice.",
            "A {term} of {money} will be paid within thirty days of your start date and is repayable if you resign within twelve months.",
            "Your {term} is twenty-five working days per annum in addition to public holidays observed in {city}.",
            "Compensation is reviewed annually; adjustments take effect at the start of the following fiscal quarter.",
            "Please confirm acceptance by countersigning below and returning this letter to People Operations by {date}.",
            "This letter supersedes any prior oral or written representations concerning the terms of your employment.",
        ],
    ),
    Category(
        key="performance_review",
        display="Performance reviews",
        owning_depts=["People"],
        base_rate=0.08,
        policy=AclPolicy(
            label_tier=_p(confidential=0.88, restricted=0.10, internal=0.02),
            link_scope=_p(none=0.96, internal_link=0.04),
            repo_type=_p(sharepoint=0.79, gdrive_shared=0.21),
            acl_origin=_p(inherited=0.80, explicit=0.19, mixed=0.01),
            group_grants=[("dept:People", 0.72), ("leadership", 0.30)],
            n_direct=(2, 5),
            external_prob=0.0,
            external_org=None,
            age_days=(15, 800),
            staleness_days=(20, 600),
            pii_density=(6.5, 2.2),
            path_depth=(4, 8),
        ),
        title_patterns=["Performance Review {date} - {person}", "H{n} Calibration Notes - {dept}"],
        lexicon=[
            "calibration",
            "rating distribution",
            "development goal",
            "competency",
            "peer feedback",
            "promotion readiness",
            "performance improvement plan",
            "growth area",
            "impact narrative",
            "stretch assignment",
            "succession risk",
        ],
        templates=[
            "This review covers the performance period ending {date} for {person}, {title} in {dept}.",
            "Overall rating: exceeds expectations. The {term} placed this outcome in the upper quartile for the function.",
            "Key strengths observed this cycle include ownership of the {project} workstream and consistent {term}.",
            "Identified {term}: broaden influence across adjacent teams and deepen written communication.",
            "Manager commentary notes reliable delivery under ambiguity and strong partnership with {dept}.",
            "{term} indicates the individual is ready for the next level within two review cycles.",
            "Peer input was gathered from four colleagues; themes were consistent with manager assessment.",
            "A {term} has not been initiated; performance is comfortably above the expected bar for the level.",
            "Development plan for the coming period focuses on a {term} leading the {project} migration.",
            "Compensation recommendation has been submitted separately to the compensation committee.",
        ],
    ),
    # ----------------------------------------------------------------- Legal
    Category(
        key="vendor_msa",
        display="Vendor master service agreements",
        owning_depts=["Legal"],
        base_rate=0.10,
        multilingual=True,
        policy=AclPolicy(
            label_tier=_p(confidential=0.70, internal=0.24, restricted=0.06),
            link_scope=_p(none=0.80, internal_link=0.19, domain_link=0.01),
            repo_type=_p(sharepoint=0.71, gdrive_shared=0.19, confluence=0.10),
            acl_origin=_p(inherited=0.66, explicit=0.31, mixed=0.03),
            group_grants=[
                ("dept:Legal:contracts", 0.92),
                ("dept:Finance", 0.28),
                ("project:random", 0.22),
            ],
            n_direct=(2, 9),
            # Outside counsel legitimately sees contracts. This is the control
            # that stops the scorer from simply learning "external == bad".
            external_prob=0.34,
            external_org="Halloran & Vetch LLP",
            age_days=(30, 1500),
            staleness_days=(30, 1200),
            pii_density=(1.8, 1.1),
            path_depth=(3, 6),
        ),
        title_patterns=["Master Services Agreement - {counterparty}", "MSA {counterparty} v{n}"],
        lexicon=[
            "indemnification",
            "limitation of liability",
            "service level credit",
            "governing law",
            "termination for convenience",
            "assignment",
            "force majeure",
            "statement of work",
            "acceptance criteria",
            "payment terms",
            "audit rights",
            "subprocessor",
        ],
        templates=[
            "This Master Services Agreement is entered into as of {date} between {company} and {counterparty}.",
            "The Supplier shall provide the services described in each {term} executed under this Agreement.",
            "Aggregate liability under this Agreement shall not exceed {money} or the fees paid in the preceding twelve months, whichever is greater.",
            "Each party shall maintain commercial general liability insurance of not less than {money} per occurrence.",
            "{term}: this Agreement shall be governed by the laws of the jurisdiction in which {company} maintains its registered office.",
            "The Customer may terminate any {term} for convenience on sixty days' written notice.",
            "{term} of ninety days apply to all invoices properly rendered under this Agreement.",
            "The Supplier shall not engage any {term} without the Customer's prior written consent.",
            "Failure to meet the agreed availability target entitles the Customer to a {term} calculated per Schedule 3.",
            "Neither party shall be liable for delay caused by events of {term} beyond its reasonable control.",
            "The Customer reserves {term} to verify the Supplier's compliance no more than once per calendar year.",
            "{term} shall survive termination or expiry of this Agreement for a period of six years.",
        ],
        de_templates=[
            "Dieser Rahmenvertrag wird zum {date} zwischen {company} und {counterparty} geschlossen.",
            "Der Lieferant erbringt die in der jeweiligen Leistungsbeschreibung genannten Dienstleistungen.",
            "Die Gesamthaftung aus diesem Vertrag ist auf {money} begrenzt.",
            "Jede Partei unterhaelt eine Betriebshaftpflichtversicherung in angemessener Hoehe.",
            "Der Kunde kann jede Leistungsbeschreibung mit einer Frist von sechzig Tagen kuendigen.",
            "Zahlungsbedingungen von neunzig Tagen gelten fuer alle ordnungsgemaess gestellten Rechnungen.",
            "Der Lieferant darf ohne vorherige schriftliche Zustimmung keine Unterauftragnehmer einsetzen.",
            "Vertraulichkeitsverpflichtungen gelten sechs Jahre ueber das Vertragsende hinaus fort.",
        ],
        fr_templates=[
            "Le present contrat-cadre est conclu le {date} entre {company} et {counterparty}.",
            "Le prestataire fournit les services decrits dans chaque bon de commande signe.",
            "La responsabilite totale au titre du present contrat est limitee a {money}.",
            "Chaque partie souscrit une assurance de responsabilite civile professionnelle.",
            "Le client peut resilier tout bon de commande moyennant un preavis de soixante jours.",
            "Les conditions de paiement sont de quatre-vingt-dix jours a compter de la facture.",
            "Le prestataire ne peut recourir a un sous-traitant sans accord ecrit prealable.",
            "Les obligations de confidentialite survivent six ans a l'expiration du contrat.",
        ],
    ),
    Category(
        key="nda",
        display="Non-disclosure agreements",
        owning_depts=["Legal"],
        base_rate=0.07,
        multilingual=True,
        policy=AclPolicy(
            label_tier=_p(internal=0.52, confidential=0.46, public=0.02),
            link_scope=_p(none=0.62, internal_link=0.33, domain_link=0.05),
            repo_type=_p(sharepoint=0.62, gdrive_shared=0.26, confluence=0.12),
            acl_origin=_p(inherited=0.71, explicit=0.27, mixed=0.02),
            group_grants=[("dept:Legal", 0.88), ("dept:Sales", 0.34), ("project:random", 0.18)],
            n_direct=(1, 7),
            external_prob=0.41,
            external_org="Halloran & Vetch LLP",
            age_days=(20, 1400),
            staleness_days=(40, 1300),
            pii_density=(1.4, 0.9),
            path_depth=(3, 6),
        ),
        title_patterns=["Mutual NDA - {counterparty}", "Confidentiality Agreement {counterparty}"],
        lexicon=[
            "confidential information",
            "permitted purpose",
            "residual knowledge",
            "return or destruction",
            "term of confidentiality",
            "injunctive relief",
            "no licence granted",
            "disclosure to advisers",
            "marking requirement",
            "compelled disclosure",
        ],
        templates=[
            "This mutual confidentiality agreement is made on {date} between {company} and {counterparty}.",
            "{term} means any non-public information disclosed by one party to the other, whether or not marked.",
            "The Receiving Party shall use the {term} solely for evaluating a potential commercial relationship.",
            "Obligations of confidence shall continue for five years from the date of disclosure.",
            "Nothing in this agreement constitutes a {term} in respect of any intellectual property.",
            "The Receiving Party may make {term} on a need-to-know basis provided equivalent obligations are imposed.",
            "Upon written request the Receiving Party shall effect {term} of all materials containing Confidential Information.",
            "The parties acknowledge that damages may be an inadequate remedy and that {term} may be sought.",
            "{term} pursuant to law or regulation is permitted provided prompt notice is given where lawful.",
            "This agreement does not oblige either party to enter into any further transaction.",
        ],
        de_templates=[
            "Diese gegenseitige Geheimhaltungsvereinbarung wird am {date} zwischen {company} und {counterparty} geschlossen.",
            "Vertrauliche Informationen sind alle nicht oeffentlichen Informationen, die eine Partei offenlegt.",
            "Der Empfaenger verwendet die vertraulichen Informationen ausschliesslich zum vereinbarten Zweck.",
            "Die Geheimhaltungspflicht besteht fuenf Jahre ab dem Zeitpunkt der Offenlegung fort.",
            "Diese Vereinbarung gewaehrt keine Lizenz an geistigem Eigentum.",
            "Eine Offenlegung an Berater ist zulaessig, sofern gleichwertige Pflichten auferlegt werden.",
            "Auf schriftliche Anforderung sind saemtliche Unterlagen zurueckzugeben oder zu vernichten.",
        ],
        fr_templates=[
            "Le present accord de confidentialite mutuel est conclu le {date} entre {company} et {counterparty}.",
            "Les informations confidentielles designent toute information non publique divulguee par une partie.",
            "La partie receptrice utilise ces informations uniquement aux fins convenues.",
            "Les obligations de confidentialite subsistent cinq ans a compter de la divulgation.",
            "Le present accord ne confere aucune licence sur un droit de propriete intellectuelle.",
            "La divulgation a des conseils est permise sous reserve d'obligations equivalentes.",
            "Sur demande ecrite, tous les documents doivent etre restitues ou detruits.",
        ],
    ),
    # ------------------------------------------------------------- Executive
    Category(
        key="board_minutes",
        display="Board and committee minutes",
        owning_depts=["Executive"],
        base_rate=0.04,
        policy=AclPolicy(
            label_tier=_p(restricted=0.86, confidential=0.14),
            link_scope=_p(none=0.99, internal_link=0.01),
            repo_type=_p(sharepoint=0.93, gdrive_shared=0.07),
            acl_origin=_p(explicit=0.88, inherited=0.11, mixed=0.01),
            group_grants=[("leadership", 0.62)],
            n_direct=(3, 9),
            external_prob=0.09,
            external_org="Brightmoor Audit Group",
            age_days=(20, 1100),
            staleness_days=(20, 1000),
            pii_density=(2.2, 1.0),
            path_depth=(2, 4),
        ),
        title_patterns=["Board Minutes {date}", "Audit Committee Minutes {date}"],
        lexicon=[
            "quorum",
            "resolution",
            "matters arising",
            "apologies for absence",
            "conflict of interest",
            "declaration of interests",
            "in camera session",
            "delegated authority",
            "any other business",
            "chair's report",
            "reserved matter",
        ],
        templates=[
            "Minutes of the meeting of the Board of Directors of {company} held on {date} at the registered office.",
            "Present: {person} (Chair), {person}, {person}. {term} was noted from one director.",
            "The Chair confirmed that a {term} was present and declared the meeting duly convened.",
            "The board reviewed the {project} investment case and the associated capital commitment of {money}.",
            "A {term} was passed unanimously approving the revised treasury policy.",
            "Under {term}, the board discussed the pending regulatory correspondence without management present.",
            "{term}: no member declared an interest in the matters before the meeting.",
            "The board noted the {term} covering trading performance and pipeline coverage for the quarter.",
            "The committee agreed that acquisition of any target above {money} remains a {term} for the full board.",
            "There being no {term}, the Chair closed the meeting.",
        ],
    ),
    Category(
        key="ma_term_sheet",
        display="M&A term sheets and diligence",
        owning_depts=["Executive", "Legal"],
        base_rate=0.04,
        policy=AclPolicy(
            label_tier=_p(restricted=0.79, confidential=0.21),
            link_scope=_p(none=0.97, internal_link=0.03),
            repo_type=_p(sharepoint=0.84, gdrive_shared=0.16),
            acl_origin=_p(explicit=0.83, inherited=0.15, mixed=0.02),
            group_grants=[("leadership", 0.44), ("dept:Legal:contracts", 0.36)],
            n_direct=(2, 8),
            external_prob=0.38,
            external_org="Halloran & Vetch LLP",
            age_days=(10, 500),
            staleness_days=(5, 400),
            pii_density=(1.6, 0.9),
            path_depth=(3, 5),
        ),
        title_patterns=["Project {project} - Term Sheet", "Project {project} Diligence Summary"],
        lexicon=[
            "exclusivity period",
            "enterprise value",
            "earn-out",
            "escrow",
            "working capital adjustment",
            "closing conditions",
            "material adverse change",
            "representations and warranties",
            "break fee",
            "data room",
            "synergy case",
        ],
        templates=[
            "Project {project}: indicative offer summary prepared for the board, dated {date}.",
            "Proposed {term} of {money} on a cash-free debt-free basis, subject to a customary {term}.",
            "The buyer requests a {term} of ninety days from signature of this term sheet.",
            "{money} of consideration would be held in {term} for eighteen months against warranty claims.",
            "An {term} of up to {money} is payable on achievement of the agreed revenue milestone.",
            "Diligence access will be provided through a virtual {term} hosted by outside counsel.",
            "{term} include regulatory clearance and the absence of any {term} prior to completion.",
            "The {term} identifies {money} of annualised cost savings realisable within eighteen months.",
            "A {term} of {money} is payable if the seller terminates to accept a superior proposal.",
            "Legal, financial and technical diligence workstreams are scheduled to conclude by {date}.",
        ],
    ),
    # --------------------------------------------------------------- Finance
    Category(
        key="quarterly_financials",
        display="Quarterly financial reporting",
        owning_depts=["Finance"],
        base_rate=0.08,
        policy=AclPolicy(
            label_tier=_p(confidential=0.68, restricted=0.28, internal=0.04),
            link_scope=_p(none=0.90, internal_link=0.10),
            repo_type=_p(sharepoint=0.70, gdrive_shared=0.18, s3_bucket=0.12),
            acl_origin=_p(inherited=0.62, explicit=0.35, mixed=0.03),
            group_grants=[("dept:Finance:controllers", 0.90), ("leadership", 0.40)],
            n_direct=(2, 7),
            external_prob=0.26,
            external_org="Brightmoor Audit Group",
            age_days=(15, 1000),
            staleness_days=(10, 800),
            pii_density=(0.8, 0.6),
            path_depth=(3, 6),
        ),
        title_patterns=["Q{n} Financial Summary", "Management Accounts {date}"],
        lexicon=[
            "gross margin",
            "deferred revenue",
            "accrual",
            "net revenue retention",
            "operating expenditure",
            "cash conversion",
            "impairment",
            "covenant headroom",
            "revenue recognition",
            "EBITDA bridge",
            "foreign exchange translation",
        ],
        templates=[
            "Management accounts for the quarter ended {date}, prepared for internal review prior to audit.",
            "Revenue of {money} represents growth of {pct} against the comparable prior-year period.",
            "{term} improved to {pct}, driven by favourable mix and reduced infrastructure unit costs.",
            "{term} of {money} was released in the period following delivery of contracted milestones.",
            "Operating expenditure of {money} includes a one-off {term} charge relating to the {project} platform.",
            "{term} stands at {pct}, comfortably ahead of the trailing four-quarter average.",
            "An {term} of {money} has been recognised against capitalised development costs.",
            "{term} against the senior facility remains above the required threshold at all measurement dates.",
            "{term} policy has been applied consistently with the prior period; no changes in estimate were made.",
            "The {term} reconciles reported operating profit to adjusted earnings for the quarter.",
            "{term} contributed an adverse variance of {money} relative to budgeted rates.",
        ],
    ),
    Category(
        key="invoice_ap",
        display="Accounts payable invoices",
        owning_depts=["Finance"],
        base_rate=0.09,
        policy=AclPolicy(
            label_tier=_p(internal=0.79, confidential=0.19, public=0.02),
            link_scope=_p(none=0.71, internal_link=0.27, domain_link=0.02),
            repo_type=_p(sharepoint=0.48, s3_bucket=0.30, smb_share=0.22),
            acl_origin=_p(inherited=0.86, explicit=0.13, mixed=0.01),
            group_grants=[("dept:Finance", 0.93)],
            n_direct=(1, 4),
            external_prob=0.11,
            external_org="Kestrel Cloud Systems",
            age_days=(5, 1200),
            staleness_days=(5, 1100),
            pii_density=(2.6, 1.2),
            path_depth=(5, 9),
        ),
        title_patterns=["Invoice {ref} - {counterparty}", "AP Batch {date} - {counterparty}"],
        lexicon=[
            "purchase order",
            "remittance advice",
            "line item",
            "tax point",
            "credit note",
            "three-way match",
            "goods received note",
            "withholding",
            "net thirty",
            "supplier reference",
        ],
        templates=[
            "Invoice reference {ref} issued by {counterparty} to {company} on {date}.",
            "Total amount due {money}, payable {term} from the tax point shown above.",
            "{term} number matched against the corresponding {term} prior to release for payment.",
            "{term} detail: professional services rendered in connection with the {project} engagement.",
            "A {term} of {money} was applied against invoice {ref} following a billing correction.",
            "{term} completed by accounts payable; no exceptions were raised during processing.",
            "{term} of {money} has been deducted in accordance with applicable tax regulations.",
            "{term} should be sent to the bank details held on the approved supplier master record.",
            "Supplier bank detail changes require verbal verification against the {term} on file.",
            "Payment scheduled in the run dated {date} subject to approval by the {dept} controller.",
        ],
    ),
    # ----------------------------------------------------------- Engineering
    Category(
        key="design_doc",
        display="Engineering design documents",
        owning_depts=["Engineering"],
        base_rate=0.11,
        policy=AclPolicy(
            label_tier=_p(internal=0.86, confidential=0.13, public=0.01),
            link_scope=_p(internal_link=0.58, none=0.39, domain_link=0.03),
            repo_type=_p(confluence=0.61, gdrive_shared=0.24, sharepoint=0.15),
            acl_origin=_p(inherited=0.90, explicit=0.09, mixed=0.01),
            group_grants=[
                ("dept:Engineering", 0.86),
                ("project:random", 0.52),
                ("all_employees", 0.09),
            ],
            n_direct=(0, 3),
            external_prob=0.04,
            external_org="Kestrel Cloud Systems",
            age_days=(5, 1300),
            staleness_days=(5, 900),
            pii_density=(0.3, 0.3),
            path_depth=(3, 7),
        ),
        title_patterns=["Design: {service} {project}", "RFC {n}: {service} architecture"],
        lexicon=[
            "idempotency",
            "backpressure",
            "eventual consistency",
            "sharding key",
            "circuit breaker",
            "write-ahead log",
            "blast radius",
            "cold start",
            "schema migration",
            "read replica",
            "exactly-once delivery",
            "leader election",
        ],
        templates=[
            "This document proposes a redesign of the {service} component supporting the {project} programme.",
            "Current throughput peaks at {n} requests per second; the target for this quarter is a fourfold increase.",
            "We propose partitioning on a composite {term} to distribute load evenly across replicas.",
            "The consumer implements {term} so that duplicate deliveries do not produce duplicate side effects.",
            "A {term} guards the downstream dependency and sheds load once the error budget is exhausted.",
            "{term} is acceptable for the read path; the write path requires linearisable semantics.",
            "{term} will be performed online using dual writes followed by a backfill and a verification pass.",
            "Rollout is staged by tenant cohort to limit the {term} of any regression.",
            "The {term} is fsynced before acknowledgement, trading roughly two milliseconds of latency for durability.",
            "Alternatives considered included a queue-per-tenant topology, rejected on operational cost grounds.",
            "Open questions: retention policy for the compacted topic, and ownership of the {term} runbook.",
        ],
    ),
    Category(
        key="incident_postmortem",
        display="Security and reliability postmortems",
        owning_depts=["Security", "Engineering"],
        base_rate=0.06,
        policy=AclPolicy(
            label_tier=_p(confidential=0.62, internal=0.33, restricted=0.05),
            link_scope=_p(none=0.66, internal_link=0.33, domain_link=0.01),
            repo_type=_p(confluence=0.56, sharepoint=0.30, gdrive_shared=0.14),
            acl_origin=_p(inherited=0.72, explicit=0.26, mixed=0.02),
            group_grants=[
                ("dept:Security:irt", 0.84),
                ("dept:Engineering", 0.36),
                ("leadership", 0.22),
            ],
            n_direct=(1, 6),
            external_prob=0.07,
            external_org="Kestrel Cloud Systems",
            age_days=(3, 700),
            staleness_days=(3, 600),
            pii_density=(1.2, 0.8),
            path_depth=(3, 6),
        ),
        title_patterns=["Postmortem: {service} outage {date}", "INC-{ref} Retrospective"],
        lexicon=[
            "mean time to detect",
            "contributing factor",
            "detection gap",
            "blast radius",
            "corrective action",
            "runbook",
            "paging policy",
            "error budget",
            "root cause",
            "containment",
            "indicator of compromise",
            "least privilege",
        ],
        templates=[
            "Incident {ref} affected the {service} service between 04:12 and 07:48 UTC on {date}.",
            "Customer impact: approximately {pct} of API requests returned 5xx responses for the duration.",
            "{term} was 34 minutes; alerting fired only after synthetic checks failed, indicating a {term}.",
            "The {term} was an unbounded retry loop introduced in the {project} deployment three days earlier.",
            "{term}: an expired credential, an absent circuit breaker, and a dashboard that aggregated away the signal.",
            "{term} was achieved by rolling back the deployment and draining the affected partition.",
            "No {term} were observed and no evidence of unauthorised data access was identified during review.",
            "{term} 1: add a bounded retry budget with jitter. Owner {person}, due {date}.",
            "{term} 2: extend the {term} to page on sustained elevated latency, not only on availability.",
            "The team's {term} for the quarter is now substantially consumed; feature freeze applies until reset.",
            "Access review confirmed that service credentials were scoped in line with {term}.",
        ],
    ),
    Category(
        key="source_code",
        display="Source code and configuration",
        owning_depts=["Engineering"],
        base_rate=0.08,
        policy=AclPolicy(
            label_tier=_p(internal=0.90, confidential=0.09, public=0.01),
            link_scope=_p(internal_link=0.52, none=0.46, domain_link=0.02),
            repo_type=_p(s3_bucket=0.42, confluence=0.20, smb_share=0.20, gdrive_shared=0.18),
            acl_origin=_p(inherited=0.92, explicit=0.07, mixed=0.01),
            group_grants=[
                ("dept:Engineering", 0.90),
                ("project:random", 0.44),
                ("all_employees", 0.07),
            ],
            n_direct=(0, 2),
            external_prob=0.05,
            external_org="Kestrel Cloud Systems",
            age_days=(2, 1400),
            staleness_days=(2, 800),
            pii_density=(0.5, 0.5),
            path_depth=(5, 11),
        ),
        title_patterns=["{service}/config/{project}.yaml", "{service} deployment manifest"],
        lexicon=[
            "environment variable",
            "connection string",
            "retry policy",
            "health probe",
            "resource limit",
            "service account",
            "ingress rule",
            "sidecar container",
            "config map",
            "rolling update",
            "readiness gate",
        ],
        templates=[
            "apiVersion: apps/v1 kind: Deployment metadata name: {service} namespace: {project}",
            "replicas: {n} strategy type: RollingUpdate maxUnavailable: 1 maxSurge: 2",
            "Container image pinned to the digest built from the {project} release branch.",
            "The {term} mounts non-secret configuration; credentials are injected by the platform at runtime.",
            "{term} requests 512Mi memory and 250m CPU with a limit of twice the request.",
            "{term}: HTTP GET /healthz initialDelaySeconds 15 periodSeconds 10 failureThreshold 3.",
            "The {term} is bound to a namespace-scoped role granting read access to config maps only.",
            "{term} restricts inbound traffic to the mesh gateway for the {service} hostname.",
            "A {term} performs log shipping and metric scraping alongside the primary process.",
            "{term} with exponential backoff capped at thirty seconds and five attempts.",
            "TODO: move the legacy {term} out of the manifest and into the secret store before GA.",
        ],
    ),
    # ------------------------------------------------------- Support & Mktg
    Category(
        key="support_ticket",
        display="Customer support case notes",
        owning_depts=["Support"],
        base_rate=0.10,
        policy=AclPolicy(
            label_tier=_p(internal=0.72, confidential=0.27, public=0.01),
            link_scope=_p(internal_link=0.49, none=0.47, domain_link=0.04),
            repo_type=_p(sharepoint=0.38, confluence=0.32, gdrive_shared=0.30),
            acl_origin=_p(inherited=0.88, explicit=0.11, mixed=0.01),
            group_grants=[("dept:Support", 0.91), ("dept:Engineering", 0.24)],
            n_direct=(0, 4),
            external_prob=0.06,
            external_org="Tandem Talent Partners",
            age_days=(1, 900),
            staleness_days=(1, 850),
            pii_density=(5.4, 2.0),
            path_depth=(4, 8),
        ),
        title_patterns=["Case {ref} - {counterparty}", "Escalation {ref}: {service}"],
        lexicon=[
            "severity",
            "reproduction steps",
            "workaround",
            "root cause pending",
            "customer sentiment",
            "entitlement",
            "response target",
            "escalation path",
            "known issue",
            "diagnostic bundle",
        ],
        templates=[
            "Case {ref} raised by {person} at {counterparty} on {date} against the {service} integration.",
            "{term}: 2 - significant business impact with a viable {term} in place.",
            "{term}: authenticate, call the export endpoint with a page size above 500, observe the timeout.",
            "A temporary {term} was provided: reduce page size to 200 and retry with backoff.",
            "The customer's {term} covers 24x7 support with a four-hour {term} for severity 2 cases.",
            "{term} is negative; this is the third occurrence reported by this account this quarter.",
            "Linked to {term} ENG-{ref} which is scheduled for the next maintenance release.",
            "A {term} was collected and attached; logs show connection pool exhaustion under load.",
            "{term} followed: assigned to the {service} on-call and raised with the account team.",
            "Customer contact details and account identifiers are recorded in the CRM, not in this note.",
        ],
    ),
    Category(
        key="marketing_brief",
        display="Marketing campaign briefs",
        owning_depts=["Marketing"],
        base_rate=0.06,
        policy=AclPolicy(
            # The deliberate low-sensitivity control: broad sharing here is
            # normal, so a scorer that simply flags "many principals" fails.
            label_tier=_p(public=0.46, internal=0.52, confidential=0.02),
            link_scope=_p(internal_link=0.44, domain_link=0.29, none=0.22, anyone_with_link=0.05),
            repo_type=_p(gdrive_shared=0.51, sharepoint=0.28, confluence=0.21),
            acl_origin=_p(inherited=0.87, explicit=0.12, mixed=0.01),
            group_grants=[("dept:Marketing", 0.92), ("all_employees", 0.46), ("dept:Sales", 0.55)],
            n_direct=(0, 5),
            external_prob=0.22,
            external_org="Tandem Talent Partners",
            age_days=(5, 800),
            staleness_days=(5, 700),
            pii_density=(0.4, 0.4),
            path_depth=(2, 5),
        ),
        title_patterns=["Campaign Brief: {project}", "{project} Launch Messaging"],
        lexicon=[
            "target persona",
            "value proposition",
            "channel mix",
            "creative concept",
            "call to action",
            "brand guardrail",
            "landing experience",
            "attribution window",
            "share of voice",
            "launch beat",
        ],
        templates=[
            "Campaign {project} runs from {date} across paid social, search and the partner newsletter.",
            "{term}: technical decision makers at mid-market organisations in {city} and surrounding regions.",
            "The core {term} centres on reduced time to value and lower operational overhead.",
            "{term} weights sixty percent to search, thirty to social and ten to sponsored content.",
            "{term}: a short-form video series featuring practitioner interviews from the {project} team.",
            "The primary {term} directs to a gated assessment tool hosted on the marketing site.",
            "{term} require that customer names are used only with written reference approval.",
            "The {term} is set to thirty days post-click for reporting consistency with prior campaigns.",
            "Budget of {money} is allocated for the initial eight-week flight, reviewed at the midpoint.",
            "Success is measured on qualified pipeline contribution rather than raw impression volume.",
        ],
    ),
]

CATEGORY_BY_KEY = {c.key: c for c in CATEGORIES}


def normalised_base_rates() -> np.ndarray:
    r = np.array([c.base_rate for c in CATEGORIES], dtype=float)
    return r / r.sum()
