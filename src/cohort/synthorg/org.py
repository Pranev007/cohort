"""The synthetic organisation: people, nested groups, and external partners.

Nested groups matter. A document granted to `grp-all-employees` is reachable by
2,000 principals; one granted to `grp-legal-contracts` by nine. Any posture
feature that counts "who can see this" has to expand the group graph first,
which is exactly the step real DSPM tools get wrong when they read raw ACLs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cohort.synthorg import names as N


@dataclass(slots=True)
class Person:
    person_id: str
    name: str
    email: str
    dept: str
    title: str
    domain: str
    is_external: bool = False
    partner_org: str | None = None
    tenure_days: int = 0
    is_departing: bool = False


@dataclass(slots=True)
class Group:
    group_id: str
    name: str
    kind: str  # dept | project | all_company | external | leadership
    member_ids: list[str] = field(default_factory=list)
    child_group_ids: list[str] = field(default_factory=list)


@dataclass
class Organisation:
    people: dict[str, Person]
    groups: dict[str, Group]
    dept_groups: dict[str, str]
    project_groups: list[str]
    all_company_group: str
    leadership_group: str
    external_groups: dict[str, str]
    #: Memoised transitive expansions. `All Employees` resolves to every
    #: department group and then to 2,000 people; recomputing that per document
    #: dominates generation time.
    _expand_cache: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    # -- group expansion ---------------------------------------------------
    def expand_group(self, group_id: str, _seen: set[str] | None = None) -> set[str]:
        """Resolve a group to its transitive member set (cycle-safe)."""
        cached = self._expand_cache.get(group_id)
        if cached is not None:
            return set(cached)

        seen = _seen if _seen is not None else set()
        if group_id in seen or group_id not in self.groups:
            return set()
        seen.add(group_id)
        g = self.groups[group_id]
        members = set(g.member_ids)
        for child in g.child_group_ids:
            members |= self.expand_group(child, seen)

        # Only cache a full, un-truncated expansion (i.e. the top-level call).
        if _seen is None:
            self._expand_cache[group_id] = frozenset(members)
        return members

    def resolve(self, group_ids: list[str], direct_ids: list[str]) -> set[str]:
        """Effective principal set for an ACL of groups + direct grants."""
        out: set[str] = set(direct_ids)
        for gid in group_ids:
            out |= self.expand_group(gid)
        return out

    def dept_of(self, person_id: str) -> str:
        p = self.people.get(person_id)
        return p.dept if p else "Unknown"

    def domains_of(self, person_ids: set[str]) -> set[str]:
        return {self.people[p].domain for p in person_ids if p in self.people}

    def internal_people(self) -> list[Person]:
        return [p for p in self.people.values() if not p.is_external]


def build_organisation(rng: np.random.Generator, n_employees: int, n_partners: int) -> Organisation:
    people: dict[str, Person] = {}
    groups: dict[str, Group] = {}

    # Weight departments so Engineering/Sales dominate headcount, as they do
    # in a real mid-size company. Executive stays tiny.
    dept_weights = np.array([0.30, 0.09, 0.05, 0.06, 0.18, 0.01, 0.08, 0.15, 0.08])
    dept_weights = dept_weights / dept_weights.sum()

    for i in range(n_employees):
        dept = str(rng.choice(N.DEPARTMENTS, p=dept_weights))
        name = N.person_name(rng)
        pid = f"usr-{i:05d}"
        people[pid] = Person(
            person_id=pid,
            name=name,
            email=N.email_for(f"{name} {i}", N.COMPANY_DOMAIN).replace(f" {i}", ""),
            dept=dept,
            title=str(rng.choice(N.TITLES[dept])),
            domain=N.COMPANY_DOMAIN,
            tenure_days=int(rng.integers(20, 3200)),
            # ~3% of staff are inside a notice period at any given time.
            is_departing=bool(rng.random() < 0.03),
        )

    # External partner staff.
    external_groups: dict[str, str] = {}
    for k in range(min(n_partners, len(N.PARTNER_ORGS))):
        org_name, domain, _role = N.PARTNER_ORGS[k]
        gid = f"grp-ext-{k}"
        member_ids = []
        for j in range(int(rng.integers(4, 12))):
            name = N.person_name(rng)
            pid = f"ext-{k}-{j:03d}"
            people[pid] = Person(
                person_id=pid,
                name=name,
                email=N.email_for(name, domain),
                dept="External",
                title="Partner Contact",
                domain=domain,
                is_external=True,
                partner_org=org_name,
                tenure_days=int(rng.integers(30, 1500)),
            )
            member_ids.append(pid)
        groups[gid] = Group(gid, f"{org_name} (external)", "external", member_ids)
        external_groups[org_name] = gid

    # Department groups.
    dept_groups: dict[str, str] = {}
    for dept in N.DEPARTMENTS:
        gid = f"grp-dept-{dept.lower()}"
        members = [p.person_id for p in people.values() if p.dept == dept and not p.is_external]
        groups[gid] = Group(gid, f"{dept} (all)", "dept", members)
        dept_groups[dept] = gid

    # Narrow functional sub-groups — the "correct" grant for sensitive material.
    for dept, suffix, frac in [
        ("Legal", "contracts", 0.5),
        ("Finance", "controllers", 0.3),
        ("People", "comp", 0.35),
        ("Security", "irt", 0.45),
    ]:
        parent = groups[dept_groups[dept]]
        if not parent.member_ids:
            continue
        k = max(2, int(len(parent.member_ids) * frac))
        chosen = list(
            rng.choice(parent.member_ids, size=min(k, len(parent.member_ids)), replace=False)
        )
        gid = f"grp-{dept.lower()}-{suffix}"
        groups[gid] = Group(gid, f"{dept} {suffix}", "dept", [str(c) for c in chosen])
        dept_groups[f"{dept}:{suffix}"] = gid

    # Cross-functional project groups.
    project_groups: list[str] = []
    for proj in N.PROJECTS:
        gid = f"grp-proj-{proj.lower()}"
        pool = [p.person_id for p in people.values() if not p.is_external]
        k = int(rng.integers(8, 40))
        members = [str(m) for m in rng.choice(pool, size=min(k, len(pool)), replace=False)]
        groups[gid] = Group(gid, f"Project {proj}", "project", members)
        project_groups.append(gid)

    # Leadership.
    leadership = [p.person_id for p in people.values() if p.dept == "Executive"]
    leadership += [
        p.person_id
        for p in people.values()
        if "Manager" in p.title or "Director" in p.title or "Lead" in p.title
    ]
    groups["grp-leadership"] = Group(
        "grp-leadership", "Leadership", "leadership", sorted(set(leadership))
    )

    # All-company: a *nested* group of every department group. Anything granted
    # here is effectively public internally — the classic oversharing vector.
    groups["grp-all-employees"] = Group(
        "grp-all-employees",
        "All Employees",
        "all_company",
        member_ids=[],
        child_group_ids=list(dept_groups[d] for d in N.DEPARTMENTS),
    )

    return Organisation(
        people=people,
        groups=groups,
        dept_groups=dept_groups,
        project_groups=project_groups,
        all_company_group="grp-all-employees",
        leadership_group="grp-leadership",
        external_groups=external_groups,
    )
