"""Deterministic entity pools.

Deliberately dependency-free (no Faker) so a clean checkout can regenerate the
exact corpus the published metrics were computed on. Variety comes from
combination, not from a large word list.
"""

from __future__ import annotations

import numpy as np

FIRST_NAMES = [
    "Aarav",
    "Priya",
    "Wei",
    "Sofia",
    "Diego",
    "Amara",
    "Yusuf",
    "Hana",
    "Liam",
    "Nadia",
    "Kwame",
    "Ingrid",
    "Mateo",
    "Ling",
    "Farah",
    "Tomas",
    "Anika",
    "Oskar",
    "Rania",
    "Ravi",
    "Elena",
    "Jonas",
    "Meera",
    "Andre",
    "Chiara",
    "Bo",
    "Zara",
    "Hugo",
    "Lucia",
    "Kenji",
    "Nour",
    "Petra",
    "Samir",
    "Freya",
    "Idris",
    "Camila",
    "Viktor",
    "Aisha",
    "Emil",
    "Tara",
    "Rohan",
    "Sanne",
    "Marek",
    "Leila",
    "Noah",
    "Sinead",
    "Takumi",
    "Ada",
    "Bram",
    "Yara",
]

LAST_NAMES = [
    "Sharma",
    "Okafor",
    "Zhang",
    "Novak",
    "Fernandes",
    "Haddad",
    "Lindqvist",
    "Duarte",
    "Kaur",
    "Moreau",
    "Bianchi",
    "Nakamura",
    "Oyelaran",
    "Kowalski",
    "Silva",
    "Ahmadi",
    "Vasquez",
    "Bergstrom",
    "Iyer",
    "Dubois",
    "Rossi",
    "Mensah",
    "Petrov",
    "Kim",
    "Almeida",
    "Hoffmann",
    "Rahman",
    "Larsen",
    "Costa",
    "Weber",
    "Nguyen",
    "Marchetti",
    "Sorensen",
    "Banerjee",
    "Toure",
    "Jansen",
    "Ferreira",
    "Krause",
    "Osei",
    "Volkov",
]

DEPARTMENTS = [
    "Engineering",
    "Finance",
    "Legal",
    "People",
    "Sales",
    "Executive",
    "Marketing",
    "Support",
    "Security",
]

TITLES = {
    "Engineering": [
        "Software Engineer",
        "Staff Engineer",
        "Engineering Manager",
        "SRE",
        "Data Engineer",
    ],
    "Finance": [
        "Financial Analyst",
        "Controller",
        "AP Specialist",
        "FP&A Manager",
        "Treasury Analyst",
    ],
    "Legal": ["Counsel", "Contracts Manager", "Paralegal", "Compliance Lead", "General Counsel"],
    "People": ["Recruiter", "HR Business Partner", "People Ops Lead", "Compensation Analyst"],
    "Sales": ["Account Executive", "Sales Engineer", "SDR", "Regional Director"],
    "Executive": [
        "Chief Executive Officer",
        "Chief Financial Officer",
        "Chief Technology Officer",
        "Chief Operating Officer",
        "Chief People Officer",
    ],
    "Marketing": ["Product Marketing Manager", "Content Strategist", "Demand Gen Manager"],
    "Support": ["Support Engineer", "Escalation Manager", "Technical Account Manager"],
    "Security": ["Security Engineer", "Detection Engineer", "GRC Analyst", "Security Manager"],
}

COMPANY = "Northwind Grid"
COMPANY_DOMAIN = "northwindgrid.com"

PARTNER_ORGS = [
    ("Halloran & Vetch LLP", "halloranvetch.com", "outside counsel"),
    ("Brightmoor Audit Group", "brightmoor-audit.com", "external auditor"),
    ("Kestrel Cloud Systems", "kestrelcloud.io", "infrastructure vendor"),
    ("Tandem Talent Partners", "tandemtalent.co", "contractor agency"),
]

COUNTERPARTIES = [
    "Arclight Logistics",
    "Verado Health",
    "Pinnacle Foundry",
    "Sunbelt Retail Group",
    "Corvid Analytics",
    "Meridian Freight",
    "Bluewater Utilities",
    "Ostara Biotech",
    "Ferrous Materials Co",
    "Lantern Media",
    "Skyward Aviation",
    "Harborline Shipping",
]

PROJECTS = [
    "Ridgeline",
    "Kettleback",
    "Marlowe",
    "Ptarmigan",
    "Verdigris",
    "Aldercroft",
    "Tessellate",
    "Bramblewick",
    "Halyard",
    "Quillon",
    "Sablefish",
    "Windrose",
]

CITIES = [
    "Bengaluru",
    "Amsterdam",
    "Austin",
    "Toronto",
    "Singapore",
    "Dublin",
    "Munich",
    "Sao Paulo",
    "Manchester",
    "Yokohama",
]

SERVICES = [
    "ingest-gateway",
    "billing-ledger",
    "auth-broker",
    "search-indexer",
    "notification-relay",
    "tenant-router",
    "metrics-collector",
    "policy-engine",
]


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def person_name(rng: np.random.Generator) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def email_for(name: str, domain: str) -> str:
    first, _, last = name.partition(" ")
    return f"{first.lower()}.{last.lower()}@{domain}"


def money(rng: np.random.Generator, lo: int = 5_000, hi: int = 4_000_000) -> str:
    v = int(rng.integers(lo, hi))
    return f"${v:,}"


def date_str(rng: np.random.Generator, year_lo: int = 2023, year_hi: int = 2026) -> str:
    y = int(rng.integers(year_lo, year_hi + 1))
    m = int(rng.integers(1, 13))
    d = int(rng.integers(1, 29))
    return f"{y:04d}-{m:02d}-{d:02d}"
