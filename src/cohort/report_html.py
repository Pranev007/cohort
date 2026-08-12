"""Self-contained HTML scan report.

Not a web application — a generated deliverable. One file, no build step, no
external requests, no server: open it in a browser or attach it to a ticket. That
is deliberate. A DSPM finding has to survive being emailed to someone who will
never run the tool, and a report that needs `npm install` to read is not a report.

Everything is inlined: styles, the small amount of filtering JavaScript, and the
data itself. It renders in both light and dark themes via `prefers-color-scheme`.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

_CSS = """
:root{--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--card:#f6f8fa;
--peer:#0072B2;--warn:#D55E00;--ok:#009E73;--chip:#eaeef2}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;
--line:#3d444d;--card:#151b23;--chip:#212830}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:17px;margin:40px 0 12px}
.sub{color:var(--muted);font-size:14px;margin:0 0 28px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:0 0 8px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.stat .v{font-size:22px;font-weight:640;letter-spacing:-.02em}
.stat .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 14px}
input,select{background:var(--bg);color:var(--fg);border:1px solid var(--line);
border-radius:7px;padding:8px 11px;font:inherit;font-size:14px}
input{flex:1;min-width:220px}
.finding{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--peer);
border-radius:8px;padding:16px 18px;margin:0 0 12px}
.finding.flagged{border-left-color:var(--warn)}
.fhead{display:flex;justify-content:space-between;gap:16px;align-items:baseline;flex-wrap:wrap}
.fid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:var(--muted)}
.ftitle{font-weight:600}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 0}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:11.5px;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted)}
.why,.fix{margin:12px 0 0;font-size:14px}
.why{color:var(--fg)} .fix{color:var(--muted);padding-left:12px;border-left:2px solid var(--ok)}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
font-weight:600;display:block;margin-bottom:3px}
.risk{font-variant-numeric:tabular-nums;font-weight:640}
.none{color:var(--muted);padding:28px;text-align:center}
footer{color:var(--muted);font-size:12.5px;margin-top:48px;padding-top:18px;border-top:1px solid var(--line)}
"""

_JS = """
const q=document.getElementById('q'),sel=document.getElementById('coh'),
      only=document.getElementById('only'),cards=[...document.querySelectorAll('.finding')],
      count=document.getElementById('count');
function apply(){
  const t=q.value.toLowerCase(), c=sel.value, f=only.checked;
  let n=0;
  for(const el of cards){
    const okT=!t||el.dataset.search.includes(t);
    const okC=!c||el.dataset.cohort===c;
    const okF=!f||el.dataset.flagged==='1';
    const show=okT&&okC&&okF;
    el.style.display=show?'':'none';
    if(show)n++;
  }
  count.textContent=n;
  document.getElementById('empty').style.display=n?'none':'';
}
[q,sel].forEach(e=>e.addEventListener('input',apply));
only.addEventListener('change',apply);
"""


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def render_report(
    findings: pd.DataFrame,
    stats: dict,
    cohorts: list[dict],
    timings: dict[str, float] | None = None,
) -> str:
    """Build the full HTML document as a string."""
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    tiles = [
        ("Documents scanned", f"{stats.get('n_documents', 0):,}"),
        ("Cohorts discovered", stats.get("n_cohorts", 0)),
        ("Flagged", f"{stats.get('n_flagged', 0):,}"),
        ("Flag rate", f"{stats.get('flag_rate', 0):.2%}"),
        ("Unassigned", f"{stats.get('unassigned_rate', 0):.2%}"),
        ("Throughput", f"{stats.get('docs_per_s_end_to_end', 0):,.0f} docs/s"),
    ]
    stat_html = "".join(
        f'<div class="stat"><div class="v">{_esc(v)}</div><div class="k">{_esc(k)}</div></div>'
        for k, v in tiles
    )

    def _chips(terms: list[str]) -> str:
        # The UNASSIGNED cohort has no distinguishing terms by definition;
        # rendering an empty pill for it looks like a bug.
        return "".join(f"<span class='chip'>{_esc(t)}</span>" for t in terms[:4])

    cohort_rows = "".join(
        f"<tr><td class='num'>{_esc(c['cohort_id'])}</td><td>{_esc(c['label'])}</td>"
        f"<td class='num'>{c['size']:,}</td>"
        f"<td><div class='chips'>{_chips(c['keyphrases'])}</div></td></tr>"
        for c in cohorts
    )

    options = "".join(
        f"<option value='{_esc(c['label'])}'>{_esc(c['label'])} ({c['size']:,})</option>"
        for c in cohorts
        if c["size"] > 0
    )

    cards = []
    for _, r in findings.iterrows():
        flagged = bool(r.get("is_flagged"))
        chips = "".join(
            f"<span class='chip'>{_esc(part.strip())}</span>"
            for part in str(r.get("top_features", "")).split(",")
            if part.strip()
        )
        remediation = str(r.get("remediation") or "")
        fix = (
            f"<div class='fix'><span class='lbl'>Remediation</span>{_esc(remediation)}</div>"
            if remediation
            else ""
        )
        search = " ".join(
            str(r.get(k, ""))
            for k in ("doc_id", "title", "cohort_name", "narrative", "top_features")
        ).lower()
        cards.append(
            f"<div class='finding{' flagged' if flagged else ''}' "
            f'data-cohort="{_esc(r.get("cohort_name", ""))}" '
            f'data-flagged="{"1" if flagged else "0"}" '
            f'data-search="{_esc(search)}">'
            f"<div class='fhead'><div><div class='ftitle'>{_esc(r.get('title', ''))}</div>"
            f"<div class='fid'>{_esc(r.get('doc_id', ''))} &middot; {_esc(r.get('cohort_name', ''))}</div></div>"
            f"<div class='risk'>{float(r.get('risk_score', 0)):.2f} nats"
            f"<span class='fid'> &middot; p={float(r.get('conformal_p', 1)):.4f}</span></div></div>"
            f"<div class='chips'>{chips}</div>"
            f"<div class='why'><span class='lbl'>Why</span>{_esc(r.get('narrative', ''))}</div>"
            f"{fix}</div>"
        )

    timing_note = ""
    if timings:
        parts = " &middot; ".join(f"{k} {v:.1f}s" for k, v in timings.items() if k != "total")
        timing_note = f"<br>Stage timings: {parts}"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cohort scan report</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Cohort &mdash; scan report</h1>
<p class="sub">Peer-baseline anomaly detection over unstructured data &middot; generated {generated}</p>

<div class="stats">{stat_html}</div>

<h2>Discovered cohorts</h2>
<p class="sub">Recovered from document meaning alone &mdash; no category list, no rules.</p>
<table><thead><tr><th class="num">id</th><th>label</th><th class="num">size</th>
<th>distinguishing terms</th></tr></thead><tbody>{cohort_rows}</tbody></table>

<h2>Findings <span class="fid">(<span id="count">{len(findings)}</span> shown)</span></h2>
<p class="sub">Ranked by risk. Each explanation is an exact decomposition of the score,
and each remediation was re-scored to verify the residual risk it claims.</p>
<div class="controls">
  <input id="q" placeholder="Search document, cohort, or explanation&hellip;">
  <select id="coh"><option value="">All cohorts</option>{options}</select>
  <label class="sub" style="margin:0;display:flex;align-items:center;gap:6px">
    <input type="checkbox" id="only" style="flex:none;width:auto"> flagged only</label>
</div>
{"".join(cards)}
<div class="none" id="empty" style="display:none">No findings match the current filter.</div>

<footer>
Generated by <strong>cohort</strong> &mdash; independent reimplementation of peer-baseline
risk analysis (cf. Concentric AI's Risk Distance&trade;). Not affiliated with Concentric AI.
Synthetic corpus; no real personal data.{timing_note}
</footer>
</div><script>{_JS}</script></body></html>"""


def write_report(
    findings: pd.DataFrame,
    stats: dict,
    cohorts_path: Path,
    out_path: Path,
    timings: dict[str, float] | None = None,
) -> Path:
    cohorts = json.loads(cohorts_path.read_text(encoding="utf-8")) if cohorts_path.exists() else []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(findings, stats, cohorts, timings), encoding="utf-8")
    return out_path
