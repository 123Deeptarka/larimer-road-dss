# -*- coding: utf-8 -*-
"""
Agent layer for the Road Maintenance Prioritization DSS.

A single tool-using agent (Anthropic Tool Runner) that answers natural-language
questions about the Larimer County network by calling deterministic tools.

DESIGN RULE — the agent orchestrates, it never computes.
Every number in an answer comes from a tool return value:

    query_segments      pandas filter over larimer_importance.csv
    get_segment         one row, verbatim
    grade_segment       Street View fetch + Sonnet 4.6 PSCI (Module 1, unchanged)
    compute_net_score   0.5*urgency + 0.5*importance, the same arithmetic as the app
    select_within_budget  exact 0/1 knapsack (DP), not a model judgement call

This keeps the scoring reproducible and auditable — the property the DSS is
supposed to have — while making it reachable in plain English. The agent adds
an interface, not a new scoring method.

Orchestration model is Opus 4.8; PSCI grading stays on Sonnet 4.6 because that
is the model the Model_5 benchmark measured.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import pandas as pd
from anthropic import beta_tool

import imagery
from dss_core import (
    PSCI_MAX,
    PSCI_MIN,
    classify,
    extract_grade,
    net_score,
    prepare_image,
    psci_to_urgency,
)
from psci_prompt import MODEL as PSCI_MODEL
from psci_prompt import build_user_content, system_prompt

AGENT_MODEL = "claude-opus-4-8"
MAX_TOOL_ITERATIONS = 24
DEFAULT_COST_PER_KM = 250_000.0     # placeholder resurfacing cost, $/km


# ---------------------------------------------------------------------------
# Run context — set once per agent run; tools read from it.
# ---------------------------------------------------------------------------
@dataclass
class AgentContext:
    df: pd.DataFrame
    anthropic_key: str
    svs_key: str = ""
    imp_col: str = "importance_pct"
    w_cond: float = 0.5
    w_imp: float = 0.5
    svs_pitch: int = imagery.DEFAULT_PITCH
    svs_fov: int = imagery.DEFAULT_FOV
    svs_radius: int = imagery.DEFAULT_RADIUS_M
    graded: dict = field(default_factory=dict)   # segment_id -> psci, cached


CTX: Optional[AgentContext] = None


def _ctx() -> AgentContext:
    if CTX is None:
        raise RuntimeError("AgentContext not set — call run_agent()")
    return CTX


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@beta_tool
def query_segments(route_contains: str = "", tier: str = "", min_importance: float = 0.0,
                   max_importance: float = 1.0, terrain: str = "",
                   sort_by: str = "importance", descending: bool = True,
                   limit: int = 10) -> str:
    """Search the precomputed road-importance table and return matching segments.

    Use this to find candidates before grading or scoring anything. Importance
    is network importance only — it says nothing about pavement condition.

    Args:
        route_contains: Case-insensitive substring of the route name, e.g. "COLLEGE". Empty matches all.
        tier: Filter to one of "Highway", "Major", "Local". Empty matches all.
        min_importance: Lower bound on the importance column, 0-1.
        max_importance: Upper bound on the importance column, 0-1.
        terrain: Filter on TERRAIN, e.g. "Mountainous", "Rolling", "Plains". Empty matches all.
        sort_by: Column to sort on: "importance", "aadt_filled", "length_m", or "ebc_norm".
        descending: Sort descending when True.
        limit: Maximum rows to return, 1-50.
    """
    ctx = _ctx()
    d = ctx.df
    if route_contains:
        d = d[d["route"].astype(str).str.contains(route_contains, case=False, na=False)]
    if tier:
        tiers = {t.lower() for t in d["tier"].dropna().unique()}
        if tier.strip().lower() not in tiers:
            return json.dumps({"error": f"unknown tier '{tier}'",
                               "available": sorted(tiers)})
        d = d[d["tier"].astype(str).str.lower() == tier.strip().lower()]
    if terrain:
        # Never silently drop an unsatisfiable filter — returning unfiltered
        # rows would let the caller report them as if they matched.
        if "TERRAIN" not in d.columns:
            return json.dumps({"error": "terrain filter unavailable: the importance "
                                        "table has no TERRAIN column. Re-run "
                                        "add_coordinates.py to add it."})
        vals = {str(v).lower() for v in d["TERRAIN"].dropna().unique()}
        if terrain.strip().lower() not in vals:
            return json.dumps({"error": f"unknown terrain '{terrain}'",
                               "available": sorted(vals),
                               "note": "TERRAIN is recorded on Highway-tier segments only"})
        d = d[d["TERRAIN"].astype(str).str.lower() == terrain.strip().lower()]
    col = ctx.imp_col
    d = d[(d[col] >= min_importance) & (d[col] <= max_importance)]

    sort_col = {"importance": col}.get(sort_by, sort_by)
    if sort_col not in d.columns:
        return json.dumps({"error": f"unknown sort_by '{sort_by}'",
                           "available": ["importance", "aadt_filled", "length_m", "ebc_norm"]})

    d = d.sort_values(sort_col, ascending=not descending).head(max(1, min(int(limit), 50)))
    cols = [c for c in ["segment_id", "route", "tier", "aadt_filled", "length_m",
                        "ebc_norm", col] if c in d.columns]
    return json.dumps({"matched": int(len(d)), "segments": d[cols].round(4).to_dict("records")})


@beta_tool
def get_segment(segment_id: str) -> str:
    """Return the complete record for one segment, including coordinates.

    Args:
        segment_id: Exact segment identifier, e.g. "H_35907_0".
    """
    d = _ctx().df
    row = d[d["segment_id"] == segment_id]
    if row.empty:
        return json.dumps({"error": f"no segment '{segment_id}'"})
    rec = row.iloc[0].to_dict()
    return json.dumps({k: (None if pd.isna(v) else v) for k, v in rec.items()}, default=str)


@beta_tool
def grade_segment(segment_id: str, n_images: int = 1) -> str:
    """Fetch Street View imagery for a segment and grade its pavement PSCI 1-10.

    PSCI 10 = newly surfaced, 1 = completely failed. This calls Claude Sonnet 4.6
    with the Xu et al. (2025) grading prompt — the same pipeline the benchmark
    measured. Results are cached per segment within a run.

    Costs a Street View request and a model call, so grade only segments you
    actually intend to score.

    Args:
        segment_id: Exact segment identifier.
        n_images: How many camera headings to sample and average, 1-4.
    """
    ctx = _ctx()
    if segment_id in ctx.graded:
        return json.dumps({"segment_id": segment_id, "psci": ctx.graded[segment_id],
                           "cached": True})
    if not ctx.svs_key:
        return json.dumps({"error": "no Google Maps API key configured; cannot fetch imagery"})

    row = ctx.df[ctx.df["segment_id"] == segment_id]
    if row.empty:
        return json.dumps({"error": f"no segment '{segment_id}'"})
    seg = row.iloc[0]
    if pd.isna(seg.get("lat")) or pd.isna(seg.get("lon")):
        return json.dumps({"error": f"segment '{segment_id}' has no coordinates"})

    lat, lon, bearing = float(seg["lat"]), float(seg["lon"]), float(seg["bearing"])
    n = max(1, min(int(n_images), 4))
    headings = [bearing] if n == 1 else [(bearing + i * 360.0 / n) % 360.0 for i in range(n)]

    client = anthropic.Anthropic(api_key=ctx.anthropic_key)
    grades, notes = [], []
    for h in headings:
        try:
            raw, _, meta = imagery.streetview_fetch(
                lat, lon, ctx.svs_key, heading=h, pitch=ctx.svs_pitch,
                fov=ctx.svs_fov, radius_m=ctx.svs_radius)
        except Exception as e:                     # noqa: BLE001
            return json.dumps({"error": f"Street View request failed: {e}"})
        if raw is None:
            return json.dumps({"error": f"no Street View panorama within "
                                        f"{ctx.svs_radius} m (status {meta.get('status')})"})
        b64, media = prepare_image(raw, "svs.jpg")
        resp = client.messages.create(
            model=PSCI_MODEL, max_tokens=1000, system=system_prompt,
            messages=[{"role": "user", "content": build_user_content(b64, media)}],
        )
        reply = "".join(b.text for b in resp.content if b.type == "text").strip()
        g = extract_grade(reply)
        if PSCI_MIN <= g <= PSCI_MAX:
            grades.append(g)
        else:
            notes.append(f"heading {h:.0f}deg returned an unparseable grade")

    if not grades:
        return json.dumps({"error": "no image returned a parseable PSCI grade", "notes": notes})

    psci = sum(grades) / len(grades)
    ctx.graded[segment_id] = psci
    return json.dumps({"segment_id": segment_id, "psci": round(psci, 2),
                       "per_image": grades, "n_images": len(grades),
                       "notes": notes, "scale": "10 = best, 1 = failed"})


@beta_tool
def compute_net_score(segment_id: str, psci: float) -> str:
    """Fuse a PSCI grade with the segment's network importance into a net score.

    net_score = w_cond * urgency + w_imp * importance, where
    urgency = 1 - (PSCI - 1) / 9. Weights come from the app's sidebar.

    Args:
        segment_id: Exact segment identifier.
        psci: Pavement condition grade 1-10 from grade_segment.
    """
    ctx = _ctx()
    row = ctx.df[ctx.df["segment_id"] == segment_id]
    if row.empty:
        return json.dumps({"error": f"no segment '{segment_id}'"})
    if not (PSCI_MIN <= psci <= PSCI_MAX):
        return json.dumps({"error": f"psci must be {PSCI_MIN}-{PSCI_MAX}, got {psci}"})

    seg = row.iloc[0]
    importance = float(seg[ctx.imp_col])
    urgency = psci_to_urgency(psci)
    score = net_score(urgency, importance, ctx.w_cond, ctx.w_imp)
    tier, _, _ = classify(score)
    return json.dumps({
        "segment_id": segment_id, "route": str(seg["route"]),
        "psci": round(psci, 2), "urgency": round(urgency, 4),
        "importance": round(importance, 4), "importance_column": ctx.imp_col,
        "w_condition": ctx.w_cond, "w_importance": ctx.w_imp,
        "net_score": round(score, 4), "priority_tier": tier,
    })


@beta_tool
def select_within_budget(segment_ids: list[str], net_scores: list[float],
                         budget_dollars: float,
                         cost_per_km: float = DEFAULT_COST_PER_KM) -> str:
    """Choose the highest-total-priority set of segments that fits a budget.

    This is an exact 0/1 knapsack solved by dynamic programming — it returns the
    provably optimal subset, not an estimate. Cost per segment is its length
    times cost_per_km, which is a placeholder: real treatment costs vary by
    surface, width, and treatment type.

    Args:
        segment_ids: Candidate segment identifiers.
        net_scores: Net score for each candidate, same order and length.
        budget_dollars: Total available budget in dollars.
        cost_per_km: Assumed treatment cost per kilometre, in dollars.
    """
    ctx = _ctx()
    if len(segment_ids) != len(net_scores):
        return json.dumps({"error": "segment_ids and net_scores must be the same length"})
    if not segment_ids:
        return json.dumps({"error": "no candidates supplied"})

    lengths = ctx.df.set_index("segment_id")["length_m"]
    items = []
    for sid, sc in zip(segment_ids, net_scores):
        if sid not in lengths.index:
            return json.dumps({"error": f"no segment '{sid}'"})
        cost = float(lengths.loc[sid]) / 1000.0 * cost_per_km
        items.append((sid, float(sc), cost))

    # DP over cost discretised to $1k buckets — exact on the rounded costs.
    BUCKET = 1000.0
    cap = int(budget_dollars // BUCKET)
    if cap <= 0:
        return json.dumps({"error": "budget too small for any segment"})
    weights = [max(1, int(round(c / BUCKET))) for _, _, c in items]

    best = [0.0] * (cap + 1)
    keep = [[False] * (cap + 1) for _ in items]
    for i, (w, (_, val, _)) in enumerate(zip(weights, items)):
        for cpt in range(cap, w - 1, -1):
            cand = best[cpt - w] + val
            if cand > best[cpt]:
                best[cpt] = cand
                keep[i][cpt] = True

    chosen, cpt = [], cap
    for i in range(len(items) - 1, -1, -1):
        if keep[i][cpt]:
            chosen.append(items[i])
            cpt -= weights[i]

    chosen.reverse()
    spend = sum(c for _, _, c in chosen)
    return json.dumps({
        "selected": [{"segment_id": s, "net_score": round(v, 4), "cost": round(c)}
                     for s, v, c in chosen],
        "n_selected": len(chosen), "n_candidates": len(items),
        "total_cost": round(spend), "budget": budget_dollars,
        "remaining": round(budget_dollars - spend),
        "total_priority": round(sum(v for _, v, _ in chosen), 4),
        "method": "exact 0/1 knapsack, costs rounded to $1k buckets",
        "cost_basis": f"length_km * ${cost_per_km:,.0f}/km (placeholder)",
    })


TOOLS = [query_segments, get_segment, grade_segment, compute_net_score,
         select_within_budget]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
AGENT_SYSTEM = """You are an analyst for a road-maintenance decision support system covering Larimer County, Colorado.

You answer questions by calling tools. Never state a number you did not get back from a tool — do not estimate, interpolate, or recall figures from earlier in the conversation as if they were fresh measurements.

What the data means:
- Network importance is precomputed from AADT, edge betweenness centrality, and NHS designation. It reflects how structurally important a segment is to the network. It says nothing about pavement condition.
- PSCI is a pavement condition grade from 1 to 10, where 10 is newly surfaced and 1 is completely failed. It comes from grading a Street View image and is only available after you call grade_segment.
- Net score fuses the two. It requires a PSCI, so a segment that has not been graded has no net score.

How to work:
- Find candidates with query_segments before grading anything. Grading costs a Street View request and a model call, so grade only what you need.
- To rank by priority you must grade each candidate, then call compute_net_score for each. Say so if that would take many calls, and confirm the scope before doing more than about eight.
- Use select_within_budget for budget questions. It is an exact optimiser; do not attempt the selection yourself.

What to flag, unprompted, when it bears on the answer:
- 694 of 23,190 segments sit outside the largest connected component and have no betweenness, so their importance comes from AADT and NHS only.
- Most local roads have no measured AADT and were floored to 100 vehicles/day.
- The PSCI model was benchmarked on close-up road-surface photographs. Street View is wide-angle and oblique, so grades from it are outside the benchmark's domain and less reliable than the reported accuracy implies.
- The tier cutoffs were calibrated against a different condition distribution and over-flag under the current weighting.
- Treatment costs are a placeholder of $250,000 per kilometre, not real estimates.

Answer in prose, concisely, with the specific numbers the tools returned. Report failures plainly — if imagery is unavailable for a segment, say so rather than substituting another segment silently."""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_agent(question, ctx, history=None, on_event=None):
    """Run the agent to completion. Returns (answer_text, updated_history).

    on_event(kind, payload) is called as work happens, for UI rendering:
      ("text", str) ("tool_use", {name, input}) ("tool_result", {name, content})
    """
    global CTX
    CTX = ctx

    client = anthropic.Anthropic(api_key=ctx.anthropic_key)
    messages = list(history or []) + [{"role": "user", "content": question}]

    runner = client.beta.messages.tool_runner(
        model=AGENT_MODEL,
        max_tokens=8000,
        system=AGENT_SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        tools=TOOLS,
        messages=messages,
        max_iterations=MAX_TOOL_ITERATIONS,
    )

    answer = []
    for message in runner:
        for block in message.content:
            if block.type == "text" and block.text.strip():
                answer.append(block.text)
                if on_event:
                    on_event("text", block.text)
            elif block.type == "tool_use":
                if on_event:
                    on_event("tool_use", {"name": block.name, "input": block.input})

        messages.append({"role": "assistant", "content": message.content})
        tool_response = runner.generate_tool_call_response()
        if tool_response is None:
            break
        messages.append(tool_response)
        if on_event:
            for blk in tool_response["content"]:
                if getattr(blk, "type", None) == "tool_result" or (
                        isinstance(blk, dict) and blk.get("type") == "tool_result"):
                    content = blk["content"] if isinstance(blk, dict) else blk.content
                    on_event("tool_result", {"content": content})

    return "\n\n".join(answer), messages
