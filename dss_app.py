# -*- coding: utf-8 -*-
"""
Road Maintenance Prioritization DSS — Larimer County, Colorado
==============================================================

Fuses two independently-computed halves into one Net Score per road segment:

  Module 1 (condition)  — Claude Sonnet 4.6 grades a street-level image on the
                          Xu et al. (2025) PSCI 1-10 scale, using the identical
                          prompt benchmarked in Model_5.
  Module 2 (importance) — precomputed network importance from
                          road_importance.py (AADT + edge betweenness + NHS),
                          read from larimer_importance.csv. Nothing is computed
                          here: edge betweenness on 18,486 nodes takes ~34 min
                          and cannot run inside a web request.

  Net Score = W_COND * urgency + W_IMP * importance
  urgency   = 1 - (PSCI - 1) / 9      ->  1.0 = failed pavement, 0.0 = new

Run:  streamlit run dss_app.py
"""

import base64
import io
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import imagery
from dss_core import (
    JPEG_QUALITY,
    MAX_IMAGE_PX,
    PSCI_MAX,
    PSCI_MIN,
    TIERS,
    UNPARSEABLE,
    classify,
    extract_grade,
    net_score,
    prepare_image,
    psci_to_urgency,
)
from psci_prompt import MODEL, build_user_content, system_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
IMPORTANCE_CSV_CANDIDATES = [
    APP_DIR / "larimer_importance_outputs" / "larimer_importance.csv",
    APP_DIR / "larimer_importance.csv",
]

DEFAULT_W_COND = 0.50
DEFAULT_W_IMP = 0.50

# Dark-mode categorical slots, used for the two contributions to the net score
# and for accent text. Every one of these clears 4.5:1 body contrast on the
# #0d0d0d page plane (measured, not assumed).
C_COND = "#3987e5"      # slot 1 blue
C_IMP = "#d95926"       # slot 2 orange
C_AQUA = "#199e70"      # slot 3
C_YELLOW = "#c98500"    # slot 4
C_VIOLET = "#9085e9"    # slot 7
INK_2 = "#c3c2b7"       # secondary ink

st.set_page_config(page_title="Road Prioritization DSS — Larimer County",
                   page_icon="🛣️", layout="wide",
                   initial_sidebar_state="expanded")

# Dark surface throughout (see .streamlit/config.toml). Accent colour lives on
# LABELS, headings and marks; numeric VALUES stay in primary ink so colour never
# looks like it encodes the number.
st.markdown(f"""
<style>
  /* Streamlit sizes almost everything in rem, so raising the root size scales
     the whole UI proportionally instead of fighting individual selectors. */
  html {{ font-size:18px; }}

  .dss-h        {{ font-size:1.15rem; font-weight:700; letter-spacing:.01em;
                   margin:.2rem 0 .7rem; display:flex; align-items:center;
                   gap:.55rem; }}
  .dss-h-num    {{ display:inline-flex; align-items:center; justify-content:center;
                   width:1.55rem; height:1.55rem; border-radius:50%;
                   font-size:.85rem; font-weight:700; color:#0d0d0d; }}

  .dss-hero     {{ display:flex; align-items:baseline; gap:.65rem; }}
  .dss-hero-num {{ font-size:4.25rem; font-weight:650; line-height:1;
                   letter-spacing:-.025em; }}
  .dss-hero-cap {{ font-size:.9rem; color:{INK_2}; text-transform:uppercase;
                   letter-spacing:.08em; }}

  .dss-badge    {{ display:inline-flex; align-items:center; gap:.6rem;
                   padding:.6rem 1rem; border-radius:9px;
                   border-left:5px solid var(--c);
                   background:color-mix(in srgb, var(--c) 18%, transparent);
                   font-size:1.05rem; font-weight:650; color:#ffffff; }}
  .dss-badge-ico{{ font-size:1.15rem; color:var(--c); }}

  .dss-card     {{ padding:.9rem 1.05rem; border-radius:11px;
                   background:#1a1a19;
                   border:1px solid #2b2b29;
                   border-top:3px solid var(--a, {C_COND}); height:100%; }}
  .dss-k        {{ font-size:.8rem; text-transform:uppercase;
                   letter-spacing:.07em; font-weight:700; color:var(--a, {C_COND}); }}
  .dss-v        {{ font-size:1.7rem; font-weight:650; line-height:1.3;
                   color:#ffffff; }}
  .dss-sub      {{ font-size:.84rem; color:{INK_2}; opacity:.85; }}

  .dss-meter    {{ height:14px; border-radius:7px; overflow:hidden; display:flex;
                   background:#262624; }}
  .dss-seg      {{ height:100%; }}
  .dss-legend   {{ display:flex; gap:1.6rem; font-size:.92rem; margin-top:.55rem;
                   color:{INK_2}; }}
  .dss-dot      {{ display:inline-block; width:10px; height:10px; border-radius:2px;
                   margin-right:.45rem; }}
  .dss-scale    {{ display:flex; justify-content:space-between;
                   font-size:.78rem; color:{INK_2}; opacity:.75; margin-top:.3rem;
                   font-variant-numeric:tabular-nums; }}
</style>
""", unsafe_allow_html=True)


def section(n, title, color):
    st.markdown(
        f"<div class='dss-h'><span class='dss-h-num' style='background:{color}'>"
        f"{n}</span><span style='color:{color}'>{title}</span></div>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_importance(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    required = {"segment_id", "tier", "route", "importance", "importance_pct"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Importance file is missing required columns: {sorted(missing)}")
        st.stop()
    df["route"] = df["route"].fillna("(unnamed)")
    return df


def find_importance_csv():
    return next((p for p in IMPORTANCE_CSV_CANDIDATES if p.exists()), None)


def _secret(name):
    """st.secrets raises if no secrets.toml exists, so this must be guarded."""
    try:
        return st.secrets.get(name, "")
    except Exception:                              # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Module 1 — PSCI from image (Claude Sonnet 4.6, Xu et al. prompt)
# ---------------------------------------------------------------------------
def grade_image(b64, media_type, api_key, model):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=1000, system=system_prompt,
        messages=[{"role": "user", "content": build_user_content(b64, media_type)}],
    )
    reply = "".join(b.text for b in resp.content if b.type == "text").strip()
    return extract_grade(reply), reply


# ---------------------------------------------------------------------------
# Small view helpers
# ---------------------------------------------------------------------------
def stat_card(col, label, value, sub="", accent=C_COND):
    """Label wears the accent colour; the value stays in primary ink so colour
    never reads as if it encodes the number."""
    col.markdown(
        f"<div class='dss-card' style='--a:{accent}'>"
        f"<div class='dss-k'>{label}</div>"
        f"<div class='dss-v'>{value}</div>"
        f"<div class='dss-sub'>{sub}&nbsp;</div></div>",
        unsafe_allow_html=True)


def contribution_meter(cond_part, imp_part):
    """Stacked bar showing how the two weighted halves compose the net score.
    2px surface gap between the fills, per the mark spec."""
    total_track = 1.0
    a = max(cond_part, 0) / total_track * 100
    b = max(imp_part, 0) / total_track * 100
    st.markdown(
        f"<div class='dss-meter'>"
        f"<div class='dss-seg' style='width:{a:.2f}%;background:{C_COND}'></div>"
        f"<div style='width:2px'></div>"
        f"<div class='dss-seg' style='width:{b:.2f}%;background:{C_IMP}'></div>"
        f"</div>"
        f"<div class='dss-scale'><span>0.00</span><span>0.25</span>"
        f"<span>0.50</span><span>0.75</span><span>1.00</span></div>"
        f"<div class='dss-legend'>"
        f"<span><span class='dss-dot' style='background:{C_COND}'></span>"
        f"Condition contributes {cond_part:.3f}</span>"
        f"<span><span class='dss-dot' style='background:{C_IMP}'></span>"
        f"Importance contributes {imp_part:.3f}</span></div>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Configuration")

    api_key = st.text_input(
        "Anthropic API key",
        value=_secret("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Or set ANTHROPIC_API_KEY in the environment / Streamlit secrets.")
    model = st.text_input("Model", value=MODEL,
                          help="Sonnet 4.6 is the benchmarked model.")

    st.divider()
    st.markdown("### Imagery")
    img_source = st.radio(
        "Source", ["Google Street View", "Upload manually"], horizontal=False,
        help="OpenStreetMap has no photography. Street View covers public "
             "roads but needs a billing-enabled key.")

    svs_key = ""
    search_radius, svs_pitch, svs_fov = imagery.DEFAULT_RADIUS_M, imagery.DEFAULT_PITCH, imagery.DEFAULT_FOV
    if img_source == "Google Street View":
        svs_key = st.text_input(
            "Google Maps API key",
            value=_secret("GOOGLE_MAPS_API_KEY")
            or os.environ.get("GOOGLE_MAPS_API_KEY", ""), type="password")
        search_radius = st.slider("Search radius (m)", 20, 300,
                                  imagery.DEFAULT_RADIUS_M, 10)
        svs_pitch = st.slider(
            "Camera pitch", -90, 0, imagery.DEFAULT_PITCH, 5,
            help="Negative tilts the camera toward the pavement. At 0 the shot "
                 "is level with the horizon and shows almost no road surface.")
        svs_fov = st.slider("Field of view", 30, 120, imagery.DEFAULT_FOV, 10)

    st.divider()
    st.markdown("### Scoring weights")
    w_cond = st.slider("Condition urgency", 0.0, 1.0, DEFAULT_W_COND, 0.05)
    w_imp = st.slider("Network importance", 0.0, 1.0, DEFAULT_W_IMP, 0.05)
    if abs((w_cond + w_imp) - 1.0) > 1e-9:
        st.caption(f"Weights sum to {w_cond + w_imp:.2f} — renormalized.")

    imp_col = st.radio(
        "Importance column", ["importance_pct", "importance"],
        help="importance_pct is the percentile rank (uniform on [0,1]), so a "
             "50% weight contributes 50% of the variance. The raw composite is "
             "compressed toward the middle (std 0.22 vs 0.29) and would "
             "contribute less than its nominal weight.")

    st.divider()
    st.markdown("### Data")
    csv_path = find_importance_csv()
    uploaded_csv = st.file_uploader("Override importance CSV", type="csv")
    if uploaded_csv is not None:
        imp_df = load_importance(uploaded_csv)
        st.caption(f"{len(imp_df):,} segments (uploaded)")
    elif csv_path:
        imp_df = load_importance(csv_path)
        st.caption(f"{len(imp_df):,} segments · `{csv_path.name}`")
    else:
        st.error("larimer_importance.csv not found — upload it above.")
        st.stop()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"<div style='font-size:2.2rem;font-weight:700;letter-spacing:-.02em;"
    f"color:#ffffff'>Road Maintenance Prioritization</div>"
    f"<div style='color:{INK_2};font-size:.92rem;margin-top:.25rem'>"
    f"Larimer County, Colorado · Sonnet 4.6 pavement condition (PSCI) × "
    f"network importance → Net Score</div>",
    unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------------
# Step 1 — segment
# ---------------------------------------------------------------------------
section(1, "Road segment", C_COND)
c1, c2 = st.columns([2, 3])

routes = sorted(imp_df["route"].astype(str).unique())
default_route = routes.index("S COLLEGE AVE") if "S COLLEGE AVE" in routes else 0
route = c1.selectbox("Route", routes, index=default_route)

sub = imp_df[imp_df["route"].astype(str) == route].sort_values(
    "importance_pct", ascending=False)
labels = [f"{r.segment_id}  ·  imp {r.importance_pct:.3f}" for r in sub.itertuples()]
choice = c2.selectbox(f"Segment — {len(sub):,} on this route", labels)
seg = sub.iloc[labels.index(choice)]
importance_value = float(seg[imp_col])

k1, k2, k3, k4 = st.columns(4)
stat_card(k1, "Importance", f"{importance_value:.3f}", imp_col, C_IMP)
stat_card(k2, "AADT", f"{seg['aadt_filled']:,.0f}"
          if pd.notna(seg.get("aadt_filled")) else "—", "vehicles / day", C_COND)
stat_card(k3, "Road tier", str(seg["tier"]),
          str(seg.get("funcclass", ""))[:28], C_AQUA)
stat_card(k4, "Length", f"{seg['length_m']:,.0f} m"
          if pd.notna(seg.get("length_m")) else "—",
          str(seg["segment_id"]), C_VIOLET)

if pd.isna(seg.get("ebc_norm")):
    st.warning("This segment sits outside the largest connected component, so "
               "it has no betweenness value — its importance comes from AADT "
               "and NHS only. 694 of 23,190 segments are in this position.")
if pd.isna(seg.get("aadt_raw")):
    st.info(f"AADT was missing here and floored to {seg['aadt_filled']:,.0f} "
            "veh/day, so importance leans on betweenness and NHS designation.")

st.divider()

# ---------------------------------------------------------------------------
# Step 2 — imagery
# ---------------------------------------------------------------------------
section(2, "Pavement imagery", C_AQUA)
has_coords = pd.notna(seg.get("lat")) and pd.notna(seg.get("lon"))
files = None
n_auto = 1

if img_source == "Upload manually":
    files = st.file_uploader("Street-level image(s) of this segment",
                             type=["png", "jpg", "jpeg", "gif", "webp"],
                             accept_multiple_files=True,
                             help="Multiple images are graded separately and "
                                  "their PSCI averaged.")
    if files:
        cols = st.columns(min(len(files), 4))
        for i, f in enumerate(files):
            cols[i % len(cols)].image(f, caption=f.name, width="stretch")
else:
    m1, m2 = st.columns([2, 3])
    if not has_coords:
        m1.error("No coordinates for this segment. Run `add_coordinates.py`.")
    else:
        m1.markdown(
            f"<div class='dss-card' style='--a:{C_AQUA}'>"
            f"<div class='dss-k'>Location</div>"
            f"<div class='dss-v' style='font-size:1.2rem'>"
            f"{seg['lat']:.5f}, {seg['lon']:.5f}</div>"
            f"<div class='dss-sub'>camera heading {seg['bearing']:.0f}° · "
            f"pitch {svs_pitch}° · {search_radius} m radius</div></div>",
            unsafe_allow_html=True)
        n_auto = m1.number_input("Images to grade", 1, 4, 1,
                                 help="More than one samples different headings "
                                      "from the same panorama.")
        m2.map(pd.DataFrame({"lat": [seg["lat"]], "lon": [seg["lon"]]}), zoom=14)

st.divider()

# ---------------------------------------------------------------------------
# Step 3 — run
# ---------------------------------------------------------------------------
ready = bool(api_key) and (bool(files) if img_source == "Upload manually"
                           else (has_coords and bool(svs_key)))
btn_label = ("Compute Net Score" if img_source == "Upload manually"
             else "Fetch imagery & compute Net Score")
run = st.button(btn_label, type="primary", disabled=not ready, width="stretch")

if not api_key:
    st.caption("Enter an Anthropic API key in the sidebar to enable scoring.")
elif img_source == "Upload manually" and not files:
    st.caption("Upload at least one image to enable scoring.")
elif img_source != "Upload manually" and not svs_key:
    st.caption("Enter a Google Maps API key in the sidebar to enable scoring.")


def collect_images():
    """-> list of (name, raw_bytes, caption). Stops the app if none obtainable."""
    if img_source == "Upload manually":
        return [(f.name, f.getvalue(), f.name) for f in files]

    lat, lon, bearing = float(seg["lat"]), float(seg["lon"]), float(seg["bearing"])
    n = int(n_auto)
    headings = [bearing] if n == 1 else [(bearing + i * 360.0 / n) % 360.0
                                         for i in range(n)]
    out = []
    with st.spinner("Fetching Street View imagery..."):
        for h in headings:
            try:
                raw, _, meta = imagery.streetview_fetch(
                    lat, lon, svs_key, heading=h, pitch=svs_pitch,
                    fov=svs_fov, radius_m=search_radius)
            except Exception as e:                 # noqa: BLE001
                st.error(f"Street View request failed — {e}")
                st.stop()
            if raw is None:
                st.error(f"No Street View panorama within {search_radius} m "
                         f"(status: {meta.get('status')}). Widen the search "
                         "radius or choose another segment.")
                st.stop()
            out.append((f"svs_h{h:.0f}", raw, f"heading {h:.0f}°"))
    return out


if run:
    images = collect_images()

    if img_source != "Upload manually":
        cols = st.columns(min(len(images), 4))
        for i, (_, raw, cap) in enumerate(images):
            cols[i % len(cols)].image(raw, caption=cap, width="stretch")

    grades, replies = [], []
    prog = st.progress(0.0, text="Grading pavement...")
    for i, (name, raw, _) in enumerate(images, 1):
        b64, media = prepare_image(raw, name)
        try:
            g, reply = grade_image(b64, media, api_key, model)
        except Exception as e:                     # noqa: BLE001
            st.error(f"{name}: API call failed — {e}")
            st.stop()
        grades.append((name, g))
        replies.append((name, reply))
        prog.progress(i / len(images), text=f"Graded {i}/{len(images)}")
    prog.empty()

    valid = [g for _, g in grades if PSCI_MIN <= g <= PSCI_MAX]
    if not valid:
        st.error("No image returned a parseable PSCI grade (1-10).")
        st.json(dict(replies))
        st.stop()
    if len(valid) < len(grades):
        st.warning(f"{len(grades) - len(valid)} of {len(grades)} image(s) "
                   "returned an unparseable grade and were excluded.")

    psci = sum(valid) / len(valid)
    urgency = psci_to_urgency(psci)
    score = net_score(urgency, importance_value, w_cond, w_imp)
    tier, color, icon = classify(score)
    denom = w_cond + w_imp
    cond_part = w_cond * urgency / denom
    imp_part = w_imp * importance_value / denom

    st.divider()
    section(3, "Result", color)

    hero, badge = st.columns([2, 3])
    # The hero number is coloured by TIER — a status role, not a series colour —
    # and the icon + label sit beside it so colour never carries it alone.
    hero.markdown(
        f"<div class='dss-hero'>"
        f"<span class='dss-hero-num' style='color:{color}'>{score:.3f}</span>"
        f"<span class='dss-hero-cap'>net score</span></div>",
        unsafe_allow_html=True)
    badge.markdown(
        f"<div class='dss-badge' style='--c:{color}'>"
        f"<span class='dss-badge-ico'>{icon}</span>"
        f"<span>{tier} priority</span></div>"
        f"<div class='dss-sub' style='margin-top:.45rem'>"
        f"{seg['segment_id']} · {route}</div>",
        unsafe_allow_html=True)

    st.write("")
    contribution_meter(cond_part, imp_part)

    st.write("")
    r1, r2, r3 = st.columns(3)
    stat_card(r1, "PSCI", f"{psci:.2f}",
              f"mean of {len(valid)} image(s) · 10 = best", C_YELLOW)
    stat_card(r2, "Condition urgency", f"{urgency:.3f}", "1 − (PSCI−1)/9", C_COND)
    stat_card(r3, "Network importance", f"{importance_value:.3f}", imp_col, C_IMP)

    st.caption(
        f"Net Score = {w_cond:.2f}×{urgency:.3f} (urgency) + "
        f"{w_imp:.2f}×{importance_value:.3f} (importance)"
        + (f", renormalized by {denom:.2f}" if abs(denom - 1) > 1e-9 else "")
        + f" = {score:.3f}   ·   Tier cutoffs: Critical ≥ 0.75, "
          "High ≥ 0.50, Medium ≥ 0.25, Low < 0.25")

    with st.expander("Per-image grades and raw model replies"):
        st.dataframe(pd.DataFrame(grades, columns=["image", "PSCI"]),
                     width="stretch", hide_index=True)
        for name, reply in replies:
            st.text(f"{name}: {reply}")

    with st.expander("Full segment record"):
        st.dataframe(seg.astype(str).to_frame("value"), width="stretch")

    st.download_button(
        "Download result (CSV)",
        pd.DataFrame([{
            "segment_id": seg["segment_id"], "route": route,
            "tier": seg["tier"], "psci": psci, "urgency": urgency,
            "importance_column": imp_col, "importance": importance_value,
            "w_condition": w_cond, "w_importance": w_imp,
            "net_score": score, "priority_tier": tier,
            "n_images": len(valid), "model": model,
            "imagery_source": img_source,
            "lat": seg.get("lat"), "lon": seg.get("lon"),
        }]).to_csv(index=False),
        file_name=f"dss_{seg['segment_id']}.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Step 4 — agent
#
# Placed at the bottom of the page rather than in a tab: the scoring flow above
# runs at module level, so wrapping it in a tab context would mean either a
# large re-indent or relying on Streamlit container internals. A section is
# equivalent for the user and robust to Streamlit changes.
# ---------------------------------------------------------------------------
st.divider()
section(4, "Ask the analyst", C_VIOLET)

EXAMPLES = [
    "Which mountainous highway segments have the highest importance?",
    "Grade the top segment on E MULBERRY ST and give me its net score.",
    "I have $500k. Of the 5 most important segments on S COLLEGE AVE, which should I treat?",
]
st.markdown("".join(f"<div class='dss-sub'>· {e}</div>" for e in EXAMPLES),
            unsafe_allow_html=True)
st.write("")

question = st.text_area("Your question", height=90, key="agent_q",
                        placeholder="Ask in plain English…")
ask = st.button("Ask", type="primary", disabled=not (question and api_key),
                width="stretch")
if not api_key:
    st.caption("Enter an Anthropic API key in the sidebar to enable the analyst.")

if ask:
    import dss_agent

    agent_ctx = dss_agent.AgentContext(
        df=imp_df, anthropic_key=api_key, svs_key=svs_key, imp_col=imp_col,
        w_cond=w_cond, w_imp=w_imp, svs_pitch=svs_pitch, svs_fov=svs_fov,
        svs_radius=search_radius)

    status = st.status("Working…", expanded=True)

    def on_agent_event(kind, payload):
        with status:
            if kind == "tool_use":
                args = ", ".join(f"{k}={v!r}" for k, v in payload["input"].items()
                                 if v not in ("", 0.0, None, 0))
                st.markdown(
                    f"<span style='color:{C_AQUA};font-weight:600'>▸ {payload['name']}</span>"
                    f" <span class='dss-sub'>{args[:170]}</span>",
                    unsafe_allow_html=True)
            elif kind == "tool_result":
                txt = str(payload["content"])[:280].replace("<", "&lt;")
                st.markdown(f"<span class='dss-sub'>&nbsp;&nbsp;↳ {txt}…</span>",
                            unsafe_allow_html=True)

    try:
        agent_answer, _ = dss_agent.run_agent(question, agent_ctx,
                                              on_event=on_agent_event)
        status.update(label="Done", state="complete", expanded=False)
    except Exception as e:                         # noqa: BLE001
        status.update(label="Failed", state="error")
        st.error(f"Agent run failed — {e}")
        agent_answer = None

    if agent_answer:
        st.markdown(
            f"<div class='dss-k' style='--a:{C_VIOLET};color:{C_VIOLET};"
            f"margin-top:.8rem'>Answer</div>", unsafe_allow_html=True)
        st.markdown(agent_answer)
