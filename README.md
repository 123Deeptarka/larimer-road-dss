# Road Maintenance Prioritization DSS — Larimer County, Colorado

A decision support system that ranks road segments for maintenance by fusing two
things agencies normally hold apart: **how badly a pavement has deteriorated**,
read from street-level imagery by a vision-language model, and **how much the
network loses if that segment is allowed to deteriorate**, computed from the road
graph.

Neither half answers the question on its own. A failed cul-de-sac and a
moderately worn arterial can carry the same repair cost and wildly different
consequences. The net score makes that trade-off explicit and, more importantly,
shows its two components separately so a planner can see *why* something ranked
where it did.

This repository holds the deployed application and the scored segment table for
all 23,190 segments in Larimer County. It accompanies the TRB paper *Coupling
Zero-Shot Vision-Language Pavement Assessment with Graph-Theoretic Network
Importance for Road Maintenance Prioritization*.

---

## How the score is built

**Module 1 — condition.** Claude Sonnet 4.6 grades a street-level image on the
PSCI 1–10 scale using the identical zero-shot chain-of-thought prompt that was
benchmarked in the paper (`psci_prompt.py`, verbatim). PSCI is inverted to an
urgency on [0, 1]:

```
urgency = 1 − (PSCI − 1) / 9        1.0 = failed pavement, 0.0 = newly surfaced
```

Sonnet 4.6 was selected over Opus 4.8 and Fable 5 not because it led on every
benchmark but because it combined the best accuracy on the realistic
street-level dataset (MAE 0.823, MSE 1.253) with materially lower latency and
cost.

**Module 2 — importance.** Precomputed per segment from three factors — traffic
demand, length-weighted edge betweenness centrality, and National Highway System
designation — combined under AHP-derived weights (0.444 / 0.389 / 0.167). This is
**not** computed in the app: exact edge betweenness over 18,486 nodes takes about
34 minutes and cannot run inside a web request. It is read from
`larimer_importance.csv`.

**Fusion.**

```
net = (w_cond × urgency + w_imp × importance) / (w_cond + w_imp)
```

Both weights default to 0.50 and are exposed as sliders. The denominator
renormalizes onto [0, 1] for any non-negative pair, so you can set 0.7 / 0.3
without the score leaving its range.

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run dss_app.py
```

The app opens with no configuration. Every segment is browsable and sortable
immediately; only the *grading* step needs credentials, and those are pasted into
the sidebar at runtime. To preload them instead, copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in —
that path is gitignored. Read the security section before doing this on a public
deployment.

You need an **Anthropic API key** to grade anything, and additionally a **Google
Maps API key** if you want the app to fetch Street View imagery rather than
supplying your own photographs.

---

## Using the app

The main pane is three numbered steps, top to bottom. The sidebar holds
credentials and everything that changes how the score is computed.

### Sidebar

| Control | What it does |
|---|---|
| **Anthropic API key** | Required for grading and for the analyst. Never stored. |
| **Model** | Defaults to the benchmarked Sonnet 4.6. Change only if you intend to depart from the paper. |
| **Imagery source** | `Street View` fetches automatically from the segment's coordinates; `Upload manually` lets you supply your own photographs. |
| **Google Maps API key** | Street View mode only. |
| **Search radius** | 20–300 m, default 60. How far from the segment point to look for a panorama. Larger values find imagery more often but risk returning a view of a different road. |
| **Camera pitch** | Default −40°, angled down at the surface rather than at the horizon. |
| **Field of view** | 30–120°. Narrower crops tighter on the pavement. |
| **Condition urgency / Network importance** | The two weights. They need not sum to 1 — the score renormalizes. |
| **Importance column** | `importance_pct` (percentile rank, uniform on [0, 1]) or `importance` (the raw composite). |
| **Override importance CSV** | Substitute your own scored table, e.g. for a different county or a re-weighted index. |

**On the importance column.** Prefer `importance_pct`. The raw composite is
compressed toward the middle of its range, so a nominal 50% weight delivers
considerably less than half the influence on the ranking. The percentile rank is
uniform by construction, so a 50% weight really is 50%. This is the easiest way
to misread the output.

### Step 1 — choose a segment

Pick a **route** from the dropdown (it opens on S College Ave), then a
**segment**. Segments are listed in descending importance and labelled with their
ID and importance value, so the top of the list is the most network-critical part
of that route.

Four cards then summarise the segment: its **importance**, **AADT**, **road
tier** with functional class, and **length** with the segment ID.

Two advisories appear when they apply, and both are worth heeding:

- *Outside the largest connected component.* 694 of the 23,190 segments sit in
  small disconnected fragments of the graph and have no betweenness value at all.
  Their importance rests on traffic and NHS designation alone, so it is measuring
  less than it does elsewhere.
- *AADT was imputed.* Traffic counts are reported for the highway and major tiers
  but are absent on most local roads. Where missing, the segment takes a
  class-based planning capacity rather than a measurement, so the traffic term is
  expressing a road class, not observed demand.

### Step 2 — get pavement imagery

In **Street View** mode the app shows the segment's coordinates, the camera
heading it will use, and a map. *Images to grade* (1–4) samples that many
headings from the same panorama; more images give a steadier grade at
proportionally more cost.

In **Upload** mode you supply the photographs. Multiple images are graded
independently and their PSCI averaged. Close-up, roughly overhead shots of the
running surface work best — the benchmark that established the model's accuracy
used photographs of that kind.

### Step 3 — compute, and read the result

The button grades the imagery and returns:

- the **net score** as a large number, coloured by priority tier, with the tier
  named beside it so colour never carries the meaning alone;
- a **contribution meter** splitting the score into its condition and importance
  halves;
- three cards — **PSCI** (mean across images), **condition urgency**, and
  **network importance**;
- the arithmetic spelled out, e.g. `0.50×0.444 + 0.50×1.000 = 0.722`;
- expanders holding the per-image grades with the model's raw replies, and the
  full segment record;
- a CSV download of the result.

**Worked example.** The highest-importance segment on S College Ave has an
importance percentile of 1.000. A PSCI of 6 gives urgency 0.444, so the condition
half contributes 0.222 against the importance half's 0.500, for a net score of
**0.722** and a High classification. Presenting the split rather than the total
alone is the point: the same 0.722 could come from a badly deteriorated
peripheral street or a sound but critical arterial, and those two warrant
completely different responses.

**A caveat on the tier cutoffs.** The app ships with the inherited thresholds
(Critical ≥ 0.75, High ≥ 0.50, Medium ≥ 0.25). These were calibrated against a
different condition measure under different weights, and under equal weighting
they over-flag badly — 76.3% of the graded sample lands in a single band. The
paper recalibrates them against an estimated network-wide distribution to
**Critical ≥ 0.657, High ≥ 0.517, Medium ≥ 0.292**, a 10 / 25 / 40 / 25 split.
Treat the in-app tier label as indicative and the net score itself as the ranking
quantity.

### The analyst

Below the three steps is a natural-language interface. Ask a question and a
tool-using agent answers it by calling deterministic functions over the same
segment table:

| Tool | What it returns |
|---|---|
| `query_segments` | Filter by route, tier, importance threshold |
| `get_segment` | The full record for one segment |
| `grade_segment` | Fetches imagery and grades it |
| `compute_net_score` | The fusion arithmetic for a given PSCI |
| `select_within_budget` | Exact budget-constrained selection, solved as a 0/1 knapsack |

Every number the analyst reports is a function return value, not a model
estimate. That is deliberate: it keeps the answers reproducible and auditable
while removing the need to write code to query the system.

Useful things to ask:

- *Which ten segments on US-287 have the highest network importance?*
- *Grade the most important segment on E Elkhorn Ave and give me its net score.*
- *I have $2 million. Which segments should I resurface first?*

The budget tool uses a placeholder treatment cost of $250,000/km. Replace it with
your own unit costs before treating its output as a programme.

---

## Files

| File | Role |
|---|---|
| `dss_app.py` | The application. Entry point for `streamlit run`. |
| `dss_core.py` | Pure scoring helpers — urgency, net score, tier classification. Side-effect free. |
| `dss_agent.py` | The tool-using analyst and its five deterministic tools. Imported by the app, not run directly. |
| `imagery.py` | Google Street View Static API fetcher. |
| `psci_prompt.py` | The PSCI grading prompt, verbatim from the benchmark. |
| `larimer_importance.csv` | 23,190 scored segments (~6 MB). |
| `requirements.txt` | Pinned dependencies. |

### The segment table

`larimer_importance.csv`, one row per segment. Columns the app reads:

| Column | Meaning |
|---|---|
| `segment_id` | Unique ID: tier letter + source OBJECTID + explode index |
| `route`, `tier`, `funcclass` | Route name, Highway/Major/Local, FHWA functional class |
| `aadt_raw`, `aadt_filled` | Reported AADT, and the value after class-based imputation |
| `ebc_norm` | Length-weighted edge betweenness, percentile rank. Blank outside the largest component |
| `nhs_norm` | 1.0 mainline NHS, 0.5 connector, 0.0 otherwise |
| `importance` | The AHP-weighted composite |
| `importance_pct` | Its percentile rank — the column to prefer |
| `lat`, `lon`, `bearing` | Segment midpoint and camera heading for Street View |
| `length_m` | Segment length in metres |

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository. Verify `git status` shows no
   `secrets.toml` before committing.
2. At [share.streamlit.io](https://share.streamlit.io) → **New app**, select the
   repository, set the main file to `dss_app.py`, and choose Python 3.11.
3. **Leave the Secrets box empty** for a public deployment — see below.
4. Deploy. The first build takes a few minutes.

## Security — read before setting secrets

The app pre-fills the sidebar from `st.secrets`. On a **public** deployment that
turns it into an open proxy to your billed APIs: any visitor can run Claude calls
and Street View requests on your account, without limit.

For a public demo:

- Leave both secrets **unset**. All 23,190 segments stay browsable and sortable;
  only grading needs a key, and each visitor supplies their own.
- If you must preload keys, restrict the app to named viewers in Streamlit Cloud
  settings, cap the Google key by quota and API restriction in Google Cloud
  Console, and set a spend limit on the Anthropic key.

Rotate any key that has ever been pasted into a chat, a commit, or a log.

---

## Known limitations

- Grades come from Street View imagery, which is wide-angle and oblique. The
  benchmark accuracy was measured on close-up road-surface photographs and does
  **not** transfer directly to these grades.
- Imagery coverage is not uniform. 98.5% of sampled candidates had a panorama
  within the search radius, but coverage fell to 92.8% on the rural reference
  group — imagery-based assessment is least available exactly where the network
  is sparsest.
- 694 of 23,190 segments lie outside the largest connected component and carry no
  betweenness; their importance derives from traffic and NHS only.
- 16,036 of 23,190 segments have no reported AADT and take a class-based planning
  capacity instead of a measurement. The traffic factor treats these as an
  ordering of road classes, not as an estimate of demand.
- Equal nominal weights do not produce equal influence — see the note on the
  importance column above.
- The shipped tier cutoffs over-flag under equal weighting; use the recalibrated
  values given in Step 3.
- Treatment costs in the budget tool are a placeholder ($250,000/km).

## Citation

If you use this work, please cite the TRB paper: Roy, D., Xiao, Y., and Jana, D.
*Coupling Zero-Shot Vision-Language Pavement Assessment with Graph-Theoretic
Network Importance for Road Maintenance Prioritization.*
