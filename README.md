# Road Maintenance Prioritization DSS — Larimer County

Streamlit decision support system fusing VLM-derived pavement condition with
graph-theoretic network importance.

- **Module 1 — condition.** Claude Sonnet 4.6 grades street-level imagery on the
  Xu et al. (2025) PSCI 1–10 scale.
- **Module 2 — importance.** Precomputed per segment from AADT, edge betweenness
  centrality and NHS designation (`larimer_importance.csv`, 23,190 segments).
- **Fusion.** `net = w_cond × urgency + w_imp × importance`, where
  `urgency = 1 − (PSCI − 1) / 9`.

Module 2 is **not** computed here — exact edge betweenness over 18,486 nodes
takes ~34 minutes and cannot run inside a web request. It is read from the CSV.

## Files

| File | Role |
|---|---|
| `dss_app.py` | The app. Entry point for `streamlit run`. |
| `dss_core.py` | Pure scoring helpers, side-effect free. |
| `dss_agent.py` | Tool-using agent (Opus 4.8) over five deterministic tools. Imported by the app, not run directly. |
| `imagery.py` | Google Street View Static API fetcher. |
| `psci_prompt.py` | PSCI grading prompt, verbatim from the benchmark. |
| `larimer_importance.csv` | 23,190 scored segments (~6 MB). |

## Run locally

```bash
pip install -r requirements.txt
streamlit run dss_app.py
```

Keys can be pasted into the sidebar at runtime — no configuration needed to
start. To preload them, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and fill it in. That file is gitignored.

## Deploy to Streamlit Community Cloud

1. Create a **new GitHub repository** and push the contents of this folder.
   Verify `git status` shows no `secrets.toml` before committing.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**, select
   the repo, set the main file to `dss_app.py`, and choose Python 3.11.
3. **Leave the Secrets box empty** for a public deployment — see below.
4. Deploy. First build takes a few minutes.

## Security — read before setting secrets

The app pre-fills the sidebar from `st.secrets`. On a **public** deployment that
turns it into an open proxy to your billed APIs: any visitor can run Claude
calls and Street View requests on your account, without limit.

Recommended for a public demo:

- Leave both secrets **unset**. All 23,190 segments remain browsable and
  sortable; only the grading step needs a key, and each visitor supplies their
  own.
- If you must preload keys, restrict the app to specific viewers in Streamlit
  Cloud settings, cap the Google key by quota and API restriction in Google
  Cloud Console, and set a spend limit on the Anthropic key.

Rotate any key that has ever been pasted into a chat, a commit, or a log.

## Known limitations

- Grades come from Street View imagery, which is wide-angle and oblique. Module
  1's benchmark accuracy was measured on close-up road-surface photographs and
  does **not** transfer directly to these grades.
- 694 of 23,190 segments lie outside the largest connected component and carry
  no betweenness; their importance derives from AADT and NHS only.
- Most local roads have no measured AADT and are floored to 100 veh/day.
- The default tier cutoffs (0.75 / 0.50 / 0.25) over-flag under equal weighting;
  recalibrated values are 0.656 / 0.517 / 0.292.
- Treatment costs in the agent's budget tool are a placeholder ($250,000/km).
