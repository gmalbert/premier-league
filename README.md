# Premier League Consumer

This is the thin EPL deployment for Pitch Oracle. Shared prediction, data
preparation, training, artifact validation, provider adapters, and Streamlit pages
are installed from the single pinned `pitch-oracle-core` release in
`requirements.txt`.

The consumer owns only:

- `config.py` and `predictions.py`;
- EPL historical data, generated feature files, precomputed data, and trained models;
- EPL deployment workflows, secrets, and consumer-specific verification tests.

Run locally with:

```bash
pip install -r requirements.txt
streamlit run predictions.py
```

Verify shared-core parity with:

```bash
python scripts/verify_parity.py
```

The sole scheduled writer is `.github/workflows/nightly-pipeline.yml`. It builds
data, models, predictions, diagnostics, and the league-bound manifest as one
coherent artifact set before running the full consumer test and quality gates.
