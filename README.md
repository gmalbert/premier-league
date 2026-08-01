# Premier League Consumer

This is the thin EPL deployment for Pitch Oracle. Shared prediction, data pipeline,
reporting, provider adapters, and Streamlit shell code are installed from the pinned
`pitch-oracle-core` release in `requirements.txt`.

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
