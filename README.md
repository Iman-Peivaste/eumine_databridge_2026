# EuMINe DataBridge 2026

Multi-source materials property prediction for the EuMINe hackathon: formation energy and band gap from the Bridge Dataset, augmented with Materials Project and JARVIS data.

## Setup

```bash
conda activate ip
cd ~/EuMINe/eumine_databridge
pip install -e .
cp .env.example .env  # edit MP_API_KEY and WANDB_ENTITY
python scripts/smoke_test.py
```

## Layout

- `src/eumine_databridge/` — package (data loaders, harmonizer, ALIGNN/MACE models, MatFed predictor)
- `scripts/` — training, external fetch, submission generation
- `data/raw/` — Bridge Dataset CSVs and CIFs (not in git)
- `hackathon_ref/` — clone of [eumine_hackathon_2026](https://github.com/EuMINe-COST/eumine_hackathon_2026) for scoring and MatFed tests

## License

MIT — see [LICENSE](LICENSE).
