
# DSR-EFA

Code for the CIKM 2026 paper, **“Accelerating Feature Aggregation for Imbalanced Multimodal Learning.”**

The paper proposes DSR-EFA, which retrieves same-class samples in logit space to enrich minority-class multimodal training data. It allocates retrieval budgets according to modality importance and combines retrieved features with prototype clustering and dynamic EMA updates.

## Repository Layout

```text
DSR_EFA/
├── configs/                  # CREMA-D and pool-builder configurations
├── dsr_efa/
│   ├── config/               # Shared default configuration
│   ├── datasets/             # Dataset loaders (CREMA-D entry point)
│   ├── models/               # Audio/video/text encoders and multimodal classifiers
│   └── utils/                # Losses, schedulers, retrieval and training helpers
├── scripts/
│   ├── prepare/              # CREMA-D candidate-pool builder
│   └── train/                # CREMA-D training entry point
├── tools/
│   └── analysis/             # Reserved for analysis and plotting utilities
├── data/
├── checkpoints/
├── outputs/
├── requirements.txt
└── README.md
```

## Example

Install the dependencies first, update the dataset, checkpoint, and logit paths in the CREMA-D configuration and scripts, then run the following commands from the repository root:

```bash
python -m scripts.prepare.build_cremad_pool --config configs/cremad_pool.json
python -m scripts.train.train_cremad --config configs/cremad.json
```
