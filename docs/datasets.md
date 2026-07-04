# Datasets

## Upstream: public CT SSL (FLARE + AMOS)

Tier-1 Synapse reproduction uses unlabeled CT from **FLARE** and **AMOS** for self-supervised pretraining.

1. Register and download datasets per their challenge pages.
2. Set `ssl_ct_root` in `paths.yaml`.
3. Run:

```bash
python scripts/cli.py preprocess --task synapse --stage upstream
```

This builds `processed/manifest_ssl_ct.csv` under your SSL CT root.

## Downstream: Synapse (nnFormer Task002)

Downstream finetune/eval expects **nnFormer-preprocessed** Synapse npz volumes:

- `Task002_Synapse/stage1/*.npz`
- `splits_final.pkl` (5-fold)

Set `nnformer_dir` in `paths.yaml` to the parent of `nnFormer_preprocessed/`.

Obtain raw Synapse data from the [Multi-Atlas Abdominal Segmentation challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and preprocess with [nnFormer](https://github.com/282857341/nnFormer) or use an existing nnFormer export.

```bash
python scripts/cli.py preprocess --task synapse --stage downstream
```

## Official UNETR++

See [setup_unetrpp.md](setup_unetrpp.md). The official UNETR++ weights/architecture are loaded from an external clone — not vendored in this repo.
