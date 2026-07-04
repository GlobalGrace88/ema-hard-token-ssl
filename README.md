# EMA Hard-Token SSL — Synapse Reproduction

Open-source release for **EMA hard-token self-supervised learning** with inpainting + multi-stage feature reconstruction, targeting the Synapse multi-organ CT segmentation benchmark.

Tier-1 scope: UNETR++ (official) pretrain → 5-fold Synapse finetune → official npz eval.

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure paths
cp paths.yaml.example paths.yaml
# edit paths.yaml — see docs/setup_unetrpp.md for UNETR++

# 3. Preprocess data
python scripts/cli.py preprocess --task synapse --stage both

# 4. Pretrain
python scripts/cli.py run pretrain --task synapse --model unetrpp

# 5. 5-fold finetune (ours = SSL-init)
python scripts/cli.py run finetune --task synapse --model unetrpp --method ours --fold all

# 6. Official eval
python scripts/cli.py run eval --task synapse --checkpoint outputs/downstream/synapse/ours/fold_0/best_model.pt --fold all --official
```

## CLI reference

| Command | Description |
|---------|-------------|
| `preprocess --task synapse --stage upstream\|downstream\|both` | Prepare SSL manifest / Synapse manifests |
| `run pretrain --task synapse --model unetrpp` | Upstream feature-reconstruction SSL |
| `run finetune --task synapse --method scratch\|ours --fold all` | Downstream 5-fold segmentation |
| `run eval --task synapse --checkpoint ... --fold all --official` | Paper-style sliding-window metrics |
| `materialize --task synapse --phase all` | Generate per-fold YAML configs |

## Repository layout

```
ema-hard-token-ssl/
├── dinomim_pytorch/     # core library (segmentation, SSL, eval)
├── configs/             # task registry, model defs, templates
├── scripts/             # CLI, trainers, preprocessing
├── docs/                # dataset setup, reproduction guide
└── tests/               # smoke / registry tests
```

## Path configuration

All machine-specific paths are resolved via `paths.yaml` or environment variables:

| Key | Env var |
|-----|---------|
| `data_root` | `DINOMIM_DATA_ROOT` |
| `nnformer_dir` | `DINOMIM_NNFORMER_DIR` |
| `ssl_ct_root` | `DINOMIM_SSL_CT_ROOT` |
| `unetr_pp_root` | `UNETR_PP_ROOT` |

## Checkpoint metadata

Pretrain and finetune checkpoints embed `release_metadata` (JSON sidecar `.pt.meta.json`). Eval validates task/model/dataset match — cross-dataset eval is blocked.

## Documentation

- [datasets.md](docs/datasets.md) — upstream FLARE+AMOS, downstream Synapse
- [reproduction.md](docs/reproduction.md) — full paper reproduction steps
- [setup_unetrpp.md](docs/setup_unetrpp.md) — clone official UNETR++

## Citation

```bibtex
@article{ema_hard_token_ssl,
  title   = {EMA Hard-Token SSL for 3D CT Segmentation},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Tier 2 (TODO)

- nnFormer backbone in public CLI
- UNETR (MONAI) downstream baseline
- Multi-task registry beyond Synapse
- Pretrained checkpoint release on Hugging Face
- DDP launch helpers / Slurm templates
