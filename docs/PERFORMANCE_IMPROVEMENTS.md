# Performance improvements (v5)

Optional training improvements on top of the v4 paper recipe (`--recipe v4`, default).

## SSL pretrain (`v5_adaptive_boundary`)

Template: `configs/pretrain/ours_v5_adaptive.yaml` → materialize with `--recipe v5`.

| Feature | YAML keys |
|---------|-----------|
| Objective preset (boundary stages 2–4) | `feature_reconstruction.objective_preset: boundary` |
| Adaptive hard-token mining | `hard_token_mining.mode: error_mass`, `error_mass_fraction: 0.65` |
| Mining curriculum | `curriculum_epochs`, `curriculum_error_mass_start` |
| Per-stage error mass | `error_mass_fraction_per_stage.stage2/3/4` |
| Anatomy-aware masking | `inpainting.mask_strategy: aamm` |
| Longer SSL + cosine λ schedule | `training.epochs: 100`, `lambda_feature_schedule: cosine` |

```bash
python scripts/cli.py run pretrain --task synapse --model unetrpp --recipe v5
```

Presets: `boundary` | `overlap` | `balanced` (see `SSL_OBJECTIVE_PRESETS` in `unetrpp_feature_reconstruction.py`).

Legacy v4 behaviour: `--recipe v4` or `hard_token_mining.mode: topk_error`.

## Downstream boundary loss

Template: `configs/downstream/synapse_5fold_v5.yaml`

```bash
python scripts/cli.py run finetune --task synapse --method ours --recipe v5 --fold all
```

```yaml
loss:
  name: dice_ce_boundary
  boundary_weight: 0.15
```

## 5-fold ensemble eval

```bash
bash scripts/run_synapse_5fold_ensemble_eval.sh ours v5 0,1,2,3,4 0.625
```

Uses `eval.py --checkpoint_list` (mean logits in `official_npz`).

## Tests

```bash
python -m pytest tests/test_hard_token_mining.py -q
```
