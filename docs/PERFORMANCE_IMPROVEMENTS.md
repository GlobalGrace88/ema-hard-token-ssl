# Training recipes

**Default (v5):** adaptive hard-token mining (`error_mass`), anatomy-aware masking, boundary objective preset, cosine λ schedule, and downstream `dice_ce_boundary` loss.

**Paper baseline (v4):** pass `--recipe v4` to pretrain/finetune/materialize commands.

## SSL pretrain (`v5_adaptive_boundary`, default)

Template: `configs/pretrain/ours_v5_adaptive.yaml`

| Feature | YAML keys |
|---------|-----------|
| Objective preset (boundary stages 2–4) | `feature_reconstruction.objective_preset: boundary` |
| Adaptive hard-token mining | `hard_token_mining.mode: error_mass`, `error_mass_fraction: 0.65` |
| Mining curriculum | `curriculum_epochs`, `curriculum_error_mass_start` |
| Per-stage error mass | `error_mass_fraction_per_stage.stage2/3/4` |
| Anatomy-aware masking | `inpainting.mask_strategy: aamm` |
| Longer SSL + cosine λ schedule | `training.epochs: 100`, `lambda_feature_schedule: cosine` |

```bash
python scripts/cli.py run pretrain --task synapse --model unetrpp
```

Presets: `boundary` | `overlap` | `balanced` (see `SSL_OBJECTIVE_PRESETS` in `unetrpp_feature_reconstruction.py`).

## Downstream (boundary loss + v5 init, default)

Template: `configs/downstream/synapse_5fold_v5.yaml`

```bash
python scripts/cli.py run finetune --task synapse --method ours --fold all
```

```yaml
loss:
  name: dice_ce_boundary
  boundary_weight: 0.15
```

## Paper baseline (v4)

```bash
python scripts/cli.py run pretrain --task synapse --model unetrpp --recipe v4
python scripts/cli.py run finetune --task synapse --method ours --recipe v4 --fold all
```

Uses `hard_token_mining.mode: topk_error`, stages 2/3/4, 50-epoch pretrain, standard `dice_ce` downstream loss.

## 5-fold ensemble eval

```bash
bash scripts/run_synapse_5fold_ensemble_eval.sh ours v5 0,1,2,3,4 0.625
# paper baseline checkpoints:
bash scripts/run_synapse_5fold_ensemble_eval.sh ours v4 0,1,2,3,4 0.625
```

Uses `eval.py --checkpoint_list` (mean logits in `official_npz`).

## Tests

```bash
python tests/test_hard_token_mining.py
```
