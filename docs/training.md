# Training details

## SSL pretrain

Template: `configs/pretrain/ours.yaml`

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

Checkpoint: `outputs/pretrain/synapse/best.pt`

Presets: `boundary` | `overlap` | `balanced` (see `SSL_OBJECTIVE_PRESETS` in `unetrpp_feature_reconstruction.py`).

## Downstream (5-fold, boundary loss + SSL init)

Template: `configs/downstream/synapse_5fold.yaml`

```bash
python scripts/cli.py run finetune --task synapse --method ours --fold all
```

Checkpoints: `outputs/downstream/synapse/ours/fold_{0..4}/best_model.pt`

```yaml
loss:
  name: dice_ce_boundary
  boundary_weight: 0.15
```

Scratch baseline uses `dice_ce` (`configs/downstream/synapse_5fold_scratch.yaml`).

## 5-fold ensemble eval

Runs **after** all five folds are trained — averages logits across fold checkpoints:

```bash
bash scripts/run_synapse_5fold_ensemble_eval.sh ours 0,1,2,3,4 0.625
```

## Tests

```bash
python tests/test_hard_token_mining.py
```
