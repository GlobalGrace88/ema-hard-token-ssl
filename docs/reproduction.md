# Synapse paper reproduction

End-to-end steps for Tier-1 release (UNETR++ official, 5-fold Synapse).

## 1. Install

```bash
conda env create -f environment.yml
conda activate ema-hard-token-ssl
pip install -r requirements.txt
```

## 2. Paths

```bash
cp paths.yaml.example paths.yaml
# edit: data_root, nnformer_dir, ssl_ct_root, unetr_pp_root
```

Or export env vars: `DINOMIM_DATA_ROOT`, `DINOMIM_NNFORMER_DIR`, `DINOMIM_SSL_CT_ROOT`, `UNETR_PP_ROOT`.

## 3. UNETR++

Follow [setup_unetrpp.md](setup_unetrpp.md).

## 4. Preprocess

```bash
python scripts/cli.py preprocess --task synapse --stage both
```

## 5. Materialize configs (optional — CLI auto-materializes)

```bash
python scripts/cli.py materialize --task synapse --model unetrpp --phase all
```

## 6. Pretrain (EMA hard-token feature reconstruction)

```bash
python scripts/cli.py run pretrain --task synapse --model unetrpp
```

Checkpoint: `outputs/pretrain/synapse/best.pt` (+ `.meta.json` sidecar).

## 7. 5-fold finetune

Scratch baseline:

```bash
python scripts/cli.py run finetune --task synapse --model unetrpp --method scratch --fold all
```

Ours (SSL-init from pretrain):

```bash
python scripts/cli.py run finetune --task synapse --model unetrpp --method ours --fold all
```

## 8. Official eval (sliding-window, overlap 0.5)

```bash
python scripts/cli.py run eval --task synapse --model unetrpp --method ours \
  --checkpoint outputs/downstream/synapse/ours/fold_0/best_model.pt \
  --fold all --official
```

Eval refuses checkpoints whose embedded metadata does not match the Synapse task (no cross-dataset eval).

## 9. Optional 5-fold ensemble eval

```bash
bash scripts/run_synapse_5fold_ensemble_eval.sh ours 0,1,2,3,4 0.625
```

## Hyperparameters

| Stage | Key settings |
|-------|----------------|
| Pretrain | 100 epochs, error_mass mining, AAMM masking, boundary preset, cosine λ |
| Finetune | 400 samples/epoch, dice_ce_boundary (ours), poly LR, early stop patience 150 (min 400 ep) |
| Eval | ROI 64×128×128, overlap 0.5, 8-organ Dice+HD95 |

## Citation

If you use this code, please cite the Synapse SSL paper (bibtex TBD).
