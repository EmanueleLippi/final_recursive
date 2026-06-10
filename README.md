# final_recursive

Versione modulare TF2 nativa di `code/Network/recursive1_GirsanovLike.py`.

Entry point unico:

```bash
PYTHONPATH=code python -m final_recursive run --mode recursive
PYTHONPATH=code python -m final_recursive plot --run_dir path/to/run
PYTHONPATH=code python -m final_recursive test
PYTHONPATH=code python -m final_recursive test --include-v1-parity
```

In questa workspace ho creato anche una virtualenv isolata per evitare conflitti del
base env Anaconda:

```bash
PYTHONPATH=code code/final_recursive/.venv/bin/python -m final_recursive test --include-v1-parity
PYTHONPATH=code code/final_recursive/.venv/bin/python -m final_recursive run --mode recursive
PYTHONPATH=code code/final_recursive/.venv/bin/python -m final_recursive plot --run_dir path/to/run
```

La versione target di TensorFlow verificata via pip e `2.21.0`. Il file
`requirements.txt` blocca anche `numpy<2.0` e `h5py<3.15`, perche TensorFlow 2.21
e lo stack binario scientifico locale non sono stabili con NumPy 2.4.

Il test `--include-v1-parity` confronta la nuova implementazione TF2 nativa con
`code/Network/recursive1_GirsanovLike.py` in un processo separato: stesso blob
di pesi, stesso batch, predizioni `X/Y/Z` e un passo Adam deterministico.

## Miglioramenti opzionali

I default restano compatibili con la loss legacy. I miglioramenti vanno abilitati
esplicitamente da CLI:

```bash
PYTHONPATH=code python -m final_recursive run \
  --mode recursive \
  --dynamic_loss_dt_normalization \
  --terminal_z_loss_weight 2.0 \
  --terminal_z_component_weights 3,0.25,2,0 \
  --structural_z_loss_weight 0.5 \
  --structural_z_component_weights 0,1,0,0 \
  --same_xi_antithetic_sampling \
  --selection_metric exact_mae_z_s
```

Le run salvano nei CSV anche componenti diagnostiche della loss con prefisso
`eval_mean_loss_*`, inclusi residuo dinamico, terminale `Y`, terminale `Z` e
penalita strutturale.

## Figure Exact

Le metriche exact restano calcolate sull'evaluation bundle grande. Per le figure
leggibili, la pipeline genera anche un piccolo visual bundle deterministico:

```bash
--visual_sample_paths 8
--visual_seed 72051
```

Se `--visual_seed` vale `-1`, il seed viene derivato da `--eval_seed`. I plot
`recursive_stitched_Y_exact*.png` e `recursive_stitched_Z_*_exact*.png` usano
questi path visuali; le curve `recursive_stitched_abs_error*.png` e
`recursive_stitched_Z_rel_error*.png` restano invece basate sul bundle grande.

## Figure Pascucci Paper

Il comando plot-only rigenera i grafici paper Pascucci #35-#40 da artifact gia'
salvati, senza avviare training:

```bash
PYTHONPATH=code python -m final_recursive plot --run_dir path/to/run
```

Input attesi:

- `run_config.json`
- `recursive/stitched_predictions_final.npz`
- `recursive/application_metrics_final.npz`

Il `run_config.json` deve dichiarare `application_metric_schema =
pascucci_application_metrics_v2`; gli artifact precedenti vanno rigenerati per
ottenere le tracce cumulative `cost_J_running_cumulative` usate dal grafico
`#37`. Nel grafico `#40` le barre indicano la mediana con intervallo q05-q95 e
il diamante nero indica la media, cosi' le distribuzioni sbilanciate restano
rappresentabili senza interpretare i quantili come intervallo di confidenza
della media.

Output default:

- `recursive/plots/pascucci_paper/pascucci_paper_35_S_ou_band.png`
- `recursive/plots/pascucci_paper/pascucci_paper_36_H_ou_band.png`
- `recursive/plots/pascucci_paper/pascucci_paper_37_accumulated_cost.png`
- `recursive/plots/pascucci_paper/pascucci_paper_38_alpha.png`
- `recursive/plots/pascucci_paper/pascucci_paper_39_state_bands_S_V_Q.png`
- `recursive/plots/pascucci_paper/pascucci_paper_40_controlled_uncontrolled.png`
- `recursive/plots/pascucci_paper/pascucci_paper_plots_manifest.json`
