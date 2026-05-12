# final_recursive

Versione modulare TF2 nativa di `code/Network/recursive1_GirsanovLike.py`.

Entry point unico:

```bash
PYTHONPATH=code python -m final_recursive run --mode recursive
PYTHONPATH=code python -m final_recursive test
PYTHONPATH=code python -m final_recursive test --include-v1-parity
```

In questa workspace ho creato anche una virtualenv isolata per evitare conflitti del
base env Anaconda:

```bash
PYTHONPATH=code code/final_recursive/.venv/bin/python -m final_recursive test --include-v1-parity
PYTHONPATH=code code/final_recursive/.venv/bin/python -m final_recursive run --mode recursive
```

La versione target di TensorFlow verificata via pip e `2.21.0`. Il file
`requirements.txt` blocca anche `numpy<2.0` e `h5py<3.15`, perche TensorFlow 2.21
e lo stack binario scientifico locale non sono stabili con NumPy 2.4.

Il test `--include-v1-parity` confronta la nuova implementazione TF2 nativa con
`code/Network/recursive1_GirsanovLike.py` in un processo separato: stesso blob
di pesi, stesso batch, predizioni `X/Y/Z` e un passo Adam deterministico.
