# Examen DVC

Ce depot contient un workflow DVC pour predire la concentration de silice (`silica_concentrate`) a partir des variables du processus de flottation.

## Pipeline

Le pipeline est decoupe en cinq etapes :

1. `split` : creation des jeux d'entrainement et de test.
2. `normalize` : normalisation de `X_train` et `X_test`.
3. `grid_search` : recherche des meilleurs hyperparametres.
4. `training` : entrainement du modele final.
5. `evaluate` : generation des predictions et des metriques.

## Execution

```bash
pip install -r requirements.txt
dvc repro
dvc metrics show
```

## Resultats

Les metriques sont sauvegardees dans `metrics/scores.json`.

```text
mse  = 0.77121
rmse = 0.87819
mae  = 0.67862
r2   = 0.22946
```
