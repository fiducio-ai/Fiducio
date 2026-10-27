# Relecture du plan MHR — 17 octobre 2026

Document relu : [PLAN-MHR.md](PLAN-MHR.md), code de référence `d93570f`.
Cette revue porte sur un plan, sans implémentation et sans tests exécutés.

## Tentatives OpenCode

Le modèle configuré et listé par `opencode models` est
`edelweiss/DeepSeek-V4.1-Flash`. La tentative de relecture avec l'agent `plan`
en lecture seule a échoué avant toute réponse du modèle : HTTP 403,
`provider.auth`, code fournisseur `private-unavailable`, message
« Inference is temporarily unavailable to this credential. »
Session : `ses_eb72d0ab3ffehCYeLr0YwVbKtu`.

La tentative de repli `opencode/mimo-v2.6-flash-free` a échoué avec
`provider.no-route`, « Model unavailable: opencode/mimo-v2.6-flash-free ».
Session : `ses_eb72cbc6dffeN9J7p0V51QSTjH`.
Le [catalogue Zen](https://docs.opencode.ai/docs/zen/) répertorie ce modèle
gratuit, mais l'instance installée ne le propose pas. Une instance indépendante
`opencode models --standalone` donne la même liste limitée à DeepSeek.
Le coût de l'accès Edelweiss n'était pas indiqué dans la configuration locale ;
sa gratuité n'a pas été établie. Aucun avis d'OpenCode n'a été obtenu.

## Relecture indépendante de secours

Un second agent Codex a lu le plan et les fichiers `base.py`,
`utils/tensors.py`, `calibrators/ensemble_temperature.py`,
`metrics/calibration.py`, `persistence.py` et `plots/__init__.py`.
Son avis : les objectifs CE-MHR, les conventions Brier, la distinction entre
les deux ECE et le PAVA pondéré sont cohérents. Trois corrections concrètes
ont été retenues avant implémentation.

| Priorité | Constat | Correction intégrée au plan |
| --- | --- | --- |
| Haute | Le fit appelle aussi `_to_canonical` ; le chemin actuel `safe_log` détruit la distinction entre 0 et les probabilités sous `1e-12` avant l'ajustement isotone | Surcharge ciblée de la canonisation isotone pour conserver les probabilités directes ; oracle sur `0`, `1e-13`, `1e-12` |
| Moyenne | ETS pourrait choisir son chemin dur/souple uniquement à partir du train et échouer lorsque la validation utilise l'autre format | Branchement indépendant par jeu ; tests croisés train dur/validation souple et inverse avec early stopping |
| Moyenne | Deux abscisses moyennes distinctes en float64 peuvent devenir égales en float32 ; une simple validation après conversion ferait échouer des entrées valides | Stockage et rechargement des abscisses en float64, interpolation dans cette précision, sortie publique float32 ; test de roundtrip |

Les trois remarques ont été intégrées. Aucune remarque de cette revue n'a été
écartée. Le plan précise aussi l'interaction entre le contrôle numérique de la
maintenance et les `-inf` légitimes du `decision_function` isotone.

## Limites

Les remarques ont été vérifiées par lecture des chemins locaux. Aucune suite
de tests, exécution CUDA, mesure de mémoire ou expérience sur données patient
n'a été effectuée pour cette rédaction. Le code original de TwinTrack n'a pas
été comparé : l'identité exacte des règles de binning, d'égalité et
d'interpolation reste ouverte. Les sources de l'article sont liées dans le plan.
