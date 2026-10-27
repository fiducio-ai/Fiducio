# Relecture de la PR #18 par OpenCode / DeepSeek Free

Relecture du 17 octobre 2026, à la demande du mainteneur, sur le commit
`03df95c3e74a140c76052f3e6c9b2874fa5e35ee`.

- Outil : OpenCode 2.0.32, agent `plan`, lecture seule.
- Modèle : `opencode/deepseek-v4.1-flash-free`, catalogue OpenCode Console.
- Session : `ses_eb6da5352ffeiTEckCuSMa8nFn`.
- Entrée : diff depuis `d93570f` et quinze fichiers complets numérotés
  (base, tensors, ETS, isotone, optimisation, affine, conditionnels, métriques,
  persistence, plots, tests MHR/isotone/métriques/numérique et guide multirater).
- Aucun outil d'édition exécuté par le modèle. La première réponse a épuisé
  son budget de raisonnement ; la continuation a livré l'avis ci-dessous.
  Une relance redondante a été interrompue dès réception de cet avis.

Le fournisseur `opencode` a été ajouté aux autorisations globales avec l'accord
explicite du mainteneur reçu via Starbridge. La sélection du modèle gratuit et
les permissions de lecture seules ont été appliquées dans un répertoire de
revue temporaire ; le modèle par défaut global n'a pas été remplacé.

## Vérification des remarques

**Verdict du modèle : aucun défaut bloquant identifié.** Cela reste une revue
statique ; le modèle n'a pas exécuté les tests et n'a pas inspecté tous les
fichiers du dépôt.

1. **Sorties 0/1 : clarification documentaire retenue.** La garantie concerne
   les valeurs de sortie de la carte apprise, qui ne sont pas écrêtées par un
   `safe_log`, pas la conservation des scores d'entrée 0 et 1. Le comportement
   d'extrapolation constante était déjà spécifié. La docstring et le guide
   explicitent maintenant cette distinction ; aucune règle spéciale ne force
   un score d'entrée nul à rester nul.
   L'exemple proposé par le modèle a été exécuté : pour les scores
   `[.1,.2,.2,.2,.8,.9]`, les cibles `[1,0,0,0,1,1]` et trois bins, les points
   appris sont `x=[.175,.85]`, `y=[.25,1]`. Le score 0 donne **.25**, pas .5
   comme indiqué dans la réponse du modèle. Le calcul du code est correct.
2. **Garde numérique et `-inf` : aucune correction nécessaire.** L'isotone
   retourne directement ses probabilités et utilise leur logarithme dans
   `decision_function`. Le refus d'overflow des méthodes paramétriques est
   intentionnel et testé ; il ne rejette pas ces `-inf` isotones légitimes.
   Contrairement à une phrase de la réponse brute, `safe_log` ne produit pas
   `-inf` : il applique un plancher. C'est précisément pourquoi l'isotone
   utilise `.log()` et son propre chemin de probabilités.
3. **Liste des méthodes génériques : vérifiée localement.**
   `tests/conftest.py::ALL_CALIBRATOR_IDS` contient les neuf méthodes
   paramétriques et exclut l'isotone, qui dispose de tests dédiés. La réserve
   du modèle sur cette liste est donc levée. Le bundle ne contenait pas les
   corps complets de TS/vector/matrix/Dirichlet ; leur prise en charge est
   couverte par les tests paramétrés et la CI, pas par cette lecture externe.
4. **Suggestions facultatives : non appliquées.** Aucun défaut calculatoire
   n'a été démontré dans les libellés ECE, les messages d'erreur ou la boucle
   par rater de l'exemple. Les populations de voxels et votes absents sont
   explicitement documentées.

Les 583 tests locaux et les 16 checks CI du commit relu sont verts. La revue
ne remplace pas la validation sur données patient, CUDA ou le code original
TwinTrack ; ces vérifications restent ouvertes.

## Réponse brute du modèle

Le texte suivant est conservé pour traçabilité. Les corrections factuelles
ci-dessus font foi pour l'interprétation des remarques.

## Avis de revue — PR #18 (HEAD `03df95c`)

### Verdict
**Aucun défaut bloquant introduit.** J'ai recalculé à la main les oracles principaux (binning isotone, PAVA pondéré, interpolation, ETS étape 2, MHR vs moyenne par rater) : ils sont corrects. Les flux `targets`/`mask`/`ignore_index`, la persistance et la rétrocompatibilité sont cohérents. Les deux points ci‑dessous sont mineurs (le premier est une incohérence doc/comportement, pas un calcul faux).

### Défauts concrets

**1. Préservation 0/1 : la doc promet plus que le code (faible, doc)**
`src/fiducio/calibrators/isotonic.py:106-118` (`_apply`) : hors de `[x[0], x[-1]]`, la sortie est la valeur d'extrémité (`y[0]` / `y[-1]`). Or `x[0]` est la **moyenne** des scores de la première bin, pas `min(score)`. Un score de test exactement `0` (ou `1`) qui tombe sous `x[0]` est donc mappé vers `y[0]` (resp. `y[-1]`), qui n'est pas nécessairement `0` (resp. `1`). Exemple reproductible : `fit(probabilities([0.1,0.2,0.2,0.2,0.8,0.9]), probabilities([0,0,0,0,1,1]), n_bins=3)` → `x=[0.175,0.85]`, `y=[0,1]` ; un test à `0.0` donne `y[0]=0` ici, mais avec `targets=[1,0,0,0,1,1]` la première bin fusionne et `transform(score=0)` renvoie `0.5`, pas `0`.
Le comportement isotone est défendable ; ce sont `docs/multi_rater.md:83` (« exact output 0/1 are preserved ») et la docstring de classe (`isotonic.py:45-46`) qui surpromettent. Correction minimale : soit reformuler (« les valeurs de la carte en 0/1 sont retournées telles quelles »), soit forcer `positive = torch.where(scores <= self.x[0], self.y[0], positive)` **plus** `torch.where(scores == 0, 0.0, ...)` — mais le plus honnête est de reformuler la doc. Priorité **faible**.

**2. `decision_function` isotone et la garde d'overflow (information, pas un bug)**
`src/fiducio/base.py:404-408` ajoute un refus des logits non finis. C'est cohérent, mais `BinaryIsotonicCalibration` y échappe volontairement : elle **surcharge** `transform` (`isotonic.py:120-131`) et `decision_function` (`:133-135`), donc `_map_logits`/`_calibrated_logits` ne sont jamais atteints. Conséquence : le `-inf` documenté en sortie de `decision_function` est légitime ici, mais la garde du socle reste un **changement de comportement pour les autres calibration** (une `TemperatureScaling` avec `T` très petit peut désormais lever au lieu de renvoyer des probabilités saturées). Le test `test_numerical_regressions.py` couvre le cas ; à noter comme intentionnel. Priorité **faible**.

### Points vérifiés sans défaut
- **Binning/égalités/PAVA** : `ranks = ceil(r·n/k)` puis arrondi en fin de groupe d'égalité (`group_ends`), déduplication `unique_consecutive`, moyennes de scores strictement croissantes ; PAVA à poids de comptes, fusion `>` (garde les plateaux), `x/y` de même longueur. Oracles `test_weighted_pava_known_optimum…` et `test_binning_uses_voxel_weights…` reproduits à la main : exacts.
- **Interpolation** : `searchsorted` + `clamp(1, k-1)` + `fraction.clamp(0,1)` donnent bien l'extrapolation constante et couvrent le cas `numel==1`. `x` étant strictement croissante, pas de division par zéro.
- **0/1 et -inf** : `safe_log`/`.log()` donnent `-inf` uniquement aux zéros, et `decision.softmax(1) == transform(p)`.
- **MHR** : `counts = present.sum(1)`, dénominateur `clamp_min(1)`, masque `counts > 0` ; l'identité CE(MHR) = moyenne des CE par rater (valeur et gradient) est correcte et testée ; en `-100`, `num_classes<2` et labels hors bornes sont rejetés.
- **Deux ECE** : `class_index` compare `p_c` à `1[y=c]` ou `q_c` ; le mode top‑1 soft utilise `q[argmax p]` ; `sum_acc` par `scatter_add_` remplace `bincount(idx[correct])` de façon équivalente pour le dur. L'exemple 0.5 / votes `[0,1]` (MHR‑ECE 0 vs rater‑ECE 0.5) est correct.
- **NLL/Brier** : formules conformes ; Brier accumulateur float64 évite la cancellation, facteur 2 binaire et terme de désaccord `1-Σq²` corrects ; populations vides → NaN.
- **ETS étape 2** : `component_probs` renvoie les distributions complètes si `y.ndim==2`, `mixture_nll` fait la CE de la **mixture** ; le routage conditionnel reste par argmax de la prédiction. Les formats train/val durs/mous sont bien indépendants (`_prepare_validation` + `flatten_valid`).
- **Persistence** : pas de bump de `FORMAT_VERSION`, aucune clé nouvelle requise, blocs existants intacts ; validation isotone ajoutée uniquement pour `binary_isotonic_calibration` (monotonie stricte de `x`, non‑décroissance de `y`, dtypes float64, bornes [0,1], rejet NaN/vide/shape) ; états non‑fittés `x=y=None` acceptés.
- **One‑hot vs labels** : `flatten_valid` réduit un one‑hot exact au kernel dur ; l'entraînement soft garde toutes les classes.

### Améliorations facultatives
- Message d'erreur de `_voxels`/`validate_targets` quand `C>2` sur un `input_type="probs"` : pas de distinction « cible molle » vs « prédiction » — cosmétique.
- `reliability_diagram` annote désormais toujours « ECE (against targets) », y compris en top‑1 : libellé, pas calcul.
- `examples/multi_rater_calibration.py` : `mean-rater-ECE` agrège des ECE calculés sur des populations possiblement distinctes ; ici les masques sont identiques, mais la doc (`docs/multi_rater.md:110-112`) avertit déjà — factoriser serait plus sûr.

### Limites non vérifiées
- `conftest.py` n'est pas dans le bundle : je n'ai pas pu confirmer que `ALL_CALIBRATOR_IDS` **exclut** `binary_isotonic_calibration` des tests génériques (`make_calibrator(cid, max_iter=…, patience=…)` échouerait pour l'isotone, qui n'accepte ni `max_iter` ni early stopping). C'est une hypothèse nécessaire pour que la suite passe ; à confirmer.
- Chemins CUDA non exécutés (le bundle inclut `test_cuda.py` mais pas les kernels déterministes réels).
- L'identité avec le code original de TwinTrack et le choix exact des 250 bins ne sont pas vérifiables à partir du bundle ; la revendication « O(N) mémoire » n'a pas été mesurée.
- Je n'ai pas inspecté `TemperatureScaling._fit_core`, `vector/matrix/dirichlet` : ils ne figurent pas en entier ; je m'appuie sur les tests paramétrés soft/one‑hot pour leur compatibilité.

**Risque à retenir** : le seul écart réel est documentaire (préservation 0/1 hors plage apprise) ; le reste est sain.
