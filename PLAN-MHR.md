# Intégration de la calibration multi-rater vers la MHR

Plan préparé le 13 octobre 2026, complété le 17 octobre après vérification du
code de `d93570f`. Il décrit les changements à réaliser ; aucune fonctionnalité
MHR n'est encore implémentée. Le worktree `feat/mhr-calibration`, situé dans
`/tmp/fiducio-mhr-mimo`, pointe encore sur ce même commit sans diff de code.
Ce document planifie le travail ; il ne déclenche ni implémentation ni publication.

## Objectif et référence

Permettre à Fiducio de calibrer les sorties d'un modèle vers la proportion
d'annotateurs qui attribuent chaque classe à un voxel. En binaire, si deux
annotateurs sur cinq indiquent une tumeur, la cible vaut `[0.6, 0.4]`.

[TwinTrack v2, méthode et annexe A](https://arxiv.org/html/2604.15950v2)
justifie l'ajustement isotone contre cette moyenne. Le
[PDF complet, annexes B et D](https://arxiv.org/pdf/2604.15950v2) précise
250 bins de même effectif pour l'ajustement, 50 bins uniformes pour l'ECE,
calculé par annotateur puis moyenné, et un CRPS portant sur le volume tumoral.
La version HTML ne contient pas toutes ces annexes.

La MHR est une moyenne empirique des raters disponibles. Elle n'est pas une
vérité anatomique unique ni une estimation garantie pour une autre population
d'annotateurs. Les sorties restent des probabilités marginales par voxel.

## Attribution scientifique et citation dans la documentation

Citer explicitement **Kirscher et al., TwinTrack: Post-hoc Multi-Rater
Calibration for Medical Image Segmentation, MIDL 2026 (short paper)** pour
la calibration isotone binaire vers la MHR. La
[notice arXiv](https://arxiv.org/abs/2604.15950) confirme l'acceptation à MIDL
2026 ; utiliser la [version v2](https://arxiv.org/abs/2604.15950v2) comme
référence du protocole décrit dans ce plan.

L'attribution accompagne les fonctionnalités lors de leur implémentation :

| Emplacement | Citation attendue | Lot |
| --- | --- | --- |
| `docs/concepts.md`, section multirater/MHR | Définition de la MHR et motivation de la calibration, avec référence TwinTrack dès la première présentation | B |
| `docs/calibrators.md` et docstring de `BinaryIsotonicCalibration` visible dans l'API | Référence de l'approche isotone vers la MHR ; préciser les conventions d'intégration et la limite binaire | D |
| `docs/plots.md` et exemple multirater | Référence des diagnostics TwinTrack ; distinguer diagramme contre MHR et ECE moyen par rater | C / E |
| `README.md`, section MHR et section Citation | Lien vers TwinTrack près de l'exemple et référence bibliographique réutilisable pour les travaux utilisant cette approche | E |

Conserver la citation existante de *Rethinking Post-Hoc Calibration in Semantic
Segmentation* pour les méthodes paramétriques et la référence principale du
logiciel dans `CITATION.cff`. Présenter leur entraînement contre des cibles
souples comme l'extension proposée par Fiducio ; ne pas attribuer à TwinTrack
l'ensemble des neuf méthodes ni une reproduction complète de son pipeline.
Les liens locaux doivent suivre les conventions MkDocs existantes.

Entrée bibliographique minimale, sans inventer un volume ou une pagination
de proceedings non vérifiés :

```bibtex
@misc{kirscher2026twintrack,
  title         = {TwinTrack: Post-hoc Multi-Rater Calibration for Medical Image Segmentation},
  author        = {Kirscher, Tristan and Ertl, Alexandra and Maier-Hein, Klaus and
                   Coubez, Xavier and Meyer, Philippe and Faisan, Sylvain},
  year          = {2026},
  eprint        = {2604.15950},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {Accepted at MIDL 2026, short paper},
  url           = {https://arxiv.org/abs/2604.15950v2}
}
```

Critère de livraison : l'attribution est présente aux endroits ci-dessus,
les liens sont valides et les deux articles restent clairement associés aux
contributions correspondantes. La revue documentaire de E vérifie ce point.

## Contrat de données proposé

Conserver `fit(predictions, targets, mask=...)` et les classes existantes.
Reconnaître les deux formes exactes suivantes, sans deviner les axes :

| Entrée | Forme | Sens |
| --- | --- | --- |
| Prédictions | `(B,C,*spatial)` | Logits ou probabilités, contrat actuel |
| Cibles dures | `(B,*spatial)` | Labels entiers, contrat actuel |
| Cibles souples | `(B,C,*spatial)` | Distribution de classes, donc MHR possible |
| Annotations pour un helper dédié | `(B,R,*spatial)` | Label entier de chaque rater |

Les cibles souples doivent être flottantes, finies, dans `[0,1]` et sommer à 1
sur les classes aux positions valides. Un label fractionnaire de forme
`(B,*spatial)` reste invalide : il faut fournir les deux canaux en binaire.
`ignore_index` concerne les labels durs ; pour une MHR, la sélection des voxels
passe par `mask`. Préserver les cibles détachées de l'autograd.
Détecter le format par la forme avant toute conversion en entier. Conserver le
contrat actuel des labels durs flottants à valeurs entières. Pour les cibles
souples, reprendre la tolérance de somme des probabilités, sans normalisation
silencieuse. Des valeurs invalides dans des voxels masqués ne doivent contaminer
ni les pertes ni les métriques : sélectionner ces voxels avant le calcul, ou
neutraliser les valeurs avant les produits, car `0 * NaN` reste `NaN`.

Ajouter un helper `mean_human_response(annotations, num_classes=...,
ignore_index=-100)` retournant **la distribution et le masque des voxels ayant
au moins une annotation**. L'axe des raters est fixé à 1. `ignore_index` marque
une annotation absente ; aucun vote absent ne doit être compté comme background.
Valider les labels, ignorer les annotations absentes et accumuler les comptes
par classe sans matérialiser un one-hot pour chaque rater.
Retour : distribution float32 et masque booléen sur le device d'entrée ; la
distribution vaut zéro aux voxels sans annotation et doit donc toujours être
accompagnée du masque. Rejeter `num_classes < 2`, les booléens comme nombre de
classes, les labels fractionnaires/hors plage et les tenseurs sans axe rater.
Un axe rater vide produit une distribution nulle et un masque entièrement faux ;
le fit rejette ensuite une sélection sans voxel valide, comme aujourd'hui.

Exemple d'API projetée :

```python
# predictions: (B,C,*spatial), annotations: (B,R,*spatial)
mhr, annotated = mean_human_response(annotations, num_classes=2)
cal = TemperatureScaling(input_type="probs").fit(predictions, mhr, mask=annotated)
calibrated = cal.transform(test_predictions)
```

Chaque voxel valide a le même poids dans le premier contrat. Si le nombre de
raters varie, cette moyenne de pertes par voxel diffère d'une somme sur tous les
votes disponibles, qui donnerait plus de poids aux voxels davantage annotés.
La documenter explicitement. Les résultats TwinTrack supposent un nombre fixe
d'annotateurs ; leur équivalence ne justifie pas une pondération implicite des
annotations manquantes. Le biais dû à des annotations manquantes reste possible.

## Lot 1 : cibles souples sur les neuf calibrateurs

1. Dans `base.py` et `utils/tensors.py`, préparer et valider les deux formats,
   puis aplatir les cibles en `(N,)` ou `(N,C)`. La même sélection s'applique aux
   prédictions et aux cibles. Faire suivre ce contrat à la validation d'early stopping.
2. TS, les calibrateurs affines et les experts conditionnels utilisent déjà
   `F.cross_entropy` : PyTorch accepte les distributions `(N,C)` flottantes.
   Réutiliser cette opération après validation explicite des distributions.
3. Garder le routage des experts par les prédictions initiales. La MHR ne
   détermine pas l'expert, y compris pour `independent_experts=True`.
4. Adapter **la seconde étape ETS** : son optimisation actuelle extrait une
   seule probabilité avec `gather` sur le vrai label. Avec une cible `q`, il faut
   minimiser `-mean(sum_c q_c log(p_mix,c))`. Garder le chemin actuel pour les
   cibles dures ; le chemin souple requiert les composantes par classe et perd
   le gain mémoire de cette extraction. Ne pas moyenner les composantes contre
   `q` avant le logarithme : ce serait une autre perte.
   Le format des cibles d'entraînement et celui des cibles de validation sont
   indépendants : choisir le chemin dur/souple séparément pour chacun, jamais
   une seule fois d'après les cibles d'entraînement.
5. Ne pas modifier le format des checkpoints existants : ces cibles changent
   l'apprentissage, pas la forme des paramètres appris. Les exports du helper,
   les docstrings, README, guide, exemple et changelog doivent suivre.

À nombre fixe de raters, la cross-entropy contre la MHR est exactement la
moyenne des cross-entropies contre leurs labels. Cette identité fournit un
test scientifique simple. Elle ne transforme pas TS en régression isotone.

## Lot 2 : métriques cohérentes avec les cibles

- **NLL :** cibles dures inchangées ; cibles souples
  `-mean(sum_c q_c log(p_c))`, avec le plancher existant documenté.
- **Brier contre MHR :** définir pour les distributions
  `mean(sum_c (p_c-q_c)^2)`, et corriger le calcul numérique audité. Cela se
  réduit au contrat existant pour une cible one-hot. La moyenne des Brier contre
  les raters ajoute le terme `mean(1-sum_c q_c²)` ; les valeurs absolues diffèrent.
  Dans le cas binaire à deux canaux, la distance multiclasses vaut deux fois
  l'erreur quadratique du seul canal positif. Écrire ces conventions dans la doc.
- **Reliability top-1 existante :** pour les cibles souples, l'accord observé
  vaut `q[argmax(p)]`. Garder ses bins et son sens top-1.
- **Reliability de classe :** ajouter un chemin explicite pour une classe,
  qui compare `p_c` à `q_c` sur `[0,1]`. Il permet le diagramme tumoral de TwinTrack.
  Utiliser ce même chemin pour calculer l'ECE de classe contre chaque rater.
- **ECE contre MHR et ECE moyen des raters :** les nommer distinctement dans
  l'exemple et le rapport. Le second se calcule avant la moyenne des annotations.
  Les annotations individuelles doivent être conservées pour cette évaluation.
- **Plots :** reprendre les statistiques des courbes de classe et appeler la
  fréquence observée « proportion d'annotateurs », avec des annotations qui
  indiquent quel ECE est affiché.

Contre-exemple à tester : un voxel de probabilité tumorale 0,5, deux raters 0
et 1. L'ECE de classe contre MHR vaut 0 ; l'ECE moyen des raters vaut 0,5.
Cette différence vient de la valeur absolue et subsiste avec les mêmes bins.

API retenue : ajouter un argument keyword-only `class_index=None` à
`reliability_curve`, `expected_calibration_error`, `unweighted_calibration_error`
et `reliability_diagram`. `None` conserve le top-1 ; un entier valide choisit
la classe, avec observation `1[y=c]` pour un label dur ou `q_c` pour une MHR.
Conserver `ReliabilityCurve`, ses champs et les paramètres positionnels actuels ;
documenter que `bin_accuracy` devient une fréquence/proportion dans ce mode.
Propager le choix dans les wrappers et les plots, sans nouveau framework de métriques.
L'alias public `average_calibration_error` doit exposer le même argument que
`unweighted_calibration_error`, avec un test de compatibilité de l'alias.
Les bins vides restent à zéro dans les statistiques et hors du calcul ACE ;
une sélection vide conserve les conventions NaN actuelles.

L'ECE moyen par rater reste une boucle explicite dans l'exemple : mêmes
prédictions, classe, bords de bins et ROI, puis masque propre à chaque rater
si des annotations manquent. Faire la moyenne non pondérée des ECE des raters
évaluables, signaler leur nombre et les effectifs exclus ; ne jamais remplacer
un ECE NaN par zéro. Aucun rater évaluable donne NaN. Avec des masques différents,
les deux ECE n'ont plus exactement la même population d'évaluation : le préciser.

L'erreur quadratique voxel par voxel ne doit pas être appelée « CRPS TwinTrack ».
Le CRPS de l'article est une métrique de volume et requiert le protocole officiel.

## Lot 3 : calibrateur isotone binaire

Proposer `BinaryIsotonicCalibration(n_bins=250, input_type=...)` avec les mêmes
méthodes publiques et persistence que les calibrateurs existants. Le nom expose
sa limite binaire ; aucune renormalisation multiclasse indépendante n'est promise.
Il accepte des cibles dures et souples, en utilisant la classe positive 1.

Algorithme conforme au principe décrit dans l'article :

1. Convertir les scores en probabilités, sélectionner les voxels valides,
   puis trier le canal positif.
2. Regrouper d'abord les scores strictement identiques avec la somme des
   cibles et leur effectif. Former au plus `n_bins` bins contigus d'effectifs
   aussi proches que possible, sans couper un groupe d'égalités, puis calculer
   leur score moyen, leur MHR moyenne et leur effectif. Définir les coupures
   par les rangs cumulés `j*N/min(n_bins,N)` : déplacer chaque coupure à la fin
   du groupe qui contient ce rang, supprimer les coupures dupliquées et les
   bins vides. Les effectifs sont donc approximativement égaux en présence
   d'égalités ; le résultat ne dépend pas de l'ordre des voxels ex aequo.
3. Appliquer PAVA pondéré par les effectifs : fusionner les blocs voisins
   lorsque leurs valeurs ajustées sont décroissantes.
4. Interpoler linéairement entre les points ajustés, saturer aux valeurs des
   extrémités hors de l'intervalle appris, et retourner `[1-m(p), m(p)]`.
   Un unique score distinct produit une fonction constante.

Valider `n_bins` comme entier strictement positif, sans accepter un booléen.
Le PAVA minimise la somme pondérée des carrés sur les bins ; avec binning,
ce n'est pas nécessairement l'optimum isotone sur tous les voxels individuels.
Conserver une abscisse par bin après PAVA, même lorsque plusieurs valeurs
ajustées forment un plateau, pour définir l'interpolation sans ambiguïté.
Accumuler les sommes et effectuer PAVA en float64. Conserver les abscisses
apprises en float64 dans l'état et après rechargement, et interpoler avec ces
coordonnées : leur conversion en float32 pourrait fusionner deux moyennes de
bins distinctes et faire échouer un fit valide. L'inférence publique reste
float32. Ne pas ajouter d'optimiseur ou d'early
stopping à ce fit déterministe ; les arguments de validation restent rejetés
selon le contrat de la base lorsque l'early stopping est désactivé.

Implémenter PAVA sur les quelques centaines de bins avec les dépendances
existantes ; aucune dépendance scikit-learn n'est nécessaire. Le tri de tous les
voxels reste en `O(N log N)` avec mémoire `O(N)` ; l'agrégation ne rend pas le
fit streaming. Documenter une préparation reproductible de tables de voxels
pour les jeux trop volumineux, sans promettre une équivalence après sampling.

Réutiliser la validation, le masque, le refit transactionnel, le registre et les
outils de formes. Prévoir un chemin direct pour `transform` et `predict_proba`
isotones : le `safe_log` suivi de softmax du chemin actuel ne restitue pas les
probabilités exactement nulles ou égales à 1. Définir `decision_function` comme
les log-probabilités, autorisant `-inf` aux zéros ; les points appris sauvegardés
restent finis. Ne pas présenter ces infinis mathématiques comme un débordement.
Ne pas généraliser l'architecture à d'autres familles sans besoin concret.

Préserver aussi les probabilités d'entrée au **fit** : surcharger
`_to_canonical` dans le calibrateur isotone pour conserver les probabilités
directes ou appliquer softmax aux logits, sans passer par `safe_log`. Sinon,
les scores 0 et ceux inférieurs à `1e-12` seraient écrasés avant le tri. Ce
contrat doit être commun au fit et à l'inférence, en réutilisant la préparation
et le refit transactionnel de la base. Le contrôle numérique de la PR A doit
rejeter NaN et `+inf` sans rejeter les `-inf` mathématiques des log-probabilités
isotones, et sans relâcher la validation des paramètres sauvegardés.

Sauvegarder uniquement les abscisses et valeurs ajustées, avec un id stable.
Dans `persistence.py`, valider : tableaux 1D de même taille non nulle, abscisses
strictement croissantes dans `[0,1]`, valeurs non décroissantes dans `[0,1]`,
finitude, dtype et `num_classes=2`. L'ajout d'un nouvel id peut rester en format 1
si la structure du payload est inchangée ; les anciens lecteurs rejetteront
cet id inconnu et les nouveaux devront continuer de lire les anciens fichiers.

L'isotone préserve un ordre **non décroissant entre scores de voxels**, en créant
éventuellement des plateaux. Il peut changer une segmentation au seuil 0,5.
Cette garantie diffère de la conservation de l'argmax entre classes de CMSap.

Les conventions d'égalité, interpolation et extrémités ci-dessus sont des choix
d'intégration explicites. Leur identité avec le code original TwinTrack reste
à vérifier avant toute affirmation de reproduction exacte.

## Tests d'acceptation et fichiers concernés

| Partie | Tests utiles | Fichiers à atteindre |
| --- | --- | --- |
| Préparation | MHR 2/5, multiclasses, annotation absente, aucun rater, distributions invalides, masque | `base.py`, `utils/tensors.py`, `utils/__init__.py`, `tests/test_validation.py` |
| Fitting | Cible one-hot équivalente au label, CE-MHR égale à CE moyenne des raters, validation souple, refit échoué préservé | TS, `_affine.py`, `class_conditional.py`, tests calibrateurs/early stopping |
| ETS | Perte souple calculée sur la mixture, jamais logarithme d'une moyenne pondérée par q | `ensemble_temperature.py`, tests ETS |
| Métriques | NLL souple, conventions Brier, différence des deux ECE, masques, binaire facteur 2 | `metrics/calibration.py`, exports métriques, `tests/test_metrics.py`, plots |
| Isotone | PAVA connu, poids inégaux, scores identiques, un seul score, N inférieur aux bins, plateaux, extrémités 0/1, 2D/3D | Nouveau calibrateur et tests dédiés, exports calibrateurs et racine |
| Persistence | Roundtrip isotone, fichier corrompu, monotonicité invalide, anciens fichiers toujours lisibles | `persistence.py`, `tests/test_persistence.py`, fixtures existantes conservées |
| Exemple | Calibration/validation/test séparés par cas, pooling fixé avant fit, ECE par rater | `examples/multi_rater_calibration.py`, README, guides, API, changelog |

Les tests des calibrateurs doivent couvrir les neuf classes et les deux
optimiseurs, les experts conditionnels joints et indépendants, ainsi que la
validation souple. Garder les régressions de labels durs et de chargement 0.1.0.
Le test isotone doit comparer le résultat à un petit optimum calculable à la
main ; un simple test de forme ou de finitude ne vérifie pas PAVA.

## Ordre de livraison

1. Maintenance numérique et clarification des garanties, proposée pour `0.2.1`.
2. Cibles souples et helper MHR, avec adaptation ETS et validation.
3. Métriques et courbes explicitement définies pour ces cibles.
4. Isotone binaire, persistence, exemple complet et documentation ; proposer
   `0.3.0` lorsque les quatre lots sont validés.

## Découpage exécutable et critères de passage

**État au 17 octobre : prêt à implémenter.** Les trois remarques de la
relecture indépendante ont été intégrées, ainsi que l'exigence de citation
TwinTrack MIDL 2026. Il n'y a pas de décision scientifique ou d'API laissée
à l'utilisateur avant de commencer. L'absence de relecture OpenCode et de
comparaison au code original TwinTrack est consignée ; elle limite ce qu'on
peut affirmer sur une reproduction exacte, sans bloquer cette intégration.

Les PR ci-dessous sont séquentielles. Chaque PR inclut ses tests et la doc de
son API ; la dernière rassemble le parcours complet. Les versions restent
proposées : l'autorisation de rédiger ce plan ne vaut pas publication.

| PR | Contenu et fichiers principaux | Critère de passage |
| --- | --- | --- |
| A — Maintenance | `base.py`, `temperature.py` si nécessaire, Brier dans `metrics/calibration.py`, doc des garanties et tests de régression | Les exemples de débordement et de Brier de l'audit échouent explicitement ou donnent le résultat attendu ; aucune sortie NaN silencieuse aux positions valides |
| B — Contrat MHR | `utils/tensors.py`, `utils/__init__.py`, exports racine si cohérents avec les helpers existants, `base.py`, neuf calibrateurs dont ETS, tests validation/calibrateurs/early stopping | Tous les calibrateurs apprennent des cibles souples et gardent le comportement des labels durs ; helper correct avec votes absents et validation souple fonctionnelle |
| C — Métriques MHR | `metrics/calibration.py`, wrappers ECE/ACE, `plots/__init__.py`, exports/API si nécessaires, tests métriques/plots | NLL, Brier et courbes conformes aux formules ; exemple ECE 0 contre 0,5 vérifié ; top-1 inchangé par défaut |
| D — Isotone | `calibrators/isotonic.py`, exports calibrateurs/racine, `persistence.py`, tests dédiés et persistence | PAVA pondéré validé à la main, égalités déterministes, 0/1 exacts, masques et roundtrip fitted/unfitted conformes |
| E — Parcours et livraison | `examples/multi_rater_calibration.py`, README, `docs/concepts.md`, `docs/calibrators.md`, `docs/examples.md`, pages API et plots, `CHANGELOG.md`, liens/navigation MkDocs, scripts de release si leurs listes sont exhaustives | Exemple CPU autonome, documentation construite, tests et checks requis verts, anciens checkpoints lisibles ; version `0.3.0` seulement lors de la préparation de release |

Avant chaque PR, rechercher tous les appels des helpers modifiés, les exports,
les listes explicites de calibrateurs, les tests paramétrés et les fixtures.
Ne pas remplacer les fixtures 0.1.0 ni ajouter l'isotone à des tests supposant
nécessairement une optimisation par gradient ou une conservation de l'argmax.
`scripts/check_release.py`, `scripts/check_distribution.py`, les tests d'import
minimal et de métadonnées doivent être examinés pour les nouvelles API.
Le correctif Brier de A est conservé lors de l'extension aux cibles souples de C.

### Procédure recommandée

1. Vérifier l'état Git courant et les changements présents dans les deux
   worktrees. Conserver ce plan et sa relecture dans Git avant les changements
   de code. Partir d'une branche à jour sans écraser les travaux d'audit ; si
   A reprend des corrections déjà intégrées, ne conserver que le travail restant.
2. Exécuter les checks de référence dans l'environnement conforme au dépôt et
   noter les éventuels échecs préexistants. Livrer A avec ses régressions ciblées.
3. Implémenter B comme premier jalon fonctionnel : produire une MHR à partir
   des votes, ajuster les neuf méthodes existantes et préserver les labels durs.
   Valider ETS et l'early stopping avant de poursuivre vers les métriques.
4. Livrer C puis D sur cette base. Pour D, écrire d'abord les petits oracles
   PAVA/binning et les tests des extrémités, puis l'implémentation minimale.
5. Terminer par E, la vérification des citations, les checks complets et une
   revue du diff cumulé avant toute préparation de release. Une expérience
   réelle sur des cas séparés reste nécessaire avant de revendiquer un gain
   scientifique sur le dataset utilisateur.

Garder une PR par lot pour pouvoir contrôler chaque changement. Un échec de
test ciblé se résout dans son lot ; il ne doit pas être reporté au lot final.
Une version corrective intermédiaire peut être publiée si utile, mais sa
publication ne conditionne pas le démarrage de B. Ne pas attendre qu'OpenCode
redevienne disponible pour démarrer ; une relecture ultérieure peut compléter
les contrôles sans remplacer les critères de passage.

### Oracles numériques minimaux

- **Helper :** cinq labels dont deux positifs donnent `[0.6,0.4]` ; un label
  manquant ne compte ni au dénominateur ni comme background. Vérifier aussi
  trois classes, un rater, zéro rater et tous les votes absents.
- **CE :** sur des logits fixés et un nombre constant de raters, comparer la
  CE-MHR à la moyenne des CE individuelles, pertes et gradients. L'équivalence
  one-hot/labels porte aussi sur chaque calibrateur, avec tolérance numérique
  documentée ; comparer les prédictions plutôt que des paramètres non identifiables.
- **ETS :** calcul manuel sur une mixture à trois composantes, avec `q` non
  one-hot ; vérifier aussi le chemin de validation et la restauration du meilleur
  état, notamment entraînement dur/validation souple et l'inverse.
- **Brier :** préserver les deux cas proches de zéro de l'audit ; vérifier
  one-hot/label, le facteur 2 binaire et le terme de désaccord entre raters.
- **PAVA :** scores distincts, cibles `[0.8,0.2,0.9]`, poids `[1,3,1]` donnent
  `[0.35,0.35,0.9]`. Réaliser ce cas par des effectifs de voxels et des bins
  connus, puis tester séparément la règle de binning et l'interpolation.
- **Égalités :** permuter les annotations associées à des scores identiques
  ne change ni les points appris ni les sorties ; un score unique donne la
  moyenne des cibles. Couvrir `N<n_bins`, `n_bins=1` et un groupe traversant
  plusieurs coupures théoriques.
- **Précision isotone :** distinguer au fit les scores `0`, `1e-13` et `1e-12`
  avec des cibles croissantes et assez de bins ; couvrir des moyennes de bins
  distinctes en float64 qui coïncideraient après conversion float32, y compris
  après sauvegarde/rechargement.
- **Contrat complet :** logits/probabilités, NumPy/Torch, tables `(N,C)`,
  segmentation 2D/3D, entrées détachées, refit valide/échoué, padding NaN masqué,
  sélection vide et erreur pour `C != 2` en isotonique. Les NaN masqués en
  isotonique doivent être écartés avant tri et interpolation.
- **Persistence :** ancien format 1 conservé, nouveau roundtrip sur CPU et
  device demandé, état non ajusté accepté, état ajusté vide/incohérent rejeté,
  abscisses dupliquées, valeurs décroissantes, hors plage ou non finies rejetées.

### Commandes de validation

Utiliser l'environnement conforme au Python et au lockfile du dépôt ; ne pas
présenter un ancien environnement `--no-sync` comme une reproduction du lock.
Lancer d'abord les tests de la PR, puis une fois les lots intégrés :

```bash
uv run pytest --cov=fiducio --cov-report=term-missing
uv run ruff check .
uv run mypy
uv run --extra docs mkdocs build --strict
uv run python examples/multi_rater_calibration.py
```

Exécuter les scripts de vérification distribution/release selon leurs arguments
réels et le guide `docs/releasing.md` lors de la préparation de version. Garder
le seuil de couverture actuel (95 %). La CI existante doit vérifier les wheels,
les OS, les versions Python et le minimum Torch ; exécuter les tests CUDA sur
une machine disponible et déclarer explicitement leur absence sinon. Les neuf
méthodes sont couvertes avec Adam/L-BFGS et les deux modes d'experts ; l'isotone
utilise des tests déterministes dédiés.

## Exemple et protocole d'évaluation

L'exemple synthétique doit être exécutable sans téléchargement et sans données
patient. Il conserve les annotations individuelles, sépare les cas en
calibration/validation/test avant de sélectionner des voxels, fixe la seed et
compare le modèle non calibré, le fit sur un rater et le fit MHR pour TS et
l'isotone. Les tests paramétrés couvrent les autres calibrateurs ; l'exemple
n'a pas besoin d'une option CLI par méthode.

Rapporter sur le test NLL-MHR, Brier-MHR, ECE-MHR de classe et ECE moyen par
rater (50 bins uniformes), ainsi qu'un diagramme de classe. Pour comparer les
raters incomplets, indiquer la règle de masque et les effectifs. Distinguer la
moyenne sur tous les voxels d'une moyenne des métriques par cas ; le premier
exemple utilise le pooling voxel explicite. Ne promettre aucune amélioration
sur chaque métrique : l'isotone ajuste une erreur quadratique, les méthodes
paramétriques ajustent une CE, et leurs objectifs ne coïncident pas avec l'ECE.

Pour les données réelles, consigner les identifiants des splits, les raters,
la fusion des classes vers le binaire, la ROI, le pooling des ensembles avant
calibration, le sampling éventuel et sa seed, les versions et l'id du checkpoint.
Ne pas adapter les bins, la ROI ou le choix de méthode sur le test. Les masques
réduits à la lésion annotée ne doivent pas servir de ROI d'inférence implicite.
En cas de rééchantillonnage, la distribution cible est celle des voxels retenus ;
un sampling équilibrant artificiellement foreground/background change la
calibration attendue sur les volumes complets.

## Risques et limites décidées

- Le contrat est une moyenne égale par voxel, sans poids de confiance par rater,
  pondération patient ni estimation de qualité des annotateurs.
- Les neuf méthodes peuvent apprendre contre la MHR multiclasses ; cela ne
  garantit ni une calibration parfaite, ni la conservation de la segmentation.
- L'isotone de cette livraison est binaire. Les bins, les égalités et
  l'interpolation sont spécifiés ici ; seule une comparaison au code TwinTrack
  permettra de parler de reproduction exacte. Si ce code n'est pas accessible,
  la livraison reste possible en documentant cette limite.
- Le fit reste en mémoire. Le helper évite le one-hot par rater ; le chemin
  souple ETS matérialise davantage de probabilités que son chemin dur. Pas de
  streaming ni de benchmark de capacité promis dans cette version.
- Aucun changement de format des anciens checkpoints, aucune réécriture des
  fixtures historiques et aucune nouvelle dépendance runtime ne sont prévus.

## Relecture externe

Relecture tentée le 17 octobre 2026 avec OpenCode :

- `edelweiss/DeepSeek-V4.1-Flash` : refus fournisseur HTTP 403,
  `provider.auth`, « Inference is temporarily unavailable to this credential. »
- `opencode/mimo-v2.6-flash-free` : `provider.no-route`, modèle indisponible.
- Une instance `--standalone` liste le même accès DeepSeek ; aucun avis n'a
  donc été obtenu d'OpenCode et aucun réglage global n'a été changé.

Une seconde lecture a été menée en secours par un agent Codex indépendant sur
ce plan et les chemins locaux base/tensors/ETS/métriques/persistence/plots.
Ses trois remarques ont été intégrées : canonisation isotone sans `safe_log`
au fit, formats train/validation indépendants dans ETS et abscisses conservées
en float64. Le détail et les limites sont dans
[RELECTURE-PLAN-MHR.md](RELECTURE-PLAN-MHR.md). Aucun désaccord restant n'a été
identifié lors de cette revue. Elle ne remplace ni les tests numériques ni la
comparaison au code de TwinTrack.

Ce plan n'inclut pas le segmenter nnU-Net, les métriques vasculaires, le CRPS de
volume ni une reproduction des résultats CURVAS. L'évaluation réelle devra
séparer les patients, conserver les annotations individuelles, fixer le pooling
des ensembles et les masques/ROI, puis comparer non calibré, calibration contre
un rater et calibration MHR sur des cas de test indépendants.
