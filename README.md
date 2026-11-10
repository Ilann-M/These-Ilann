# Code de traitement et d'analyse des données de la thèse

**Étude exploratoire des effets à court terme d'un trajet à vélo à proximité du trafic sur les fonctions cognitives**
Ilann Mahou, thèse de doctorat de l'Université Gustave Eiffel (ED 476 NSCo), laboratoires LESCOT et UMRESTTE, 2026.

Ce dépôt rassemble le code de prétraitement et d'analyse statistique décrit au chapitre 3 de la thèse, afin que les traitements rapportés aux chapitres 3 et 4 puissent être reproduits. Il ne contient que le code des analyses rapportées dans le manuscrit.

> *English summary.* This repository contains the preprocessing and statistical analysis code of the PhD thesis *Exploratory study of the short-term effects of cycling near traffic on cognitive functions* (Université Gustave Eiffel, 2026). Participants cycled (conventional bike or e-bike) along a route close to traffic and a route away from it, then performed an N-back and an ANT-I task under EEG recording. The code covers EEG preprocessing and ERP extraction (MATLAB/EEGLAB), behavioural indicators, sensor fusion and noise descriptors, questionnaires, the merged subject table, descriptive statistics, the four levels of linear mixed-model / regression analyses with Benjamini–Hochberg FDR correction, and the robustness checks reported in the thesis. **The data are not public** (personal data under the GDPR), so the scripts cannot be run without them. Comments and messages are in French.

## Données

Les données ne sont pas diffusées : elles contiennent des données personnelles de santé et de localisation, collectées et conservées conformément au Règlement général sur la protection des données (chapitre 3, section 3.10). Les scripts ne peuvent donc pas être exécutés sans elles. Ils documentent les traitements appliqués.

Les métriques d'exposition aux polluants (concentrations moyennes, centiles et doses par trajet) ont été calculées en amont à partir des mesures des capteurs portables, hors de ce dépôt. Elles sont importées sous forme de tableaux par `05_tableau_combine/format_all3.py`.

## Organisation et ordre d'exécution

| Dossier | Contenu | Manuscrit |
|---|---|---|
| `00_taches/` | Scripts OpenSesame des tâches cognitives (1-back, 2-back, ANT-I), dans leur version de chaque campagne (`campagne_automne_2024/`, `campagne_printemps_2025/`), à ouvrir avec OpenSesame 4 | 3.5 |
| `01_pretraitement_EEG/` | Pipelines EEGLAB de l'ANT-I et du N-back (`Pipeline_ANT_final.m`, `Pipeline_NBack.m`), extraction des potentiels évoqués par régions d'intérêt (`extract_erp_*.m`) | 3.7 |
| `02_comportement/` | Indicateurs comportementaux du N-back (TR, d′) et des réseaux de l'ANT-I (`pre_traitement_all.py`), mise en forme (`mise_en_F.py`) | 3.5 |
| `03_trajets_bruit/` | Fusion des capteurs portables par campagne (`fusion_capteurs_automne_2024/`, `fusion_capteurs_printemps_2025/`) et descripteurs du bruit (LAeq, L10, L50, L90) | 3.4.5 à 3.4.7, 4.1.2 |
| `04_questionnaires/` | Traitement des questionnaires de vie et de passation | 3.6 |
| `05_tableau_combine/` | Fusion EEG, comportement, questionnaires et exposition en un tableau par participant (`format_all3.py`) | 3.9.2 |
| `06_descriptifs/` | Effectifs par niveau (tableau 4.1), contrôle de la manipulation et fréquence cardiaque (4.1.2), statistiques descriptives (tableau 4.3), caractérisation de l'échantillon par mode (tableau 4.4 : moyennes et tests dans `descriptifs_questionnaires.py`, g de Hedges dans `Comparabilite_groupes_mode.py`) | 3.4.7, 4.1 |
| `07_modeles/` | Niveau 1, analyse d'intention (`LMM_intention.py`) et ANOVA mixte de contrôle (`ANOVA_robustesse.py`) ; niveau 2, analyse corrective (`LMM_corr_FDR_2niveau_2.py`) ; niveau 3, exposition réétiquetée (`LMM_expoR_FDR_2niveau.py`) ; niveau 4, régressions continues (`Regression_continue_FDR_2niveau.py`) | 3.9, 4.2 à 4.5 |
| `08_robustesse/` | Sensibilité de puissance (3.2.4), analyses de sensibilité des deux effets principaux (tableau 4.5), facteurs confondants constants par participant et ressenti du trajet (4.2.3), confondants et corrélations de Spearman au niveau 4 | 3.2.4, 4.2.3, 4.5 |

Chaque script écrit dans un dossier de sortie daté, dont le nom correspond à celui des résultats rapportés dans le manuscrit.

Les pipelines EEG gardent un mode `runMode = 'freq'`, qui a servi à une analyse fréquentielle exploratoire non rapportée dans la thèse. Seul le mode `'erp'` correspond aux traitements décrits.

## Configuration

Aucun chemin n'est écrit en dur dans les scripts. Ils passent par `chemins.py` (Python) et `chemins.m` (MATLAB), à la racine du dépôt :

- `THESE_RACINE` : racine des données (arborescence `Data/`, `Campagne_aut_24/`, `Campagne_print_25/`) ;
- `THESE_SORTIES` (facultatif) : racine des sorties, par défaut `<THESE_RACINE>/Data/Traitement` ;
- `EEGLAB_DIR` : dossier d'installation d'EEGLAB.

Exemple sous Windows (PowerShell) :

```powershell
$env:THESE_RACINE = 'D:\donnees_these'
python 07_modeles\LMM_intention.py
```

## Environnement logiciel

- **Python** 3.11 ou plus récent. Dépendances dans `requirements.txt` (versions avec lesquelles les scripts ont été vérifiés). Les modèles mixtes sont ajustés avec `statsmodels` (`mixedlm`), la correction de Benjamini–Hochberg avec `statsmodels.stats.multitest`.
- **MATLAB** R2025b avec **EEGLAB** 2025.1 et les extensions *clean_rawdata* (ASR), *ICLabel* et *BIOSIG* (import des fichiers .bdf). Le traitement parallèle utilise la *Parallel Computing Toolbox*.

## Citation

Mahou, I. (2026). *Code de traitement et d'analyse des données de la thèse* [Logiciel]. https://github.com/Ilann-M/These-Ilann

Article associé : Mahou, I. et al. (2026). *Environmental Advances*, 100777. https://doi.org/10.1016/j.envadv.2026.100777

## Licence

Code distribué sous licence MIT (voir `LICENSE`).
