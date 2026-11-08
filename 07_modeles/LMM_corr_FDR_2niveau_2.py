# === CODAGE SOMME DES FACTEURS — corrige le 2026-09-09 ===
# Le modele Mode x Trajet etait ajuste en codage treatment (reference 'Bike'), de sorte que
# le terme lu comme 'effet principal du Trajet' etait l'effet du trajet CHEZ LES CYCLISTES
# CLASSIQUES — un effet simple. Le codage somme C(..., Sum) rend les termes marginaux.
# Les post-hocs par mode et les tailles d'effet (calculees sur les donnees) sont inchanges.
# ATTENTION : C(Trajet, Sum)[S.Away] a le signe OPPOSE a Trajet[T.Close].
# Sauvegarde de la version anterieure dans Data\Codes\_Archives\.
"""
════════════════════════════════════════════════════════════════════════════════
ANALYSE LMM CORRECTIVE - FILTRAGE SUR COHÉRENCE D'EXPOSITION
Version FDR par TÂCHE (N-Back : 1B + 2B  /  ANT)

Principe :
  Même modèle LMM 2×2 que l'analyse d'intention
  (Valeur ~ Mode_transport * Trajet), mais restreint aux sujets dont
  le rapport d'exposition Close/Away va dans le sens attendu :
      mesure_C >= RATIO_MIN × mesure_A   (par défaut RATIO_MIN = 1.1)

Quatre sous-analyses indépendantes (une par critère d'exposition) :
    1) BC880_Mean_A_EXP  vs BC880_Mean_C_EXP
    2) BC880_P95_A_EXP   vs BC880_P95_C_EXP
    3) Number_Mean_A_EXP vs Number_Mean_C_EXP
    4) Number_P95_A_EXP  vs Number_P95_C_EXP

Pour chaque sous-analyse, deux niveaux (= tâches) :
  N-BACK (8 tests : 1B + 2B regroupés) :
    EEG         : N1_Moy + P3_Moy (Valides) × 1B / 2B = 4
    Comportement: RT_moyen (log) + d_prime × 1B / 2B = 4
    → Correction FDR Benjamini-Hochberg sur 8 tests

  ANT (18 tests) :
    EEG flèche  : N1_Moy + P3_Moy × 5 réseaux = 10
    EEG sonore  : N1_Moy + P2_Moy (Alerte Standard) = 2
    EEG indice  : N1_Moy (Alerte Visuelle) = 1
    Comportement: RT (log) × 5 réseaux = 5
    → Correction FDR Benjamini-Hochberg sur 18 tests

Post-hoc : Effet Trajet par Mode (Bonferroni ×2) si interaction sig.

Sélection initiale : Selection_4 (N=51), puis filtrage cohérence exposition.

Sorties :
  <racine>/Data/Traitement/Resultats_LMM_correction/<date>_FDR_taches/<critere>/
════════════════════════════════════════════════════════════════════════════════
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402

import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

# Console UTF-8 (caractères de cadre / accents dans les print)
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_FILE = D(r'Data\Table_combine\Tab_sujet_20260205.xlsx')
OUTPUT_DIR = S(r'Resultats_LMM_correction\26-10-16_concentration_sans134')

VERSION_DONNEES = 'filtre'

SELECTION_4 = [101, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
               201, 202, 203, 204, 205, 206, 209, 210, 211, 212, 215, 216,
               121, 122, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 139,
               221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]

ALPHA = 0.05
METHODE_CORRECTION = 'fdr_bh'

STRATEGIE_NAN = 'paires_completes'

# ── Filtrage cohérence exposition ──
RATIO_MIN = 1.1

CRITERES_EXPOSITION = [
    {'nom': 'BC880_Mean',  'col_A': 'BC880_Mean_A_EXP',  'col_C': 'BC880_Mean_C_EXP'},
    {'nom': 'Number_Mean', 'col_A': 'Number_Mean_A_EXP', 'col_C': 'Number_Mean_C_EXP'},
]


# ==============================================================================
# DÉFINITION DES INDICATEURS — REGROUPÉS PAR TÂCHE
# ==============================================================================

def generer_indicateurs_nback():
    """
    N-BACK : 8 indicateurs (1B + 2B, EEG + Comportement mélangés).
    FDR appliqué sur ces 8 tests.
    """
    indicateurs = []

    # ── EEG : N1 et P3 Valides × 1B/2B (4 tests) ──
    for tache in ['1B', '2B']:
        for composante in ['N1', 'P3']:
            indicateurs.append({
                'nom': f'{composante}_Moy_Valide_{tache}',
                'tache': tache,
                'domaine': 'EEG',
                'col_away': f'{composante}_Moy_Valide_A_{tache}',
                'col_close': f'{composante}_Moy_Valide_C_{tache}',
                'transformation': None
            })

    # ── Comportement : RT_moyen (log) + d' × 1B/2B (4 tests) ──
    for tache in ['1B', '2B']:
        indicateurs.append({
            'nom': f'RT_moyen_{tache}',
            'tache': tache,
            'domaine': 'Comportement',
            'col_away': f'RT_moyen_{VERSION_DONNEES}_A_{tache}',
            'col_close': f'RT_moyen_{VERSION_DONNEES}_C_{tache}',
            'transformation': 'log'
        })
        indicateurs.append({
            'nom': f'd_prime_{tache}',
            'tache': tache,
            'domaine': 'Comportement',
            'col_away': f'd_prime_{VERSION_DONNEES}_A_{tache}',
            'col_close': f'd_prime_{VERSION_DONNEES}_C_{tache}',
            'transformation': None
        })

    return indicateurs


def generer_indicateurs_ant():
    """
    ANT : 18 indicateurs (EEG + Comportement mélangés).
    FDR appliqué sur ces 18 tests.
    """
    indicateurs = []

    # ── EEG flèche : N1 + P3 × 5 réseaux (10 tests) ──
    reseaux_eeg_fleche = {
        'Alerte_Standard': 'Alerte_Standard',
        'Alerte_Auditive': 'Alerte_Auditive',
        'Alerte_Visuelle': 'Alerte_Visuelle',
        'Orientation': 'Orientation',
        'Controle_Executif': 'Controle'
    }
    for reseau_label, condition_eeg in reseaux_eeg_fleche.items():
        for composante in ['N1', 'P3']:
            indicateurs.append({
                'nom': f'{composante}_Moy_fleche_{reseau_label}_ANT',
                'tache': 'ANT',
                'domaine': 'EEG',
                'col_away': f'{composante}_{condition_eeg}_Moy_fleche_A_ANT',
                'col_close': f'{composante}_{condition_eeg}_Moy_fleche_C_ANT',
                'transformation': None
            })

    # ── EEG sonore : N1 + P2 (Alerte Standard) (2 tests) ──
    for composante in ['N1', 'P2']:
        indicateurs.append({
            'nom': f'{composante}_Moy_sonore_Alerte_Standard_ANT',
            'tache': 'ANT',
            'domaine': 'EEG',
            'col_away': f'{composante}_Alerte_Standard_Moy_sonore_A_ANT',
            'col_close': f'{composante}_Alerte_Standard_Moy_sonore_C_ANT',
            'transformation': None
        })

    # ── EEG indice : N1 (Alerte Visuelle) (1 test) ──
    indicateurs.append({
        'nom': 'N1_Moy_indice_Alerte_Visuelle_ANT',
        'tache': 'ANT',
        'domaine': 'EEG',
        'col_away': 'N1_Alerte_Visuelle_Moy_indice_A_ANT',
        'col_close': 'N1_Alerte_Visuelle_Moy_indice_C_ANT',
        'transformation': None
    })

    # ── Comportement : RT × 5 réseaux (log) (5 tests) ──
    reseaux_rt = [
        'Alerte_Standard', 'Alerte_Auditive', 'Alerte_Visuelle',
        'Orientation', 'Controle_Executif'
    ]
    for reseau in reseaux_rt:
        indicateurs.append({
            'nom': f'RT_{reseau}_ANT',
            'tache': 'ANT',
            'domaine': 'Comportement',
            'col_away': f'RT_{reseau}_A_ANT',
            'col_close': f'RT_{reseau}_C_ANT',
            # echelle brute : score de difference, pas un temps de reaction
            # brut ; trois des cinq contrastes comportent des valeurs
            # negatives, incompatibles avec le log (correction 2026-08-12)
            'transformation': None
        })

    # ── ANT Comportement : ER (taux d'erreur) × 5 réseaux (5) ──
    # Ajouté le 2026-08-12 : le jeu d'indicateurs doit être identique aux quatre
    # niveaux d'analyse, faute de quoi les familles FDR n'ont pas la même taille
    # et les niveaux ne sont plus comparables entre eux (chapitre 3, section 3.9.2).
    # transformation=None : les ER sont analysés en échelle brute, comme au niveau 1.
    for reseau in reseaux_rt:
        indicateurs.append({
            'nom': f'ER_{reseau}_ANT',
            'tache': 'ANT',
            'domaine': 'Comportement',
            'col_away': f'ER_{reseau}_A_ANT',
            'col_close': f'ER_{reseau}_C_ANT',
            'transformation': None,
        })

    return indicateurs


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def creer_dossiers(output_dir):
    """Crée la structure de dossiers par tâche et par domaine."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'NBack')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'ANT')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'par_domaine')).mkdir(parents=True, exist_ok=True)


def charger_donnees(fichier_path):
    """Charge les données"""
    print(f"Chargement: {fichier_path}...")
    df = pd.read_excel(fichier_path)
    _ordre = _charger_ordre()
    # Session 2-back du trajet eloigne du participant 134 : boitier de reponse defaillant (0 reponse).
    # Retiree des indicateurs comportementaux du 2-back, comme au niveau 1 (LMM_intention.py, chapitre 3).
    df.loc[df['Participant'] == 134, [c for c in df.columns if c.startswith('d_prime_') and c.endswith('_A_2B')]] = float('nan')
    if _ordre is not None and 'Participant' in df.columns:
        df = df.merge(_ordre, on='Participant', how='left')
    print(f"  OK: {df.shape[0]} participants, {df.shape[1]} colonnes")
    return df


def filtrer_selection(df, selection_ids):
    """Filtre pour la sélection"""
    return df[df['Participant'].isin(selection_ids)].copy()


def filtrer_coherence_exposition(df, col_A, col_C, ratio_min):
    """
    Filtre les sujets dont le rapport Close/Away est cohérent :
        df[col_C] >= ratio_min * df[col_A]

    Les sujets avec NaN sur col_A ou col_C sont exclus.
    """
    stats = {
        'n_initial': len(df),
        'n_nan': 0,
        'n_inverse_ou_insuffisant': 0,
        'n_final': 0,
        'participants_gardes': [],
        'participants_exclus_nan': [],
        'participants_exclus_inverse': [],
    }

    if col_A not in df.columns or col_C not in df.columns:
        raise KeyError(f"Colonnes manquantes : {col_A} et/ou {col_C}")

    mask_nan = df[col_A].isna() | df[col_C].isna()
    stats['n_nan'] = int(mask_nan.sum())
    stats['participants_exclus_nan'] = df.loc[mask_nan, 'Participant'].tolist()

    df_non_nan = df[~mask_nan].copy()
    mask_coherent = df_non_nan[col_C] >= ratio_min * df_non_nan[col_A]

    stats['n_inverse_ou_insuffisant'] = int((~mask_coherent).sum())
    stats['participants_exclus_inverse'] = df_non_nan.loc[~mask_coherent, 'Participant'].tolist()

    df_filtre = df_non_nan[mask_coherent].copy()
    stats['n_final'] = len(df_filtre)
    stats['participants_gardes'] = df_filtre['Participant'].tolist()

    return df_filtre, stats


# ==============================================================================
# GESTION DES NaN ET TRANSFORMATIONS
# ==============================================================================

def nettoyer_donnees_paire(df, col_away, col_close):
    """Nettoie les données (paires complètes)"""
    stats_nan = {
        'n_initial': len(df),
        'n_nan_away': df[col_away].isna().sum(),
        'n_nan_close': df[col_close].isna().sum(),
        'participants_exclus': [],
        'n_final': 0,
        'pct_perte': 0
    }

    mask_incomplet = df[col_away].isna() | df[col_close].isna()
    if mask_incomplet.sum() > 0:
        stats_nan['participants_exclus'] = df.loc[mask_incomplet, 'Participant'].tolist()

    df_clean = df[~mask_incomplet].copy()
    stats_nan['n_final'] = len(df_clean)
    stats_nan['pct_perte'] = (
        (stats_nan['n_initial'] - stats_nan['n_final']) / stats_nan['n_initial'] * 100
        if stats_nan['n_initial'] > 0 else 0
    )
    return df_clean, stats_nan


def appliquer_transformation(valeurs, type_transformation):
    """Applique la transformation"""
    if type_transformation is None or type_transformation == 'none':
        return valeurs
    valeurs_clean = valeurs.copy()
    if type_transformation == 'log':
        if (valeurs_clean <= 0).any():
            min_pos = valeurs_clean[valeurs_clean > 0].min() if (valeurs_clean > 0).any() else 0.001
            valeurs_clean = valeurs_clean.clip(lower=min_pos * 0.001)
        return np.log(valeurs_clean)
    return valeurs_clean


def preparer_donnees_long(df, col_away, col_close, transformation=None):
    """Convertit en format long"""
    df_clean, stats_nan = nettoyer_donnees_paire(df, col_away, col_close)
    if len(df_clean) == 0:
        return None, stats_nan

    valeurs_away = appliquer_transformation(df_clean[col_away], transformation)
    valeurs_close = appliquer_transformation(df_clean[col_close], transformation)

    df_away = df_clean[[c for c in ('Participant', 'Mode_transport', 'Ordre')
                        if c in df_clean.columns]].copy()
    df_away['Trajet_reel'] = 'Away'
    df_away['Valeur'] = valeurs_away
    df_away['Trajet'] = 'Away'

    df_close = df_clean[[c for c in ('Participant', 'Mode_transport', 'Ordre')
                        if c in df_clean.columns]].copy()
    df_close['Trajet_reel'] = 'Close'
    df_close['Valeur'] = valeurs_close
    df_close['Trajet'] = 'Close'

    df_long = pd.concat([df_away, df_close], ignore_index=True)
    df_long = df_long.dropna(subset=['Valeur'])
    return _ajouter_session(df_long), stats_nan


# ==============================================================================
# FONCTIONS STATISTIQUES
# ==============================================================================

def tester_normalite(valeurs, seuil_p=0.05):
    """Tests de normalité"""
    resultats = {'conclusion': 'INSUFFISANT', 'p_shapiro': np.nan, 'p_dagostino': np.nan}
    if len(valeurs) < 3:
        return resultats

    if len(valeurs) <= 5000:
        try:
            _, p_sw = stats.shapiro(valeurs)
            resultats['p_shapiro'] = p_sw
        except:
            pass

    if len(valeurs) >= 20:
        try:
            _, p_dp = stats.normaltest(valeurs)
            resultats['p_dagostino'] = p_dp
        except:
            pass

    tests = []
    if not np.isnan(resultats['p_shapiro']):
        tests.append(resultats['p_shapiro'] > seuil_p)
    if not np.isnan(resultats['p_dagostino']):
        tests.append(resultats['p_dagostino'] > seuil_p)

    if tests:
        n_normal = sum(tests)
        if n_normal == len(tests):
            resultats['conclusion'] = 'NORMAL'
        elif n_normal >= len(tests) / 2:
            resultats['conclusion'] = 'PROBABLEMENT_NORMAL'
        else:
            resultats['conclusion'] = 'NON_NORMAL'

    return resultats


def diagnostiquer_modele(model_result, df_long):
    """Diagnostics du modèle"""
    diagnostics = {}
    try:
        residus = model_result.resid
        normalite = tester_normalite(residus)
        diagnostics['normalite_conclusion'] = normalite['conclusion']
        diagnostics['p_shapiro'] = normalite['p_shapiro']
        diagnostics['p_dagostino'] = normalite['p_dagostino']
        diagnostics['skewness'] = stats.skew(residus)
        diagnostics['kurtosis'] = stats.kurtosis(residus)

        try:
            groupes = [residus[df_long['Trajet'] == t].values for t in df_long['Trajet'].unique()]
            _, p_levene = stats.levene(*groupes)
            diagnostics['homoscedasticite_p'] = p_levene
        except:
            diagnostics['homoscedasticite_p'] = np.nan

        z_scores = np.abs((residus - residus.mean()) / residus.std())
        diagnostics['n_outliers'] = int((z_scores > 3).sum())
    except Exception as e:
        diagnostics['erreur'] = str(e)
    return diagnostics


def cohen_d_av(df_long, facteur, niv_ref, niv_cible):
    """Cohen's d_av intra-sujet (Lakens, 2013) : différence de moyennes brutes
    rapportée à la MOYENNE des SD des deux conditions (référence = variance
    totale, comparable aux seuils 0.2/0.5/0.8). Signe : cible - reference."""
    v_ref = df_long.loc[df_long[facteur] == niv_ref, 'Valeur']
    v_cible = df_long.loc[df_long[facteur] == niv_cible, 'Valeur']
    if len(v_ref) < 2 or len(v_cible) < 2:
        return np.nan
    sd_moy = (v_ref.std(ddof=1) + v_cible.std(ddof=1)) / 2
    return (v_cible.mean() - v_ref.mean()) / sd_moy if sd_moy > 0 else np.nan


def cohen_d_s(df_long, facteur, niv_ref, niv_cible):
    """Cohen's d_s inter-sujet : agrégation par participant (moyenne sur les
    conditions intra) puis SD poolée des deux groupes. Signe : cible - reference."""
    moyennes = df_long.groupby('Participant', as_index=False).agg(
        Valeur=('Valeur', 'mean'), grp=(facteur, 'first'))
    g_ref = moyennes.loc[moyennes['grp'] == niv_ref, 'Valeur']
    g_cible = moyennes.loc[moyennes['grp'] == niv_cible, 'Valeur']
    n1, n2 = len(g_ref), len(g_cible)
    if n1 < 2 or n2 < 2:
        return np.nan
    sd_pool = np.sqrt(((n1 - 1) * g_ref.std(ddof=1) ** 2 +
                       (n2 - 1) * g_cible.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return (g_cible.mean() - g_ref.mean()) / sd_pool if sd_pool > 0 else np.nan


def cohen_dz(df_long, facteur, niv_ref, niv_cible):
    """Cohen's dz intra-sujet (Lakens, 2013) : moyenne des différences appariées
    rapportée à la SD des différences (référence INTRA ; dépend de la corrélation
    entre conditions, NON comparable entre études). Signe : cible - reference."""
    piv = df_long.pivot_table(index='Participant', columns=facteur, values='Valeur')
    if niv_ref not in piv.columns or niv_cible not in piv.columns:
        return np.nan
    diff = (piv[niv_cible] - piv[niv_ref]).dropna()
    if len(diff) < 2:
        return np.nan
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else np.nan


def calculer_taille_effet(model_result, effet_type, df_long):
    """Tailles d'effet adaptées au plan de l'effet.

    - Trajet (intra-sujet) -> dz (reporté) + d_av (variance totale, colonne
      silencieuse / comparable) (Lakens, 2013).
    - Mode   (inter-sujet) -> Cohen's d_s poolé (dz non défini en inter).
    - Interaction          -> pas de d ; crossover porté par eta2 + post-hocs.

    eta2 : eta2 PARTIEL = t²/(t²+df_resid), cadre intra/signal (cohérent avec
    dz). Ce n'est PAS un eta2 généralisé."""
    try:
        clefs = {
            'Mode': 'C(Mode_transport, Sum)[S.Bike]',
            'Trajet': 'C(Trajet, Sum)[S.Away]',
            'Interaction': 'C(Mode_transport, Sum)[S.Bike]:C(Trajet, Sum)[S.Away]',
        }
        coef_key = clefs[effet_type]
        t_value = model_result.tvalues.get(coef_key, np.nan)
        eta2 = (t_value ** 2) / (t_value ** 2 + model_result.df_resid) \
            if not np.isnan(t_value) else np.nan

        if effet_type == 'Trajet':
            cohens_d = cohen_d_av(df_long, 'Trajet', 'Away', 'Close')
            dz = cohen_dz(df_long, 'Trajet', 'Away', 'Close')
        elif effet_type == 'Mode':
            cohens_d = cohen_d_s(df_long, 'Mode_transport', 'Bike', 'e-Bike')
            dz = np.nan  # facteur inter-sujet : dz non défini
        else:  # Interaction : pas de d interprétable
            cohens_d, dz = np.nan, np.nan
        return {'cohens_d': cohens_d, 'dz': dz, 'eta2': eta2}
    except Exception:
        return {'cohens_d': np.nan, 'dz': np.nan, 'eta2': np.nan}


# ==============================================================================
# AJUSTEMENT LMM
# ==============================================================================

# ==============================================================================
# TERME DE SESSION (periode) — ajout du 2026-08-12
# ==============================================================================
# Le plan est un croise a deux periodes : la session de test est un facteur
# intra-participant, determine par l'ordre d'attribution des itineraires et par
# l'itineraire effectivement realise. Un effet de pratique residuel subsiste
# entre les deux sessions de test sur la tache 2-back, alors que la seance
# d'habituation etait censee le neutraliser (chapitre 3, sections 3.3.4 et
# 3.9.2). Le terme est donc introduit dans le modele.
AVEC_SESSION = True
BASE_ORDRE = D(r'Data\Base.xlsx')


def _charger_ordre():
    """Participant -> 'A_first' / 'C_first' (condition de la 1re session de test)."""
    try:
        b = pd.read_excel(BASE_ORDRE)
    except Exception:
        return None
    b.columns = [str(c).strip() for c in b.columns]
    col = 'Passation-2' if 'Passation-2' in b.columns else 'Passation_2'
    if col not in b.columns or 'Participant' not in b.columns:
        return None
    b['Participant'] = b['Participant'].astype(str).str.extract(r'[Pp](\d+)')[0]
    b = b.dropna(subset=['Participant'])
    b['Participant'] = b['Participant'].astype(int)
    p2 = b[col].astype(str).str.strip().str.upper()
    b['Ordre'] = np.where(p2 == 'A', 'A_first',
                          np.where(p2 == 'C', 'C_first', np.nan))
    return b[['Participant', 'Ordre']].dropna()


def _ajouter_session(df_long):
    """Ajoute Session : S2 = 1re session de test, S3 = 2e session de test."""
    if not AVEC_SESSION or df_long is None:
        return df_long
    if 'Ordre' not in df_long.columns or 'Trajet_reel' not in df_long.columns:
        return df_long
    df_long = df_long.copy()
    df_long['Session'] = np.where(
        ((df_long['Ordre'] == 'A_first') & (df_long['Trajet_reel'] == 'Away')) |
        ((df_long['Ordre'] == 'C_first') & (df_long['Trajet_reel'] == 'Close')),
        'S2', 'S3')
    return df_long


def _formule(df_long, base):
    """Ajoute le terme de session a la formule s'il est disponible et informatif."""
    if (AVEC_SESSION and df_long is not None and 'Session' in df_long.columns
            and df_long['Session'].nunique() > 1):
        return base + ' + Session'
    return base


def ajuster_lmm(df_long):
    """LMM avec fallback (pente aléatoire → intercept seul)"""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = mixedlm(_formule(df_long, "Valeur ~ C(Mode_transport, Sum) * C(Trajet, Sum)"), df_long,
                            groups=df_long["Participant"], re_formula="~Trajet")
            result = model.fit(method='lbfgs', maxiter=200)
            if result.converged:
                return result, 'LMM_pente'
    except:
        pass

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = mixedlm(_formule(df_long, "Valeur ~ C(Mode_transport, Sum) * C(Trajet, Sum)"), df_long,
                            groups=df_long["Participant"])
            result = model.fit(method='lbfgs', maxiter=200)
            if result.converged:
                return result, 'LMM_intercept'
    except:
        pass

    return None, 'Echec'


# ==============================================================================
# POST-HOC
# ==============================================================================

def tester_effet_trajet_par_mode(df_long):
    """Post-hoc : effet Trajet par Mode (Bonferroni ×2)"""
    resultats = {}

    for mode in ['Bike', 'e-Bike']:
        df_mode = df_long[df_long['Mode_transport'] == mode].copy()
        res = {'mode': mode}

        if len(df_mode) < 10:
            res['erreur'] = 'N insuffisant'
            resultats[mode] = res
            continue

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model = mixedlm(_formule(df_mode, "Valeur ~ Trajet"), df_mode,
                                groups=df_mode["Participant"])
                model_fit = model.fit(method='lbfgs', maxiter=200)

            if not model_fit.converged:
                res['erreur'] = 'Non convergé'
                resultats[mode] = res
                continue

            if 'Trajet[T.Close]' in model_fit.pvalues:
                p_val = model_fit.pvalues['Trajet[T.Close]']
                coef = model_fit.params['Trajet[T.Close]']

                res['p_value'] = p_val
                res['p_corrige'] = min(p_val * 2, 1.0)
                res['coefficient'] = coef
                res['cohens_d'] = cohen_d_av(df_mode, 'Trajet', 'Away', 'Close')
                res['cohens_dz'] = cohen_dz(df_mode, 'Trajet', 'Away', 'Close')
                res['direction'] = 'Close > Away' if coef > 0 else 'Away > Close'
                res['significatif'] = res['p_corrige'] < ALPHA
            else:
                res['erreur'] = 'Paramètre non trouvé'
        except Exception as e:
            res['erreur'] = str(e)

        resultats[mode] = res
    return resultats


# ==============================================================================
# ANALYSE D'UN INDICATEUR
# ==============================================================================

def analyser_indicateur(df, config_indic):
    """Analyse complète d'un indicateur"""
    nom = config_indic['nom']
    col_away = config_indic['col_away']
    col_close = config_indic['col_close']
    transformation = config_indic['transformation']
    tache = config_indic['tache']
    domaine = config_indic['domaine']

    print(f"  {nom}...", end=" ")

    if col_away not in df.columns or col_close not in df.columns:
        print(f"SKIP (colonnes manquantes)")
        return None

    df_long, stats_nan = preparer_donnees_long(df, col_away, col_close, transformation)
    if df_long is None or len(df_long) < 20:
        print("SKIP (N insuffisant)")
        return None

    model_result, type_modele = ajuster_lmm(df_long)
    if model_result is None:
        print("ECHEC (non convergé)")
        return None

    pvalues = model_result.pvalues
    tvalues = model_result.tvalues

    effet_mode = calculer_taille_effet(model_result, 'Mode', df_long)
    effet_trajet = calculer_taille_effet(model_result, 'Trajet', df_long)
    effet_inter = calculer_taille_effet(model_result, 'Interaction', df_long)

    diag = diagnostiquer_modele(model_result, df_long)
    post_hoc = tester_effet_trajet_par_mode(df_long)
    bike_ph = post_hoc.get('Bike', {})
    ebike_ph = post_hoc.get('e-Bike', {})

    resultat = {
        'Indicateur': nom,
        'Tache': tache,
        'Domaine': domaine,
        'Transformation': transformation if transformation else 'aucune',
        'N_obs': len(df_long),
        'N_participants': df_long['Participant'].nunique(),
        'N_NaN_away': stats_nan['n_nan_away'],
        'N_NaN_close': stats_nan['n_nan_close'],
        'Pct_perte_donnees': stats_nan['pct_perte'],

        'F_Mode': tvalues.get('C(Mode_transport, Sum)[S.Bike]', np.nan) ** 2,
        'p_Mode': pvalues.get('C(Mode_transport, Sum)[S.Bike]', np.nan),
        'eta2_Mode': effet_mode['eta2'],
        'd_Mode': effet_mode['cohens_d'],

        'F_Trajet': tvalues.get('C(Trajet, Sum)[S.Away]', np.nan) ** 2,
        'p_Trajet': pvalues.get('C(Trajet, Sum)[S.Away]', np.nan),
        'eta2_Trajet': effet_trajet['eta2'],
        'd_Trajet': effet_trajet['cohens_d'],
        'dz_Trajet': effet_trajet['dz'],

        'F_Interaction': tvalues.get('C(Mode_transport, Sum)[S.Bike]:C(Trajet, Sum)[S.Away]', np.nan) ** 2,
        'p_Interaction': pvalues.get('C(Mode_transport, Sum)[S.Bike]:C(Trajet, Sum)[S.Away]', np.nan),
        'eta2_Interaction': effet_inter['eta2'],
        'd_Interaction': effet_inter['cohens_d'],

        'Convergence': True,
        'Type_modele': type_modele,

        'Normalite_residus': diag.get('normalite_conclusion', 'N/A'),
        'p_Shapiro_Wilk': diag.get('p_shapiro', np.nan),
        'p_DAgostino_Pearson': diag.get('p_dagostino', np.nan),
        'Homoscedasticite_p': diag.get('homoscedasticite_p', np.nan),
        'N_outliers': diag.get('n_outliers', 0),

        'PostHoc_Bike_p': bike_ph.get('p_value', np.nan),
        'PostHoc_Bike_p_corr': bike_ph.get('p_corrige', np.nan),
        'PostHoc_Bike_d': bike_ph.get('cohens_d', np.nan),
        'PostHoc_Bike_dz': bike_ph.get('cohens_dz', np.nan),
        'PostHoc_Bike_sig': bike_ph.get('significatif', False),
        'PostHoc_Bike_direction': bike_ph.get('direction', 'N/A'),

        'PostHoc_eBike_p': ebike_ph.get('p_value', np.nan),
        'PostHoc_eBike_p_corr': ebike_ph.get('p_corrige', np.nan),
        'PostHoc_eBike_d': ebike_ph.get('cohens_d', np.nan),
        'PostHoc_eBike_dz': ebike_ph.get('cohens_dz', np.nan),
        'PostHoc_eBike_sig': ebike_ph.get('significatif', False),
        'PostHoc_eBike_direction': ebike_ph.get('direction', 'N/A'),
    }

    print("OK")
    return resultat


# ==============================================================================
# CORRECTION FDR
# ==============================================================================

def appliquer_correction_fdr_par_domaine(df_combine, effets=('Mode', 'Trajet', 'Interaction')):
    """
    Applique la correction FDR par DOMAINE (EEG vs Comportement),
    en poolant les indicateurs des deux tâches dans chaque domaine.

    Ajoute les colonnes p_<effet>_FDR_domaine et Sig_<effet>_FDR_domaine.
    Retourne df_combine enrichi.
    """
    if 'Domaine' not in df_combine.columns:
        return df_combine

    for effet in effets:
        col_p = f'p_{effet}'
        col_p_fdr = f'p_{effet}_FDR_domaine'
        col_sig = f'Sig_{effet}_FDR_domaine'
        if col_p not in df_combine.columns:
            continue

        df_combine[col_p_fdr] = np.nan
        for domaine, grp in df_combine.groupby('Domaine'):
            p_values = grp[col_p].values
            mask_valid = ~np.isnan(p_values)
            if mask_valid.sum() == 0:
                continue
            _, p_corrected, _, _ = multipletests(
                p_values[mask_valid], alpha=ALPHA, method=METHODE_CORRECTION
            )
            p_fdr = np.full_like(p_values, np.nan)
            p_fdr[mask_valid] = p_corrected
            df_combine.loc[grp.index, col_p_fdr] = p_fdr

        df_combine[col_sig] = df_combine[col_p_fdr].apply(
            lambda x: '***' if (pd.notna(x) and x < 0.001)
            else '**' if (pd.notna(x) and x < 0.01)
            else '*' if (pd.notna(x) and x < 0.05)
            else 'ns' if pd.notna(x) else 'N/A'
        )

        for domaine in ['EEG', 'Comportement']:
            grp = df_combine[df_combine['Domaine'] == domaine]
            n_valid = int(grp[col_p].notna().sum())
            n_sig = int(grp[col_sig].isin(['*', '**', '***']).sum())
            print(f"    [FDR domaine] {effet} / {domaine}: {n_sig}/{n_valid} significatifs")

    return df_combine


def ecrire_resultats_par_domaine(df_combine, output_critere):
    """
    Éclate df_combine par Domaine et écrit deux fichiers Excel dans par_domaine/.
    Colonnes FDR par domaine doivent déjà être calculées.
    """
    dir_domaine = os.path.join(output_critere, 'par_domaine')
    Path(dir_domaine).mkdir(parents=True, exist_ok=True)

    for domaine, suffixe in [('EEG', 'EEG'), ('Comportement', 'Comportement')]:
        sub = df_combine[df_combine['Domaine'] == domaine].copy()
        if len(sub) == 0:
            continue
        fichier = os.path.join(dir_domaine, f'Resultats_{suffixe}.xlsx')
        sub.to_excel(fichier, index=False)
        print(f"  OK: par_domaine/{suffixe} → {fichier} (N={len(sub)})")


def appliquer_correction_fdr(df_resultats, label_niveau):
    """Applique la correction FDR sur un niveau d'analyse (une tâche)"""
    n_tests = len(df_resultats)
    print(f"\n  Correction FDR ({METHODE_CORRECTION}) - {label_niveau} ({n_tests} tests)...")

    for effet in ['Mode', 'Trajet', 'Interaction']:
        col_p = f'p_{effet}'
        col_p_fdr = f'p_{effet}_FDR'
        col_sig = f'Sig_{effet}_FDR'

        p_values = df_resultats[col_p].values
        mask_valid = ~np.isnan(p_values)

        p_fdr = np.full_like(p_values, np.nan)
        if mask_valid.sum() > 0:
            _, p_corrected, _, _ = multipletests(
                p_values[mask_valid], alpha=ALPHA, method=METHODE_CORRECTION
            )
            p_fdr[mask_valid] = p_corrected

        df_resultats[col_p_fdr] = p_fdr
        df_resultats[col_sig] = df_resultats[col_p_fdr].apply(
            lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns'
            if not np.isnan(x) else 'N/A'
        )

        n_sig = df_resultats[col_sig].isin(['*', '**', '***']).sum()
        print(f"    {effet}: {n_sig}/{mask_valid.sum()} significatifs après FDR")

    return df_resultats


# ==============================================================================
# RAPPORTS
# ==============================================================================

def generer_rapport_filtrage(critere, stats_filtrage, df_sel_initial, df_filtre, output_subdir):
    """Rapport détaillant le filtrage d'exposition pour le critère courant."""
    rapport_path = os.path.join(output_subdir, f'Rapport_Filtrage_{critere["nom"]}.txt')

    n_bike_i = int((df_sel_initial['Mode_transport'] == 'Bike').sum())
    n_ebike_i = int((df_sel_initial['Mode_transport'] == 'e-Bike').sum())
    n_bike_f = int((df_filtre['Mode_transport'] == 'Bike').sum())
    n_ebike_f = int((df_filtre['Mode_transport'] == 'e-Bike').sum())

    with open(rapport_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RAPPORT DE FILTRAGE - COHÉRENCE D'EXPOSITION\n")
        f.write(f"Critère : {critere['nom']}\n")
        f.write(f"Règle   : {critere['col_C']} >= {RATIO_MIN} × {critere['col_A']}\n")
        f.write(f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("EFFECTIFS\n")
        f.write("─" * 40 + "\n")
        f.write(f"  Sélection initiale (SELECTION_4) : {stats_filtrage['n_initial']}\n")
        f.write(f"    dont Vélo : {n_bike_i}, VAE : {n_ebike_i}\n")
        f.write(f"  Exclus (NaN exposition)          : {stats_filtrage['n_nan']}\n")
        f.write(f"  Exclus (rapport < {RATIO_MIN})        : {stats_filtrage['n_inverse_ou_insuffisant']}\n")
        f.write(f"  Sélection finale filtrée         : {stats_filtrage['n_final']}\n")
        f.write(f"    dont Vélo : {n_bike_f}, VAE : {n_ebike_f}\n")
        pct_perte = (1 - stats_filtrage['n_final'] / stats_filtrage['n_initial']) * 100
        f.write(f"  Perte : {pct_perte:.1f}%\n\n")

        f.write("PARTICIPANTS GARDÉS\n")
        f.write("─" * 40 + "\n")
        f.write(f"  {sorted(stats_filtrage['participants_gardes'])}\n\n")

        f.write("PARTICIPANTS EXCLUS (NaN exposition)\n")
        f.write("─" * 40 + "\n")
        f.write(f"  {sorted(stats_filtrage['participants_exclus_nan'])}\n\n")

        f.write("PARTICIPANTS EXCLUS (rapport insuffisant ou inversé)\n")
        f.write("─" * 40 + "\n")
        f.write(f"  {sorted(stats_filtrage['participants_exclus_inverse'])}\n\n")

        if len(stats_filtrage['participants_exclus_inverse']) > 0:
            f.write("RATIOS C/A DES PARTICIPANTS EXCLUS (rapport)\n")
            f.write("─" * 40 + "\n")
            exclus_ids = stats_filtrage['participants_exclus_inverse']
            df_exclus = df_sel_initial[df_sel_initial['Participant'].isin(exclus_ids)].copy()
            df_exclus = df_exclus.dropna(subset=[critere['col_A'], critere['col_C']])
            df_exclus['ratio'] = df_exclus[critere['col_C']] / df_exclus[critere['col_A']]
            for _, row in df_exclus.sort_values('ratio').iterrows():
                f.write(f"  Participant {int(row['Participant'])}  "
                        f"A={row[critere['col_A']]:.3f}  "
                        f"C={row[critere['col_C']]:.3f}  "
                        f"ratio C/A = {row['ratio']:.3f}\n")

    print(f"  OK: {rapport_path}")


def generer_rapport_niveau(df_resultats, label_niveau, output_subdir):
    """Génère le rapport pour une tâche (N-Back ou ANT)"""
    rapport_path = os.path.join(output_subdir, f'Rapport_{label_niveau}.txt')

    with open(rapport_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RAPPORT LMM - TÂCHE {label_niveau.upper()}\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Correction : FDR Benjamini-Hochberg sur {len(df_resultats)} tests\n")
        f.write("=" * 80 + "\n\n")

        for effet in ['Mode', 'Trajet', 'Interaction']:
            col_sig = f'Sig_{effet}_FDR'
            col_p = f'p_{effet}_FDR'
            col_p_brut = f'p_{effet}'
            col_d = f'd_{effet}'

            sig_mask = df_resultats[col_sig].isin(['*', '**', '***'])
            n_sig = sig_mask.sum()

            f.write(f"\n{'─' * 60}\n")
            f.write(f"EFFET {effet.upper()} : {n_sig} significatifs après FDR\n")
            f.write(f"{'─' * 60}\n\n")

            df_sorted = df_resultats.sort_values(col_p_brut)
            for _, row in df_sorted.iterrows():
                sig_flag = row[col_sig] if row[col_sig] in ['*', '**', '***'] else '  '
                f.write(f"  {sig_flag:>3} [{row['Domaine']:<12}] "
                        f"{row['Indicateur']:<45} "
                        f"p={row[col_p_brut]:.4f}  p_FDR={row[col_p]:.4f}  "
                        f"d={row[col_d]:.3f}  η²={row[f'eta2_{effet}']:.4f}\n")

        # ── Post-hoc ──
        f.write(f"\n\n{'=' * 80}\n")
        f.write("POST-HOC (Interaction significative après FDR)\n")
        f.write(f"{'=' * 80}\n\n")

        inter_sig = df_resultats[
            df_resultats['Sig_Interaction_FDR'].isin(['*', '**', '***'])
        ]

        if len(inter_sig) == 0:
            f.write("  Aucune interaction significative après FDR.\n\n")

            tendances = df_resultats[
                (df_resultats['p_Interaction_FDR'] < 0.10) &
                (~df_resultats['Sig_Interaction_FDR'].isin(['*', '**', '***']))
            ]
            if len(tendances) > 0:
                f.write("  Tendances (p_FDR < 0.10) :\n")
                for _, row in tendances.iterrows():
                    f.write(f"    {row['Indicateur']}: p_FDR={row['p_Interaction_FDR']:.4f}, "
                            f"d={row['d_Interaction']:.3f}\n")
                    bike_sig = row.get('PostHoc_Bike_sig', False)
                    ebike_sig = row.get('PostHoc_eBike_sig', False)
                    bike_p = row.get('PostHoc_Bike_p_corr', np.nan)
                    ebike_p = row.get('PostHoc_eBike_p_corr', np.nan)
                    if not np.isnan(bike_p):
                        f.write(f"      Vélo: p_corr={bike_p:.4f}, d={row['PostHoc_Bike_d']:.3f} "
                                f"{'* ' + str(row['PostHoc_Bike_direction']) if bike_sig else 'ns'}\n")
                    if not np.isnan(ebike_p):
                        f.write(f"      VAE:  p_corr={ebike_p:.4f}, d={row['PostHoc_eBike_d']:.3f} "
                                f"{'* ' + str(row['PostHoc_eBike_direction']) if ebike_sig else 'ns'}\n")
        else:
            for _, row in inter_sig.iterrows():
                f.write(f"  {row['Indicateur']}:\n")
                f.write(f"    Interaction: p_FDR={row['p_Interaction_FDR']:.4f}, "
                        f"d={row['d_Interaction']:.3f}\n")
                bike_sig = row.get('PostHoc_Bike_sig', False)
                ebike_sig = row.get('PostHoc_eBike_sig', False)
                f.write(f"    Vélo: p_corr={row['PostHoc_Bike_p_corr']:.4f}, "
                        f"d={row['PostHoc_Bike_d']:.3f} "
                        f"{'* ' + row['PostHoc_Bike_direction'] if bike_sig else 'ns'}\n")
                f.write(f"    VAE:  p_corr={row['PostHoc_eBike_p_corr']:.4f}, "
                        f"d={row['PostHoc_eBike_d']:.3f} "
                        f"{'* ' + row['PostHoc_eBike_direction'] if ebike_sig else 'ns'}\n\n")

        # ── Diagnostics ──
        f.write(f"\n{'=' * 80}\n")
        f.write("DIAGNOSTICS\n")
        f.write(f"{'=' * 80}\n\n")

        for _, row in df_resultats.iterrows():
            problemes = []
            if row.get('Normalite_residus') == 'NON_NORMAL':
                problemes.append('non-normalité résidus')
            if row.get('N_outliers', 0) > 2:
                problemes.append(f"{row['N_outliers']} outliers")
            if row.get('Pct_perte_donnees', 0) > 10:
                problemes.append(f"{row['Pct_perte_donnees']:.0f}% perte")
            if problemes:
                f.write(f"  {row['Indicateur']:<45} ⚠ {', '.join(problemes)}\n")

        n_non_normal = (df_resultats['Normalite_residus'] == 'NON_NORMAL').sum()
        f.write(f"\n  Résidus non-normaux : {n_non_normal}/{len(df_resultats)}\n")
        f.write(f"  Modèles : {df_resultats['Type_modele'].value_counts().to_dict()}\n")

    print(f"  OK: {rapport_path}")


def generer_rapport_global(df_nback, df_ant, output_dir, critere, stats_filtrage):
    """Rapport de synthèse combinant les deux tâches pour UN critère."""
    rapport_path = os.path.join(output_dir, 'Rapport_Synthese_Globale.txt')

    with open(rapport_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"SYNTHÈSE GLOBALE - ANALYSE CORRECTIVE\n")
        f.write(f"Critère exposition : {critere['nom']} "
                f"({critere['col_C']} >= {RATIO_MIN} × {critere['col_A']})\n")
        f.write(f"N filtré : {stats_filtrage['n_final']} / {stats_filtrage['n_initial']}\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("STRATÉGIE ANALYTIQUE\n")
        f.write("─" * 40 + "\n")
        if df_nback is not None:
            f.write(f"  N-Back (1B + 2B) : {len(df_nback)} indicateurs\n")
        if df_ant is not None:
            f.write(f"  ANT              : {len(df_ant)} indicateurs\n")
        f.write(f"  Correction : FDR Benjamini-Hochberg appliquée séparément par tâche\n\n")

        for label, df_res in [('N-BACK (1B + 2B)', df_nback),
                               ('ANT', df_ant)]:
            if df_res is None:
                continue
            f.write(f"\n{'═' * 80}\n")
            f.write(f"{label}\n")
            f.write(f"{'═' * 80}\n\n")

            f.write(f"  Indicateurs EEG         : "
                    f"{int((df_res['Domaine'] == 'EEG').sum())}\n")
            f.write(f"  Indicateurs Comportement: "
                    f"{int((df_res['Domaine'] == 'Comportement').sum())}\n\n")

            for effet in ['Mode', 'Trajet', 'Interaction']:
                col_sig = f'Sig_{effet}_FDR'
                col_p_fdr = f'p_{effet}_FDR'
                col_d = f'd_{effet}'

                n_sig = df_res[col_sig].isin(['*', '**', '***']).sum()
                f.write(f"  {effet}: {n_sig} significatifs\n")

                sig = df_res[df_res[col_sig].isin(['*', '**', '***'])].sort_values(col_p_fdr)
                for _, row in sig.iterrows():
                    f.write(f"    ✓ [{row['Domaine']:<12}] {row['Indicateur']:<40} "
                            f"p_FDR={row[col_p_fdr]:.4f}  d={row[col_d]:.3f}\n")

                tend = df_res[
                    (df_res[col_p_fdr] < 0.10) &
                    (~df_res[col_sig].isin(['*', '**', '***']))
                ].sort_values(col_p_fdr)
                for _, row in tend.iterrows():
                    f.write(f"    ~ [{row['Domaine']:<12}] {row['Indicateur']:<40} "
                            f"p_FDR={row[col_p_fdr]:.4f}  d={row[col_d]:.3f}  (tendance)\n")

            f.write("\n")

        # ── Résumé ──
        f.write(f"\n{'═' * 80}\n")
        f.write("RÉSUMÉ\n")
        f.write(f"{'═' * 80}\n\n")

        n_sig_nback = 0
        if df_nback is not None:
            n_sig_nback = sum(
                df_nback[f'Sig_{e}_FDR'].isin(['*', '**', '***']).sum()
                for e in ['Mode', 'Trajet', 'Interaction']
            )
        n_sig_ant = 0
        if df_ant is not None:
            n_sig_ant = sum(
                df_ant[f'Sig_{e}_FDR'].isin(['*', '**', '***']).sum()
                for e in ['Mode', 'Trajet', 'Interaction']
            )

        f.write(f"  Effets significatifs - N-Back : {n_sig_nback}\n")
        f.write(f"  Effets significatifs - ANT    : {n_sig_ant}\n")
        f.write(f"  Total : {n_sig_nback + n_sig_ant}\n")

    print(f"  OK: {rapport_path}")


# ==============================================================================
# PIPELINE PAR CRITÈRE
# ==============================================================================

def analyser_niveau(df_sel, indicateurs, label_niveau, output_subdir):
    """Analyse une tâche complète (indicateurs → LMM → FDR → rapport)"""

    print(f"\n{'═' * 70}")
    print(f"  TÂCHE {label_niveau.upper()} ({len(indicateurs)} indicateurs)")
    print(f"{'═' * 70}\n")

    resultats_list = []
    for config in indicateurs:
        res = analyser_indicateur(df_sel, config)
        if res is not None:
            resultats_list.append(res)

    if len(resultats_list) == 0:
        print(f"  ERREUR: Aucun résultat pour {label_niveau}")
        return None

    df_resultats = pd.DataFrame(resultats_list)
    df_resultats = appliquer_correction_fdr(df_resultats, label_niveau)

    colonnes_export = [
        'Indicateur', 'Tache', 'Domaine', 'Transformation',
        'N_obs', 'N_participants', 'N_NaN_away', 'N_NaN_close', 'Pct_perte_donnees',
        'F_Mode', 'p_Mode', 'p_Mode_FDR', 'Sig_Mode_FDR', 'eta2_Mode', 'd_Mode',
        'F_Trajet', 'p_Trajet', 'p_Trajet_FDR', 'Sig_Trajet_FDR', 'eta2_Trajet', 'd_Trajet', 'dz_Trajet',
        'F_Interaction', 'p_Interaction', 'p_Interaction_FDR', 'Sig_Interaction_FDR',
        'eta2_Interaction', 'd_Interaction',
        'Convergence', 'Type_modele',
        'Normalite_residus', 'p_Shapiro_Wilk', 'p_DAgostino_Pearson',
        'Homoscedasticite_p', 'N_outliers',
        'PostHoc_Bike_p', 'PostHoc_Bike_p_corr', 'PostHoc_Bike_d', 'PostHoc_Bike_dz',
        'PostHoc_Bike_sig', 'PostHoc_Bike_direction',
        'PostHoc_eBike_p', 'PostHoc_eBike_p_corr', 'PostHoc_eBike_d', 'PostHoc_eBike_dz',
        'PostHoc_eBike_sig', 'PostHoc_eBike_direction',
    ]
    colonnes_export = [c for c in colonnes_export if c in df_resultats.columns]

    fichier = os.path.join(output_subdir, f'Resultats_{label_niveau}.xlsx')
    df_resultats[colonnes_export].to_excel(fichier, index=False)
    print(f"\n  OK: {fichier}")

    generer_rapport_niveau(df_resultats, label_niveau, output_subdir)
    return df_resultats


def analyser_critere(df_sel_initial, critere, indic_nback, indic_ant, racine_output):
    """Pipeline complet pour un critère d'exposition donné."""
    print(f"\n\n{'#' * 80}")
    print(f"#  CRITÈRE : {critere['nom']}  "
          f"({critere['col_C']} >= {RATIO_MIN} × {critere['col_A']})")
    print(f"{'#' * 80}")

    output_critere = os.path.join(racine_output, critere['nom'])
    creer_dossiers(output_critere)

    # Filtrage cohérence exposition
    df_filtre, stats_filtrage = filtrer_coherence_exposition(
        df_sel_initial, critere['col_A'], critere['col_C'], RATIO_MIN
    )

    n_bike = int((df_filtre['Mode_transport'] == 'Bike').sum())
    n_ebike = int((df_filtre['Mode_transport'] == 'e-Bike').sum())
    print(f"\nFiltrage exposition :")
    print(f"  Initial          : {stats_filtrage['n_initial']}")
    print(f"  Exclus (NaN)     : {stats_filtrage['n_nan']}")
    print(f"  Exclus (ratio<{RATIO_MIN}): {stats_filtrage['n_inverse_ou_insuffisant']}")
    print(f"  Final            : {stats_filtrage['n_final']} (Vélo: {n_bike}, VAE: {n_ebike})")

    generer_rapport_filtrage(critere, stats_filtrage, df_sel_initial, df_filtre, output_critere)

    if stats_filtrage['n_final'] < 10:
        print(f"  ⚠ Effectif insuffisant (<10) : sous-analyse ignorée.")
        return {
            'critere': critere['nom'],
            'n_final': stats_filtrage['n_final'],
            'df_nback': None,
            'df_ant': None,
        }

    # LMM N-Back (1B + 2B)
    dir_nback = os.path.join(output_critere, 'NBack')
    df_nback = analyser_niveau(df_filtre, indic_nback, 'NBack', dir_nback)

    # LMM ANT
    dir_ant = os.path.join(output_critere, 'ANT')
    df_ant = analyser_niveau(df_filtre, indic_ant, 'ANT', dir_ant)

    # Synthèse critère
    if df_nback is not None or df_ant is not None:
        generer_rapport_global(df_nback, df_ant, output_critere, critere, stats_filtrage)

    # Résultats combinés critère + FDR par domaine (EEG / Comportement)
    if df_nback is not None and df_ant is not None:
        df_nback_tag = df_nback.copy()
        df_nback_tag['Niveau_analyse'] = 'NBack'
        df_ant_tag = df_ant.copy()
        df_ant_tag['Niveau_analyse'] = 'ANT'

        df_combine = pd.concat([df_nback_tag, df_ant_tag], ignore_index=True)

        print(f"\n  Correction FDR par DOMAINE (EEG + Comportement poolés NBack+ANT) ...")
        df_combine = appliquer_correction_fdr_par_domaine(df_combine)

        fichier_combine = os.path.join(output_critere, 'Resultats_Combines.xlsx')
        df_combine.to_excel(fichier_combine, index=False)
        print(f"  OK: Résultats combinés → {fichier_combine}")

        ecrire_resultats_par_domaine(df_combine, output_critere)

    return {
        'critere': critere['nom'],
        'n_final': stats_filtrage['n_final'],
        'n_nan': stats_filtrage['n_nan'],
        'n_inverse': stats_filtrage['n_inverse_ou_insuffisant'],
        'df_nback': df_nback,
        'df_ant': df_ant,
    }


# ==============================================================================
# SYNTHÈSE CROSS-CRITÈRES
# ==============================================================================

def generer_synthese_4_criteres(resultats_par_critere, output_dir):
    """Fichier Excel récapitulatif des 4 sous-analyses."""
    fichier = os.path.join(output_dir, 'Synthese_4_criteres.xlsx')

    lignes_overview = []
    for res in resultats_par_critere:
        ligne = {
            'Critere': res['critere'],
            'N_filtre': res['n_final'],
            'N_exclus_NaN': res.get('n_nan', np.nan),
            'N_exclus_inverse': res.get('n_inverse', np.nan),
            'N_sig_NBack_Mode': np.nan,
            'N_sig_NBack_Trajet': np.nan,
            'N_sig_NBack_Interaction': np.nan,
            'N_sig_ANT_Mode': np.nan,
            'N_sig_ANT_Trajet': np.nan,
            'N_sig_ANT_Interaction': np.nan,
        }
        if res['df_nback'] is not None:
            for e in ['Mode', 'Trajet', 'Interaction']:
                ligne[f'N_sig_NBack_{e}'] = int(
                    res['df_nback'][f'Sig_{e}_FDR'].isin(['*', '**', '***']).sum()
                )
        if res['df_ant'] is not None:
            for e in ['Mode', 'Trajet', 'Interaction']:
                ligne[f'N_sig_ANT_{e}'] = int(
                    res['df_ant'][f'Sig_{e}_FDR'].isin(['*', '**', '***']).sum()
                )
        lignes_overview.append(ligne)
    df_overview = pd.DataFrame(lignes_overview)

    lignes_sig = []
    for res in resultats_par_critere:
        for tache_label, df_res in [('NBack', res['df_nback']),
                                      ('ANT', res['df_ant'])]:
            if df_res is None:
                continue
            for effet in ['Mode', 'Trajet', 'Interaction']:
                col_sig = f'Sig_{effet}_FDR'
                col_p_fdr = f'p_{effet}_FDR'
                col_d = f'd_{effet}'
                sig = df_res[df_res[col_sig].isin(['*', '**', '***'])]
                for _, row in sig.iterrows():
                    lignes_sig.append({
                        'Critere': res['critere'],
                        'Tache': tache_label,
                        'Domaine': row['Domaine'],
                        'Effet': effet,
                        'Indicateur': row['Indicateur'],
                        'Sous_tache': row['Tache'],
                        'p_FDR': row[col_p_fdr],
                        'Sig': row[col_sig],
                        'd': row[col_d],
                        'N_participants': row['N_participants'],
                    })
    df_sig = pd.DataFrame(lignes_sig)

    with pd.ExcelWriter(fichier) as writer:
        df_overview.to_excel(writer, sheet_name='Overview', index=False)
        if len(df_sig) > 0:
            df_sig.to_excel(writer, sheet_name='Significatifs', index=False)
        else:
            pd.DataFrame({'message': ['Aucun effet significatif dans les 4 sous-analyses']}) \
                .to_excel(writer, sheet_name='Significatifs', index=False)

    print(f"\n  OK: Synthèse 4 critères → {fichier}")

    rapport_path = os.path.join(output_dir, 'Rapport_Synthese_4_criteres.txt')
    with open(rapport_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SYNTHÈSE DES 4 ANALYSES CORRECTIVES\n")
        f.write(f"Ratio minimal : C >= {RATIO_MIN} × A\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")

        for _, row in df_overview.iterrows():
            f.write(f"\n{'─' * 70}\n")
            f.write(f"CRITÈRE : {row['Critere']}  (N filtré = {row['N_filtre']})\n")
            f.write(f"{'─' * 70}\n")
            f.write(f"  Exclus NaN       : {row['N_exclus_NaN']}\n")
            f.write(f"  Exclus inverse   : {row['N_exclus_inverse']}\n")
            f.write(f"  N-Back → Mode:{row['N_sig_NBack_Mode']}  "
                    f"Trajet:{row['N_sig_NBack_Trajet']}  "
                    f"Interaction:{row['N_sig_NBack_Interaction']}\n")
            f.write(f"  ANT    → Mode:{row['N_sig_ANT_Mode']}  "
                    f"Trajet:{row['N_sig_ANT_Trajet']}  "
                    f"Interaction:{row['N_sig_ANT_Interaction']}\n")

        if len(df_sig) > 0:
            f.write(f"\n\n{'=' * 80}\n")
            f.write("DÉTAIL DES INDICATEURS SIGNIFICATIFS\n")
            f.write(f"{'=' * 80}\n\n")
            for critere_nom in df_sig['Critere'].unique():
                f.write(f"\n── {critere_nom} ──\n")
                for _, row in df_sig[df_sig['Critere'] == critere_nom].iterrows():
                    f.write(f"  {row['Sig']:>3} [{row['Tache']:<5}|{row['Domaine']:<12}] "
                            f"{row['Effet']:<12} "
                            f"{row['Indicateur']:<40} "
                            f"p_FDR={row['p_FDR']:.4f}  d={row['d']:.3f}  "
                            f"N={row['N_participants']}\n")

    print(f"  OK: Rapport synthèse → {rapport_path}")


# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def main():
    print("\n" + "=" * 80)
    print("ANALYSE LMM CORRECTIVE - FDR PAR TÂCHE (N-Back / ANT)")
    print(f"Filtrage cohérence exposition : C >= {RATIO_MIN} × A")
    print("=" * 80)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\nDossier racine: {OUTPUT_DIR}")

    indic_nback = generer_indicateurs_nback()
    indic_ant = generer_indicateurs_ant()

    n_nb_eeg = sum(1 for i in indic_nback if i['domaine'] == 'EEG')
    n_nb_comp = sum(1 for i in indic_nback if i['domaine'] == 'Comportement')
    n_ant_eeg = sum(1 for i in indic_ant if i['domaine'] == 'EEG')
    n_ant_comp = sum(1 for i in indic_ant if i['domaine'] == 'Comportement')

    print(f"\nStructure de chaque sous-analyse :")
    print(f"  N-Back (1B + 2B) : {len(indic_nback)} indicateurs → FDR sur {len(indic_nback)} tests")
    print(f"    (EEG: {n_nb_eeg}, Comportement: {n_nb_comp})")
    print(f"  ANT              : {len(indic_ant)} indicateurs → FDR sur {len(indic_ant)} tests")
    print(f"    (EEG: {n_ant_eeg}, Comportement: {n_ant_comp})")

    print(f"\nCritères d'exposition : {len(CRITERES_EXPOSITION)} sous-analyses")
    for c in CRITERES_EXPOSITION:
        print(f"  - {c['nom']}")

    df = charger_donnees(DATA_FILE)
    df_sel = filtrer_selection(df, SELECTION_4)
    n_bike = int((df_sel['Mode_transport'] == 'Bike').sum())
    n_ebike = int((df_sel['Mode_transport'] == 'e-Bike').sum())
    print(f"\nSélection 4 initiale: {len(df_sel)} participants (Vélo: {n_bike}, VAE: {n_ebike})")

    resultats_par_critere = []
    for critere in CRITERES_EXPOSITION:
        res = analyser_critere(df_sel, critere, indic_nback, indic_ant, OUTPUT_DIR)
        resultats_par_critere.append(res)

    print(f"\n\n{'=' * 80}")
    print("SYNTHÈSE FINALE DES 4 CRITÈRES")
    print(f"{'=' * 80}")
    generer_synthese_4_criteres(resultats_par_critere, OUTPUT_DIR)

    print(f"\n\n{'=' * 80}")
    print("ANALYSE TERMINÉE")
    print(f"{'=' * 80}")
    for res in resultats_par_critere:
        n_sig_nback = 0
        n_sig_ant = 0
        if res['df_nback'] is not None:
            n_sig_nback = sum(
                res['df_nback'][f'Sig_{e}_FDR'].isin(['*', '**', '***']).sum()
                for e in ['Mode', 'Trajet', 'Interaction']
            )
        if res['df_ant'] is not None:
            n_sig_ant = sum(
                res['df_ant'][f'Sig_{e}_FDR'].isin(['*', '**', '***']).sum()
                for e in ['Mode', 'Trajet', 'Interaction']
            )
        print(f"  {res['critere']:<15} N={res['n_final']:<3}  "
              f"N-Back sig={n_sig_nback}  ANT sig={n_sig_ant}")

    print(f"\n  Sorties dans : {OUTPUT_DIR}")
    print(f"    <critere>/NBack/ Resultats + Rapport")
    print(f"    <critere>/ANT/   Resultats + Rapport")
    print(f"    <critere>/Rapport_Filtrage_<critere>.txt")
    print(f"    <critere>/Rapport_Synthese_Globale.txt")
    print(f"    Synthese_4_criteres.xlsx")
    print(f"    Rapport_Synthese_4_criteres.txt")


if __name__ == "__main__":
    main()
