# === CODAGE SOMME DES FACTEURS — corrige le 2026-09-09 ===
# Le modele Mode x Trajet etait ajuste en codage treatment (reference 'Bike'), de sorte que
# le terme lu comme 'effet principal du Trajet' etait l'effet du trajet CHEZ LES CYCLISTES
# CLASSIQUES — un effet simple. Le codage somme C(..., Sum) rend les termes marginaux.
# Les post-hocs par mode et les tailles d'effet (calculees sur les donnees) sont inchanges.
# ATTENTION : C(Trajet, Sum)[S.Away] a le signe OPPOSE a Trajet[T.Close].
# Sauvegarde de la version anterieure dans Data\Codes\_Archives\.
"""
════════════════════════════════════════════════════════════════════════════════
ANALYSE DE SENSIBILITÉ - INDICATEURS CLÉS

Objectif : Vérifier la robustesse des effets principaux face aux
violations diagnostiques (non-normalité, outliers, données manquantes).

Indicateurs testés :
  - N1_Moy_Valide_1B (effet Trajet significatif, non-normalité + 3 outliers)
  - P3_Moy_Valide_2B (tendance Interaction, 14% perte données)

Analyses de sensibilité :
  1. Modèle original (référence)
  2. Exclusion des outliers (|z| > 3 sur les résidus)
  3. Exclusion des outliers (|z| > 2.5, plus strict)
  4. Winsorisation des outliers (remplacement par ±3 SD)
  5. Test non-paramétrique de confirmation (permutations)

Sélection : Selection_4 (N=51)
════════════════════════════════════════════════════════════════════════════════
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402

import sys
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

# Console UTF-8 (caractères de cadre / accents dans les print)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# ==============================================================================
# TERME DE SESSION (repris a l'identique de LMM_intention.py)
# Le plan est un croise a deux periodes : la session de test est un facteur
# intra-participant, determine par l'ordre d'attribution des itineraires et par
# l'itineraire effectivement realise (chapitre 3, sections 3.3.4 et 3.9.2).
# ==============================================================================

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


DATA_FILE = D(r'Data\Table_combine\Tab_sujet_20260205.xlsx')
OUTPUT_DIR = os.path.join(
    S(r'Resultats_LMM_intention'),
    datetime.now().strftime('%y-%m-%d') + '_sensibilite_session')

VERSION_DONNEES = 'filtre'

SELECTION_4 = [101, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
               201, 202, 203, 204, 205, 206, 209, 210, 211, 212, 215, 216,
               121, 122, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 139,
               221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]

ALPHA = 0.05
N_PERMUTATIONS = 5000

# Indicateurs à tester
INDICATEURS_SENSIBILITE = [
    {
        'nom': 'N1_Moy_Valide_1B',
        'col_away': 'N1_Moy_Valide_A_1B',
        'col_close': 'N1_Moy_Valide_C_1B',
        'transformation': None,
        'effet_cible': 'Trajet',
        'diagnostic': 'non-normalité résidus, 3 outliers'
    },
    {
        # Ajoute le 2026-09-09 : apres la correction du codage des facteurs, le temps de
        # reaction en 2-back est le resultat comportemental significatif (q = 0,015 au
        # niveau 1). Le tableau 4.5 du Ch. 4 doit donc porter sa sensibilite, au meme
        # titre que celle de la N1. Colonnes et transformation identiques au niveau 1.
        'nom': 'RT_moyen_2B',
        'col_away': 'RT_moyen_filtre_A_2B',
        'col_close': 'RT_moyen_filtre_C_2B',
        'transformation': 'log',
        'effet_cible': 'Trajet',
        'diagnostic': 'effet principal comportemental, significatif apres correction du codage'
    },
    {
        'nom': 'P3_Moy_Valide_2B',
        'col_away': 'P3_Moy_Valide_A_2B',
        'col_close': 'P3_Moy_Valide_C_2B',
        'transformation': None,
        'effet_cible': 'Interaction',
        'diagnostic': '14% perte données'
    },
]


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def charger_et_filtrer(fichier, selection):
    """Charge et filtre les données"""
    df = pd.read_excel(fichier)
    _ordre = _charger_ordre()
    if _ordre is not None and 'Participant' in df.columns:
        df = df.merge(_ordre, on='Participant', how='left')
    return df[df['Participant'].isin(selection)].copy()


def appliquer_transformation(valeurs, type_transformation):
    """Reprise a l'identique de LMM_intention.py.

    Ajout du 2026-09-09 : ce script declarait la cle 'transformation' dans sa
    configuration mais ne l'appliquait jamais. Le modele de sensibilite etait donc
    ajuste sur la variable BRUTE la ou le niveau 1 l'ajuste sur le log, pour les
    seuls temps de reaction. La consequence visible etait un dz different entre le
    tableau 4.5 et le texte du §4.2.2 (0,440 contre 0,435) ; la consequence de fond
    est que les residus standardises, donc la detection des valeurs atypiques et la
    winsorisation, portaient sur une autre echelle que celle du modele.
    """
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
    """Format long avec paires complètes"""
    mask = df[col_away].isna() | df[col_close].isna()
    df_clean = df[~mask].copy()

    _cols = [c for c in ('Participant', 'Mode_transport', 'Ordre')
             if c in df_clean.columns]

    df_away = df_clean[_cols].copy()
    df_away['Trajet_reel'] = 'Away'
    df_away['Valeur'] = appliquer_transformation(df_clean[col_away], transformation).values
    df_away['Trajet'] = 'Away'

    df_close = df_clean[_cols].copy()
    df_close['Trajet_reel'] = 'Close'
    df_close['Valeur'] = appliquer_transformation(df_clean[col_close], transformation).values
    df_close['Trajet'] = 'Close'

    df_long = pd.concat([df_away, df_close], ignore_index=True)
    return _ajouter_session(df_long)


def ajuster_lmm(df_long):
    """Ajuste le LMM avec fallback"""
    # Pente aléatoire
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = mixedlm(_formule(df_long, "Valeur ~ C(Mode_transport, Sum) * C(Trajet, Sum)"),
                            df_long,
                            groups=df_long["Participant"], re_formula="~Trajet")
            result = model.fit(method='lbfgs', maxiter=200)
            if result.converged:
                return result, 'LMM_pente'
    except:
        pass

    # Intercept seul
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = mixedlm(_formule(df_long, "Valeur ~ C(Mode_transport, Sum) * C(Trajet, Sum)"),
                            df_long,
                            groups=df_long["Participant"])
            result = model.fit(method='lbfgs', maxiter=200)
            if result.converged:
                return result, 'LMM_intercept'
    except:
        pass

    return None, 'Echec'


# ==============================================================================
# TAILLES D'EFFET (référencées à la variance TOTALE, Lakens 2013)
# Portées à l'identique de LMM_intention.py (script de référence patché).
# ==============================================================================

def cohen_d_av(df_long, facteur, niv_ref, niv_cible):
    """Cohen's d_av intra-sujet (Lakens, 2013) : différence de moyennes brutes
    rapportée à la MOYENNE des SD des deux conditions (référence = variance
    totale, comparable aux seuils 0.2/0.5/0.8). Signe : cible - reference
    (cohérent avec le coefficient [T.cible] du LMM)."""
    v_ref = df_long.loc[df_long[facteur] == niv_ref, 'Valeur']
    v_cible = df_long.loc[df_long[facteur] == niv_cible, 'Valeur']
    if len(v_ref) < 2 or len(v_cible) < 2:
        return np.nan
    sd_moy = (v_ref.std(ddof=1) + v_cible.std(ddof=1)) / 2
    return (v_cible.mean() - v_ref.mean()) / sd_moy if sd_moy > 0 else np.nan


def cohen_d_s(df_long, facteur, niv_ref, niv_cible):
    """Cohen's d_s inter-sujet : agrégation par participant (moyenne sur les
    conditions intra) puis SD poolée des deux groupes. Évite la pseudo-
    réplication. Signe : cible - reference."""
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
    """Cohen's dz intra-sujet (Lakens, 2013) : moyenne des différences
    appariées rapportée à la SD des différences (référence INTRA, pour la
    puissance/réplication intra-sujet ; dépend de la corrélation entre
    conditions, donc NON comparable entre études). Signe : cible - reference."""
    piv = df_long.pivot_table(index='Participant', columns=facteur, values='Valeur')
    if niv_ref not in piv.columns or niv_cible not in piv.columns:
        return np.nan
    diff = (piv[niv_cible] - piv[niv_ref]).dropna()
    if len(diff) < 2:
        return np.nan
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else np.nan


def extraire_resultats(model_result, df_long):
    """Extrait p/t/eta2 + tailles d'effet référencées à la variance TOTALE.

    - Trajet (intra-sujet) -> Cohen's d_av (variance totale, comparable) ET
      dz (référence intra, Lakens 2013) ; on reporte les DEUX.
    - Mode   (inter-sujet) -> Cohen's d_s poolé (dz non défini en inter).
    - Interaction          -> pas de d interprétable (porté par eta2 + post-hocs).
    eta2 = eta2 PARTIEL = t²/(t²+df_resid).
    """
    clefs = {
        'Mode': 'C(Mode_transport, Sum)[S.Bike]',
        'Trajet': 'C(Trajet, Sum)[S.Away]',
        'Interaction': 'C(Mode_transport, Sum)[S.Bike]:C(Trajet, Sum)[S.Away]'
    }

    res = {}
    for effet_nom, coef_key in clefs.items():
        if coef_key in model_result.pvalues:
            p_val = model_result.pvalues[coef_key]
            t_val = model_result.tvalues[coef_key]
            eta2 = (t_val ** 2) / (t_val ** 2 + model_result.df_resid)

            if effet_nom == 'Trajet':
                cohens_d = cohen_d_av(df_long, 'Trajet', 'Away', 'Close')
                dz = cohen_dz(df_long, 'Trajet', 'Away', 'Close')
            elif effet_nom == 'Mode':
                cohens_d = cohen_d_s(df_long, 'Mode_transport', 'Bike', 'e-Bike')
                dz = np.nan  # facteur inter-sujet : dz non défini
            else:  # Interaction : pas de d interprétable
                cohens_d, dz = np.nan, np.nan

            res[effet_nom] = {
                'p': p_val, 't': t_val, 'cohens_d': cohens_d, 'dz': dz,
                'eta2': eta2, 'significatif': p_val < ALPHA
            }

    return res


def post_hoc_par_mode(df_long):
    """Post-hoc : effet Trajet par Mode"""
    resultats = {}
    for mode in ['Bike', 'e-Bike']:
        df_mode = df_long[df_long['Mode_transport'] == mode].copy()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model = mixedlm(_formule(df_mode, "Valeur ~ Trajet"), df_mode,
                                groups=df_mode["Participant"])
                fit = model.fit(method='lbfgs', maxiter=200)

            if fit.converged and 'Trajet[T.Close]' in fit.pvalues:
                p = fit.pvalues['Trajet[T.Close]']
                coef = fit.params['Trajet[T.Close]']
                resultats[mode] = {
                    'p': p, 'p_corr': min(p * 2, 1.0),
                    'cohens_d': cohen_d_av(df_mode, 'Trajet', 'Away', 'Close'),
                    'cohens_dz': cohen_dz(df_mode, 'Trajet', 'Away', 'Close'),
                    'direction': 'Close > Away' if coef > 0 else 'Away > Close'
                }
        except:
            resultats[mode] = {'erreur': 'échec'}

    return resultats


# ==============================================================================
# ANALYSES DE SENSIBILITÉ
# ==============================================================================

def sensibilite_original(df_long, effet_cible):
    """Analyse 1 : Modèle original (référence)"""
    model, type_m = ajuster_lmm(df_long)
    if model is None:
        return None

    res = extraire_resultats(model, df_long)
    ph = post_hoc_par_mode(df_long)

    # Diagnostics
    residus = model.resid
    _, p_shapiro = stats.shapiro(residus)
    z = np.abs((residus - residus.mean()) / residus.std())
    n_outliers = int((z > 3).sum())

    return {
        'label': 'Original',
        'n_obs': len(df_long),
        'n_participants': df_long['Participant'].nunique(),
        'type_modele': type_m,
        'resultats': res,
        'post_hoc': ph,
        'p_shapiro': p_shapiro,
        'n_outliers': n_outliers,
        'outliers_ids': df_long.index[z > 3].tolist() if n_outliers > 0 else [],
        'participants_outliers': df_long.loc[z > 3, 'Participant'].tolist() if n_outliers > 0 else [],
    }


def sensibilite_sans_outliers(df_long, effet_cible, seuil_z=3.0):
    """Analyse 2/3 : Exclusion des outliers"""
    # D'abord ajuster le modèle original pour identifier les outliers
    model_orig, _ = ajuster_lmm(df_long)
    if model_orig is None:
        return None

    residus = model_orig.resid
    z = np.abs((residus - residus.mean()) / residus.std())
    mask_keep = z <= seuil_z

    n_exclus = (~mask_keep).sum()
    participants_exclus = df_long.loc[~mask_keep, 'Participant'].unique().tolist()

    df_clean = df_long[mask_keep].copy()

    # Ré-ajuster
    model, type_m = ajuster_lmm(df_clean)
    if model is None:
        return None

    res = extraire_resultats(model, df_clean)
    ph = post_hoc_par_mode(df_clean)

    residus_new = model.resid
    _, p_shapiro = stats.shapiro(residus_new)
    z_new = np.abs((residus_new - residus_new.mean()) / residus_new.std())

    return {
        'label': f'Sans outliers (|z|>{seuil_z})',
        'n_obs': len(df_clean),
        'n_participants': df_clean['Participant'].nunique(),
        'n_exclus': n_exclus,
        'participants_exclus': participants_exclus,
        'type_modele': type_m,
        'resultats': res,
        'post_hoc': ph,
        'p_shapiro': p_shapiro,
        'n_outliers': int((z_new > 3).sum()),
    }


def sensibilite_winsorisee(df_long, effet_cible, seuil_z=3.0):
    """Analyse 4 : Winsorisation (remplacement par ±seuil×SD)"""
    model_orig, _ = ajuster_lmm(df_long)
    if model_orig is None:
        return None

    residus = model_orig.resid
    mean_r = residus.mean()
    sd_r = residus.std()

    # Le chapitre 3 (section 3.9.2) definit les valeurs extremes sur les RESIDUS
    # standardises, comme le fait sensibilite_sans_outliers. La winsorisation est
    # donc conduite sur la meme echelle : la valeur observee est ramenee de facon
    # que son residu ne depasse plus le seuil. Le decompte sur les valeurs brutes
    # est conserve a titre de comparaison : il vaut 0 ici, ce qui rendait la
    # winsorisation sans effet dans la version anterieure du script.
    z_valeurs = np.abs((df_long['Valeur'] - df_long['Valeur'].mean()) / df_long['Valeur'].std())
    n_winsorise_valeurs = int((z_valeurs > seuil_z).sum())

    r = np.asarray(residus, dtype=float)
    r_clip = np.clip(r, mean_r - seuil_z * sd_r, mean_r + seuil_z * sd_r)
    delta = r - r_clip
    n_winsorise = int((delta != 0).sum())

    df_wins = df_long.copy()
    df_wins['Valeur'] = np.asarray(df_wins['Valeur'], dtype=float) - delta

    model, type_m = ajuster_lmm(df_wins)
    if model is None:
        return None

    res = extraire_resultats(model, df_wins)
    ph = post_hoc_par_mode(df_wins)

    residus_new = model.resid
    _, p_shapiro = stats.shapiro(residus_new)

    return {
        'label': f'Winsorisation résidus (±{seuil_z} SD)',
        'n_obs': len(df_wins),
        'n_participants': df_wins['Participant'].nunique(),
        'n_winsorise': n_winsorise,
        'n_winsorise_valeurs_brutes': n_winsorise_valeurs,
        'type_modele': type_m,
        'resultats': res,
        'post_hoc': ph,
        'p_shapiro': p_shapiro,
        'n_outliers': 0,
    }


def sensibilite_permutations(df_long, effet_cible, n_perm=5000):
    """Analyse 5 : Test par permutations (non-paramétrique)"""
    # Modèle original
    model_orig, _ = ajuster_lmm(df_long)
    if model_orig is None:
        return None

    clefs = {
        'Trajet': 'C(Trajet, Sum)[S.Away]',
        'Interaction': 'C(Mode_transport, Sum)[S.Bike]:C(Trajet, Sum)[S.Away]'
    }
    coef_key = clefs.get(effet_cible)
    if coef_key is None or coef_key not in model_orig.tvalues:
        return None

    t_obs = abs(model_orig.tvalues[coef_key])

    # Permutations : permuter Trajet au sein de chaque participant
    np.random.seed(42)
    t_perm = np.zeros(n_perm)
    participants = df_long['Participant'].unique()

    print(f"    Permutations ({n_perm})...", end=" ", flush=True)

    for i in range(n_perm):
        df_shuffled = df_long.copy()

        # Permuter Trajet au sein de chaque participant
        for part in participants:
            mask = df_shuffled['Participant'] == part
            trajets = df_shuffled.loc[mask, 'Trajet'].values.copy()
            np.random.shuffle(trajets)
            df_shuffled.loc[mask, 'Trajet'] = trajets

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model_perm = mixedlm(
                    _formule(df_shuffled, "Valeur ~ C(Mode_transport, Sum) * C(Trajet, Sum)"),
                    df_shuffled,
                    groups=df_shuffled["Participant"])
                fit_perm = model_perm.fit(method='lbfgs', maxiter=100)

                if fit_perm.converged and coef_key in fit_perm.tvalues:
                    t_perm[i] = abs(fit_perm.tvalues[coef_key])
                else:
                    t_perm[i] = 0
        except:
            t_perm[i] = 0

        if (i + 1) % 1000 == 0:
            print(f"{i+1}", end=" ", flush=True)

    p_perm = (np.sum(t_perm >= t_obs) + 1) / (n_perm + 1)

    print(f"→ p_perm={p_perm:.4f}")

    return {
        'label': f'Permutations (n={n_perm})',
        'n_obs': len(df_long),
        'n_participants': df_long['Participant'].nunique(),
        'effet_cible': effet_cible,
        't_observe': t_obs,
        'p_permutation': p_perm,
        'significatif': p_perm < ALPHA,
        'pct_95': np.percentile(t_perm, 95),
        'pct_99': np.percentile(t_perm, 99),
    }


# ==============================================================================
# RAPPORT
# ==============================================================================

def generer_rapport(tous_resultats, output_dir):
    """Génère le rapport de sensibilité"""
    rapport_path = os.path.join(output_dir, 'Rapport_Sensibilite.txt')

    with open(rapport_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ANALYSE DE SENSIBILITÉ - INDICATEURS CLÉS\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")

        for nom_indic, resultats in tous_resultats.items():
            f.write(f"\n{'═' * 80}\n")
            f.write(f"  {nom_indic}\n")
            f.write(f"  Diagnostic : {resultats['diagnostic']}\n")
            f.write(f"  Effet cible : {resultats['effet_cible']}\n")
            f.write(f"{'═' * 80}\n\n")

            # ── Tableau comparatif ──
            f.write(f"  {'Analyse':<35} {'p':>8} {'d_av':>8} {'dz':>8} {'η²':>8} {'Sig':>5}  {'Shapiro':>8}  Notes\n")
            f.write(f"  {'─' * 103}\n")

            for analyse in resultats['analyses']:
                if analyse is None:
                    continue

                label = analyse['label']
                effet = resultats['effet_cible']
                res_effet = analyse.get('resultats', {}).get(effet, {})

                p_val = res_effet.get('p', np.nan)
                d_val = res_effet.get('cohens_d', np.nan)
                dz_val = res_effet.get('dz', np.nan)
                eta2 = res_effet.get('eta2', np.nan)
                sig = '*' if res_effet.get('significatif', False) else 'ns'
                p_shap = analyse.get('p_shapiro', np.nan)

                notes = []
                if 'n_exclus' in analyse:
                    notes.append(f"{analyse['n_exclus']} obs exclues")
                if 'n_winsorise' in analyse:
                    notes.append(f"{analyse['n_winsorise']} winsorisées")
                if 'n_outliers' in analyse:
                    notes.append(f"{analyse['n_outliers']} outliers restants")

                p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
                d_str = f"{d_val:.3f}" if not np.isnan(d_val) else "N/A"
                dz_str = f"{dz_val:.3f}" if not np.isnan(dz_val) else "N/A"
                eta2_str = f"{eta2:.4f}" if not np.isnan(eta2) else "N/A"
                shap_str = f"{p_shap:.4f}" if not np.isnan(p_shap) else "N/A"

                f.write(f"  {label:<35} {p_str:>8} {d_str:>8} {dz_str:>8} {eta2_str:>8} {sig:>5}  {shap_str:>8}  {'; '.join(notes)}\n")

            # ── Permutations ──
            perm = resultats.get('permutations')
            if perm:
                f.write(f"\n  Test par permutations ({perm['label']}) :\n")
                f.write(f"    t observé : {perm['t_observe']:.3f}\n")
                f.write(f"    p (permutation) : {perm['p_permutation']:.4f}\n")
                f.write(f"    Significatif : {'Oui' if perm['significatif'] else 'Non'}\n")
                f.write(f"    95e percentile (H0) : {perm['pct_95']:.3f}\n")
                f.write(f"    99e percentile (H0) : {perm['pct_99']:.3f}\n")

            # ── Post-hoc comparatif ──
            f.write(f"\n  Post-hoc comparatif (effet Trajet par Mode) :\n")
            f.write(f"  {'Analyse':<35} {'Vélo p_corr':>12} {'Vélo d_av':>10} {'Vélo dz':>9} {'VAE p_corr':>12} {'VAE d_av':>10} {'VAE dz':>9}\n")
            f.write(f"  {'─' * 100}\n")

            for analyse in resultats['analyses']:
                if analyse is None:
                    continue
                ph = analyse.get('post_hoc', {})
                bike = ph.get('Bike', {})
                ebike = ph.get('e-Bike', {})

                bike_p = f"{bike.get('p_corr', np.nan):.4f}" if 'p_corr' in bike else "N/A"
                bike_d = f"{bike.get('cohens_d', np.nan):.3f}" if 'cohens_d' in bike else "N/A"
                bike_dz = f"{bike.get('cohens_dz', np.nan):.3f}" if 'cohens_dz' in bike else "N/A"
                ebike_p = f"{ebike.get('p_corr', np.nan):.4f}" if 'p_corr' in ebike else "N/A"
                ebike_d = f"{ebike.get('cohens_d', np.nan):.3f}" if 'cohens_d' in ebike else "N/A"
                ebike_dz = f"{ebike.get('cohens_dz', np.nan):.3f}" if 'cohens_dz' in ebike else "N/A"

                f.write(f"  {analyse['label']:<35} {bike_p:>12} {bike_d:>10} {bike_dz:>9} {ebike_p:>12} {ebike_d:>10} {ebike_dz:>9}\n")

            # ── Conclusion ──
            f.write(f"\n  Conclusion :\n")
            analyses_valides = [a for a in resultats['analyses'] if a is not None]
            effet = resultats['effet_cible']

            p_values = [a['resultats'][effet]['p'] for a in analyses_valides
                        if effet in a.get('resultats', {})]
            d_values = [a['resultats'][effet]['cohens_d'] for a in analyses_valides
                        if effet in a.get('resultats', {})]
            dz_values = [a['resultats'][effet]['dz'] for a in analyses_valides
                         if effet in a.get('resultats', {})]
            d_values = [d for d in d_values if not np.isnan(d)]
            dz_values = [d for d in dz_values if not np.isnan(d)]

            if p_values:
                p_range = f"[{min(p_values):.4f} - {max(p_values):.4f}]"
                all_sig = all(p < ALPHA for p in p_values)
                any_sig = any(p < ALPHA for p in p_values)

                f.write(f"    Plage de p ({effet}) : {p_range}\n")
                if d_values:
                    f.write(f"    Plage de d_av : [{min(d_values):.3f} - {max(d_values):.3f}]\n")
                if dz_values:
                    f.write(f"    Plage de dz : [{min(dz_values):.3f} - {max(dz_values):.3f}]\n")

                if all_sig:
                    f.write(f"    → ROBUSTE : l'effet reste significatif dans toutes les analyses.\n")
                elif any_sig:
                    f.write(f"    → PARTIELLEMENT ROBUSTE : l'effet est significatif dans certaines analyses.\n")
                else:
                    f.write(f"    → FRAGILE : l'effet ne survit à aucune analyse de sensibilité.\n")

                if perm and perm['significatif']:
                    f.write(f"    → Le test par permutations confirme l'effet (p={perm['p_permutation']:.4f}).\n")
                elif perm:
                    f.write(f"    → Le test par permutations ne confirme pas l'effet (p={perm['p_permutation']:.4f}).\n")

            f.write("\n")

    print(f"\n  OK: {rapport_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 80)
    print("ANALYSE DE SENSIBILITÉ")
    print("=" * 80)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Charger données
    df = charger_et_filtrer(DATA_FILE, SELECTION_4)
    n_bike = len(df[df['Mode_transport'] == 'Bike'])
    n_ebike = len(df[df['Mode_transport'] == 'e-Bike'])
    print(f"\nSélection 4: {len(df)} participants (Vélo: {n_bike}, VAE: {n_ebike})")

    tous_resultats = {}

    for config in INDICATEURS_SENSIBILITE:
        nom = config['nom']
        effet_cible = config['effet_cible']

        print(f"\n{'═' * 70}")
        print(f"  {nom} (effet cible : {effet_cible})")
        print(f"  Diagnostic : {config['diagnostic']}")
        print(f"{'═' * 70}")

        # Préparer données
        df_long = preparer_donnees_long(df, config['col_away'], config['col_close'],
                                        config['transformation'])
        print(f"  N obs: {len(df_long)}, N participants: {df_long['Participant'].nunique()}")

        analyses = []

        # 1. Original
        print("\n  [1/5] Modèle original...")
        res_orig = sensibilite_original(df_long, effet_cible)
        analyses.append(res_orig)
        if res_orig:
            r = res_orig['resultats'].get(effet_cible, {})
            print(f"         p={r.get('p', np.nan):.4f}, d_av={r.get('cohens_d', np.nan):.3f}, dz={r.get('dz', np.nan):.3f}, "
                  f"outliers={res_orig['n_outliers']}, Shapiro p={res_orig['p_shapiro']:.4f}")

        # 2. Sans outliers |z| > 3
        print("  [2/5] Sans outliers (|z| > 3)...")
        res_no3 = sensibilite_sans_outliers(df_long, effet_cible, seuil_z=3.0)
        analyses.append(res_no3)
        if res_no3:
            r = res_no3['resultats'].get(effet_cible, {})
            print(f"         p={r.get('p', np.nan):.4f}, d_av={r.get('cohens_d', np.nan):.3f}, dz={r.get('dz', np.nan):.3f}, "
                  f"exclus={res_no3['n_exclus']}, Shapiro p={res_no3['p_shapiro']:.4f}")

        # 3. Sans outliers |z| > 2.5
        print("  [3/5] Sans outliers (|z| > 2.5)...")
        res_no25 = sensibilite_sans_outliers(df_long, effet_cible, seuil_z=2.5)
        analyses.append(res_no25)
        if res_no25:
            r = res_no25['resultats'].get(effet_cible, {})
            print(f"         p={r.get('p', np.nan):.4f}, d_av={r.get('cohens_d', np.nan):.3f}, dz={r.get('dz', np.nan):.3f}, "
                  f"exclus={res_no25['n_exclus']}, Shapiro p={res_no25['p_shapiro']:.4f}")

        # 4. Winsorisation
        print("  [4/5] Winsorisation (±3 SD)...")
        res_wins = sensibilite_winsorisee(df_long, effet_cible, seuil_z=3.0)
        analyses.append(res_wins)
        if res_wins:
            r = res_wins['resultats'].get(effet_cible, {})
            print(f"         p={r.get('p', np.nan):.4f}, d_av={r.get('cohens_d', np.nan):.3f}, dz={r.get('dz', np.nan):.3f}")

        # 5. Permutations
        print("  [5/5] Test par permutations...")
        res_perm = sensibilite_permutations(df_long, effet_cible, n_perm=N_PERMUTATIONS)

        tous_resultats[nom] = {
            'diagnostic': config['diagnostic'],
            'effet_cible': effet_cible,
            'analyses': analyses,
            'permutations': res_perm,
        }

    # ── Rapport ──
    print(f"\n{'=' * 60}")
    print("GÉNÉRATION DU RAPPORT")
    print(f"{'=' * 60}")

    generer_rapport(tous_resultats, OUTPUT_DIR)

    # ── Export Excel ──
    rows = []
    for nom_indic, resultats in tous_resultats.items():
        effet = resultats['effet_cible']

        for analyse in resultats['analyses']:
            if analyse is None:
                continue
            res_effet = analyse.get('resultats', {}).get(effet, {})
            ph = analyse.get('post_hoc', {})
            bike = ph.get('Bike', {})
            ebike = ph.get('e-Bike', {})

            rows.append({
                'Indicateur': nom_indic,
                'Effet_cible': effet,
                'Analyse': analyse['label'],
                'N_obs': analyse.get('n_obs', np.nan),
                'N_participants': analyse.get('n_participants', np.nan),
                'Type_modele': analyse.get('type_modele', 'N/A'),
                'p': res_effet.get('p', np.nan),
                't': res_effet.get('t', np.nan),
                'd_av': res_effet.get('cohens_d', np.nan),
                'dz': res_effet.get('dz', np.nan),
                'eta2': res_effet.get('eta2', np.nan),
                'Significatif': res_effet.get('significatif', False),
                'p_Shapiro': analyse.get('p_shapiro', np.nan),
                'N_outliers_restants': analyse.get('n_outliers', np.nan),
                'PostHoc_Bike_p_corr': bike.get('p_corr', np.nan),
                'PostHoc_Bike_d_av': bike.get('cohens_d', np.nan),
                'PostHoc_Bike_dz': bike.get('cohens_dz', np.nan),
                'PostHoc_eBike_p_corr': ebike.get('p_corr', np.nan),
                'PostHoc_eBike_d_av': ebike.get('cohens_d', np.nan),
                'PostHoc_eBike_dz': ebike.get('cohens_dz', np.nan),
            })

        # Ajouter permutations
        perm = resultats.get('permutations')
        if perm:
            rows.append({
                'Indicateur': nom_indic,
                'Effet_cible': effet,
                'Analyse': perm['label'],
                'N_obs': perm.get('n_obs', np.nan),
                'N_participants': perm.get('n_participants', np.nan),
                'Type_modele': 'Permutation',
                'p': perm['p_permutation'],
                't': perm['t_observe'],
                'd_av': np.nan,
                'dz': np.nan,
                'eta2': np.nan,
                'Significatif': perm['significatif'],
                'p_Shapiro': np.nan,
                'N_outliers_restants': np.nan,
            })

    df_export = pd.DataFrame(rows)
    fichier_excel = os.path.join(OUTPUT_DIR, 'Resultats_Sensibilite.xlsx')
    df_export.to_excel(fichier_excel, index=False)
    print(f"  OK: {fichier_excel}")

    # ── Résumé ──
    print(f"\n{'=' * 80}")
    print("ANALYSE TERMINÉE")
    print(f"{'=' * 80}")

    for nom_indic, resultats in tous_resultats.items():
        effet = resultats['effet_cible']
        print(f"\n  {nom_indic} ({effet}) :")
        for analyse in resultats['analyses']:
            if analyse is None:
                continue
            r = analyse.get('resultats', {}).get(effet, {})
            sig = '✓' if r.get('significatif', False) else '✗'
            print(f"    {sig} {analyse['label']:<35} p={r.get('p', np.nan):.4f}  d_av={r.get('cohens_d', np.nan):.3f}, dz={r.get('dz', np.nan):.3f}")

        perm = resultats.get('permutations')
        if perm:
            sig = '✓' if perm['significatif'] else '✗'
            print(f"    {sig} {perm['label']:<35} p={perm['p_permutation']:.4f}")

    print(f"\n  Fichiers dans: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()