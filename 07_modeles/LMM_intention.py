# === CODAGE SOMME DES FACTEURS — corrige le 2026-09-09 ===
# Le modele Mode x Trajet etait ajuste en codage treatment (reference 'Bike'), de sorte que
# le terme lu comme 'effet principal du Trajet' etait l'effet du trajet CHEZ LES CYCLISTES
# CLASSIQUES — un effet simple. Le codage somme C(..., Sum) rend les termes marginaux.
# Les post-hocs par mode et les tailles d'effet (calculees sur les donnees) sont inchanges.
# ATTENTION : C(Trajet, Sum)[S.Away] a le signe OPPOSE a Trajet[T.Close].
# Sauvegarde de la version anterieure dans Data\Codes\_Archives\.
"""
════════════════════════════════════════════════════════════════════════════════
ANALYSE LMM — ANALYSE D'INTENTION (script fusionné)

Fusionne LMM_int_FDR_2niveau.py (FDR 2 niveaux) et LMM_int_FDR_2niveau_2.py
(FDR par domaine). Les deux scripts utilisaient le MÊME modèle, le MÊME
échantillon (SELECTION_4, paires complètes) et les MÊMES 26 indicateurs ;
seule la partition des familles pour la correction FDR différait.

Principe : ajuster le LMM UNE seule fois par indicateur, puis appliquer la
correction FDR Benjamini-Hochberg DEUX fois selon deux partitions :

  STRATÉGIE A — FDR 2 niveaux (statut d'hypothèse) :
    Principale  = N-Back (8 tests)   |  Secondaire = ANT (23 tests)

  STRATÉGIE B — FDR par domaine (type de mesure) :
    Électrophysiologique = EEG (17)  |  Comportemental = Comport. (14)

Modèle : Valeur ~ Mode_transport * Trajet, groups=Participant,
         re_formula="~Trajet" (fallback intercept seul).
Post-hoc : effet Trajet par Mode (Bonferroni ×2) si interaction.
Échantillon : SELECTION_4 (N=51), paires complètes A–C (listwise).

Sortie supplémentaire : Comparaison_Strategies_FDR.xlsx — pour chaque
indicateur/effet, p brut + q sous chaque stratégie (rend reproductible le
contenu des Rapport_Comparaison_*.docx).

NB : total ANT = 23 (RT×5 + ER×5 + flèche 10 + sonore 2 + indice 1).
Les ER (taux d'erreur, 5 réseaux) ont été ajoutés le 2026-05-19 en plus des
RT ; transformation=None (convention projet : ER classé IDENTITY dans
ancien/transformation_data_lmm.py, contrairement au RT en log). Les anciens
LMM_int_FDR_2niveau*.py ne testaient PAS les ER (ANT=18).
════════════════════════════════════════════════════════════════════════════════
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402

import sys
import os
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

# Console UTF-8 (caractères de cadre / accents dans les print)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_FILE = D(r'Data\Table_combine\Tab_sujet_20260205.xlsx')
OUTPUT_DIR = S(r'Resultats_LMM_intention\26-10-07_codage_somme_sans134')

VERSION_DONNEES = 'filtre'

SELECTION_4 = [101, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
               201, 202, 203, 204, 205, 206, 209, 210, 211, 212, 215, 216,
               121, 122, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 139,
               221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]

ALPHA = 0.05
METHODE_CORRECTION = 'fdr_bh'
STRATEGIE_NAN = 'paires_completes'

# Les deux partitions de familles FDR.
#   clé interne -> (sous-dossier, fonction d'étiquetage des familles)
# (les fonctions d'étiquetage sont définies plus bas, après le catalogue)


# ==============================================================================
# CATALOGUE UNIQUE DES 26 INDICATEURS
# (chaque indicateur porte 'domaine' EEG/Comportement ET 'tache' 1B/2B/ANT ;
#  la catégorie N-Back/ANT en découle — d'où les deux partitions sans
#  dupliquer la définition des indicateurs)
# ==============================================================================

def generer_catalogue():
    """26 indicateurs : 8 N-Back + 18 ANT (= 17 EEG + 9 Comportement)."""
    indicateurs = []

    # ── N-Back EEG : N1 et P3 (Valides) × 1B/2B (4) ──
    for tache in ['1B', '2B']:
        for composante in ['N1', 'P3']:
            indicateurs.append({
                'nom': f'{composante}_Moy_Valide_{tache}',
                'tache': tache,
                'domaine': 'EEG',
                'col_away': f'{composante}_Moy_Valide_A_{tache}',
                'col_close': f'{composante}_Moy_Valide_C_{tache}',
                'transformation': None,
            })

    # ── ANT EEG flèche : N1 + P3 × 5 réseaux (10) ──
    # ⚠ "Controle_Executif" -> 'Controle' dans les colonnes flèche
    reseaux_eeg_fleche = {
        'Alerte_Standard': 'Alerte_Standard',
        'Alerte_Auditive': 'Alerte_Auditive',
        'Alerte_Visuelle': 'Alerte_Visuelle',
        'Orientation': 'Orientation',
        'Controle_Executif': 'Controle',
    }
    for reseau_label, cond in reseaux_eeg_fleche.items():
        for composante in ['N1', 'P3']:
            indicateurs.append({
                'nom': f'{composante}_Moy_fleche_{reseau_label}_ANT',
                'tache': 'ANT',
                'domaine': 'EEG',
                'col_away': f'{composante}_{cond}_Moy_fleche_A_ANT',
                'col_close': f'{composante}_{cond}_Moy_fleche_C_ANT',
                'transformation': None,
            })

    # ── ANT EEG sonore : N1 + P2 (Alerte Standard) (2) ──
    for composante in ['N1', 'P2']:
        indicateurs.append({
            'nom': f'{composante}_Moy_sonore_Alerte_Standard_ANT',
            'tache': 'ANT',
            'domaine': 'EEG',
            'col_away': f'{composante}_Alerte_Standard_Moy_sonore_A_ANT',
            'col_close': f'{composante}_Alerte_Standard_Moy_sonore_C_ANT',
            'transformation': None,
        })

    # ── ANT EEG indice : N1 (Alerte Visuelle) (1) ──
    indicateurs.append({
        'nom': 'N1_Moy_indice_Alerte_Visuelle_ANT',
        'tache': 'ANT',
        'domaine': 'EEG',
        'col_away': 'N1_Alerte_Visuelle_Moy_indice_A_ANT',
        'col_close': 'N1_Alerte_Visuelle_Moy_indice_C_ANT',
        'transformation': None,
    })

    # ── N-Back Comportement : RT (log) et d' × 1B/2B (4) ──
    for tache in ['1B', '2B']:
        indicateurs.append({
            'nom': f'RT_moyen_{tache}',
            'tache': tache,
            'domaine': 'Comportement',
            'col_away': f'RT_moyen_{VERSION_DONNEES}_A_{tache}',
            'col_close': f'RT_moyen_{VERSION_DONNEES}_C_{tache}',
            'transformation': 'log',
        })
        indicateurs.append({
            'nom': f'd_prime_{tache}',
            'tache': tache,
            'domaine': 'Comportement',
            'col_away': f'd_prime_{VERSION_DONNEES}_A_{tache}',
            'col_close': f'd_prime_{VERSION_DONNEES}_C_{tache}',
            'transformation': None,
        })

    # ── ANT Comportement : RT (log) × 5 réseaux (5) ──
    reseaux_rt = ['Alerte_Standard', 'Alerte_Auditive', 'Alerte_Visuelle',
                  'Orientation', 'Controle_Executif']
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
            'transformation': None,
        })

    # ── ANT Comportement : ER (taux d'erreur) × 5 réseaux (5) ──
    # Colonnes ER_<reseau>_{A|C}_ANT (noms longs, idem RT — vérifié sur
    # Tab_sujet_20260205). transformation=None (ER classé IDENTITY dans
    # ancien/transformation_data_lmm.py ; pas de log contrairement au RT).
    for reseau in reseaux_rt:
        indicateurs.append({
            'nom': f'ER_{reseau}_ANT',
            'tache': 'ANT',
            'domaine': 'Comportement',
            'col_away': f'ER_{reseau}_A_ANT',
            'col_close': f'ER_{reseau}_C_ANT',
            'transformation': None,
        })

    # Catégorie N-Back / ANT (axe de la stratégie "2 niveaux")
    for ind in indicateurs:
        ind['categorie'] = 'N-Back' if ind['tache'] in ('1B', '2B') else 'ANT'

    return indicateurs


# Partitions de familles FDR : chaque entrée renvoie (sous-dossier global,
# liste de (label_famille, prédicat sur l'indicateur)).
PARTITIONS = {
    'FDR_2niveaux': {
        'sous_dossier': 'FDR_2niveaux',
        'titre': 'FDR 2 NIVEAUX (Principale N-Back / Secondaire ANT)',
        'familles': [
            ('Principale', lambda r: r['Categorie'] == 'N-Back'),
            ('Secondaire', lambda r: r['Categorie'] == 'ANT'),
        ],
    },
    'FDR_domaine': {
        'sous_dossier': 'FDR_domaine',
        'titre': 'FDR PAR DOMAINE (Électrophysiologique / Comportemental)',
        'familles': [
            ('Electrophysiologique', lambda r: r['Domaine'] == 'EEG'),
            ('Comportemental', lambda r: r['Domaine'] == 'Comportement'),
        ],
    },
}


# ==============================================================================
# UTILITAIRES (repris à l'identique de LMM_int_FDR_2niveau_2.py)
# ==============================================================================

def charger_donnees(fichier_path):
    print(f"Chargement: {fichier_path}...")
    df = pd.read_excel(fichier_path)
    # === 07/10/2026 : le participant 134 n'a donné
    # aucune réponse pendant le 2-back de son trajet éloigné (0 hit, 42 omissions, 0 fausse alarme ; boîtier de
    # réponse défaillant, la tâche a été faite). Cette session est retirée des indicateurs comportementaux du 2-back
    # (le TR y était déjà indéfini ; le d' = 0,31 n'est pas une mesure de performance). Les ERP, calculés sur tous
    # les essais cibles, sont conservés. Sorties du 09/09 (avec 134) : 26-09-09_codage_somme.
    df.loc[df['Participant'] == 134, [c for c in df.columns if c.startswith('d_prime_') and c.endswith('_A_2B')]] = float('nan')
    _ordre = _charger_ordre()
    if _ordre is not None and 'Participant' in df.columns:
        df = df.merge(_ordre, on='Participant', how='left')
    print(f"  OK: {df.shape[0]} participants, {df.shape[1]} colonnes")
    return df


def filtrer_selection(df, selection_ids):
    return df[df['Participant'].isin(selection_ids)].copy()


def nettoyer_donnees_paire(df, col_away, col_close):
    """Nettoie les données (paires complètes / listwise A–C)."""
    stats_nan = {
        'n_initial': len(df),
        'n_nan_away': df[col_away].isna().sum(),
        'n_nan_close': df[col_close].isna().sum(),
        'participants_exclus': [],
        'n_final': 0,
        'pct_perte': 0,
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
# DIAGNOSTICS & STATISTIQUES (repris à l'identique)
# ==============================================================================

def tester_normalite(valeurs, seuil_p=0.05):
    resultats = {'conclusion': 'INSUFFISANT', 'p_shapiro': np.nan, 'p_dagostino': np.nan}
    if len(valeurs) < 3:
        return resultats
    if len(valeurs) <= 5000:
        try:
            _, p_sw = stats.shapiro(valeurs)
            resultats['p_shapiro'] = p_sw
        except Exception:
            pass
    if len(valeurs) >= 20:
        try:
            _, p_dp = stats.normaltest(valeurs)
            resultats['p_dagostino'] = p_dp
        except Exception:
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
        except Exception:
            diagnostics['homoscedasticite_p'] = np.nan
        z_scores = np.abs((residus - residus.mean()) / residus.std())
        diagnostics['n_outliers'] = int((z_scores > 3).sum())
    except Exception as e:
        diagnostics['erreur'] = str(e)
    return diagnostics


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


def calculer_taille_effet(model_result, effet_type, df_long):
    """Tailles d'effet référencées à la variance TOTALE (SD brutes).

    - Trajet (intra-sujet) -> Cohen's d_av (variance totale, comparable) ET
      dz (référence intra) -- on reporte les DEUX (Lakens, 2013).
    - Mode   (inter-sujet) -> Cohen's d_s poolé (dz non défini en inter).
    - Interaction          -> pas de d (un d d'interaction n'est pas
      interprétable) ; le crossover est porté par le eta2 + les post-hocs.

    eta2 : eta2 PARTIEL = t²/(t²+df_resid), dérivé de la statistique t du LMM.
    Il appartient au cadre intra/signal (cohérent avec dz), à NE PAS confondre
    avec un eta2 généralisé."""
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
# AJUSTEMENT LMM + POST-HOC (repris à l'identique)
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
    """LMM avec fallback (pente aléatoire → intercept seul)."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = mixedlm(_formule(df_long, "Valeur ~ C(Mode_transport, Sum) * C(Trajet, Sum)"), df_long,
                            groups=df_long["Participant"], re_formula="~Trajet")
            result = model.fit(method='lbfgs', maxiter=200)
            if result.converged:
                return result, 'LMM_pente'
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = mixedlm(_formule(df_long, "Valeur ~ C(Mode_transport, Sum) * C(Trajet, Sum)"), df_long,
                            groups=df_long["Participant"])
            result = model.fit(method='lbfgs', maxiter=200)
            if result.converged:
                return result, 'LMM_intercept'
    except Exception:
        pass
    return None, 'Echec'


def tester_effet_trajet_par_mode(df_long):
    """Post-hoc : effet Trajet par Mode (Bonferroni ×2)."""
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
# ANALYSE D'UN INDICATEUR (ajustement UNIQUE, sans FDR)
# ==============================================================================

def analyser_indicateur(df, config_indic):
    """Ajuste le LMM d'un indicateur et renvoie les résultats BRUTS (pas de FDR)."""
    nom = config_indic['nom']
    col_away = config_indic['col_away']
    col_close = config_indic['col_close']
    transformation = config_indic['transformation']
    tache = config_indic['tache']
    domaine = config_indic['domaine']
    categorie = config_indic['categorie']

    print(f"  {nom}...", end=" ")

    if col_away not in df.columns or col_close not in df.columns:
        print("SKIP (colonnes manquantes)")
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
        'Categorie': categorie,
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
# CORRECTION FDR (repris à l'identique ; opère sur une COPIE de famille)
# ==============================================================================

def appliquer_correction_fdr(df_resultats, label_niveau, verbose=True):
    """Applique la correction FDR sur un sous-ensemble (famille)."""
    df_resultats = df_resultats.reset_index(drop=True).copy()
    n_tests = len(df_resultats)
    if verbose:
        print(f"\n  Correction FDR ({METHODE_CORRECTION}) - {label_niveau} ({n_tests} tests)...")

    for effet in ['Mode', 'Trajet', 'Interaction']:
        col_p = f'p_{effet}'
        col_p_fdr = f'p_{effet}_FDR'
        col_sig = f'Sig_{effet}_FDR'

        p_values = df_resultats[col_p].values
        mask_valid = ~np.isnan(p_values)
        p_fdr = np.full_like(p_values, np.nan, dtype=float)
        if mask_valid.sum() > 0:
            _, p_corrected, _, _ = multipletests(
                p_values[mask_valid], alpha=ALPHA, method=METHODE_CORRECTION
            )
            p_fdr[mask_valid] = p_corrected

        df_resultats[col_p_fdr] = p_fdr
        df_resultats[col_sig] = df_resultats[col_p_fdr].apply(
            lambda x: 'N/A' if np.isnan(x)
            else '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns'
        )
        if verbose:
            n_sig = df_resultats[col_sig].isin(['*', '**', '***']).sum()
            print(f"    {effet}: {n_sig}/{int(mask_valid.sum())} significatifs après FDR")

    return df_resultats


# ==============================================================================
# RAPPORTS (repris/adaptés de LMM_int_FDR_2niveau_2.py)
# ==============================================================================

COLONNES_EXPORT = [
    'Indicateur', 'Tache', 'Domaine', 'Categorie', 'Transformation',
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


def generer_rapport_famille(df_resultats, label_famille, output_subdir):
    rapport_path = os.path.join(output_subdir, f'Rapport_{label_famille}.txt')
    with open(rapport_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RAPPORT LMM - {label_famille.upper()}\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Correction : FDR Benjamini-Hochberg sur {len(df_resultats)} tests\n")
        f.write("=" * 80 + "\n\n")

        for effet in ['Mode', 'Trajet', 'Interaction']:
            col_sig = f'Sig_{effet}_FDR'
            col_p = f'p_{effet}_FDR'
            col_p_brut = f'p_{effet}'
            col_d = f'd_{effet}'
            n_sig = df_resultats[col_sig].isin(['*', '**', '***']).sum()
            f.write(f"\n{'─' * 60}\n")
            f.write(f"EFFET {effet.upper()} : {n_sig} significatifs après FDR\n")
            f.write(f"{'─' * 60}\n\n")
            for _, row in df_resultats.sort_values(col_p_brut).iterrows():
                sig_flag = row[col_sig] if row[col_sig] in ['*', '**', '***'] else '  '
                f.write(f"  {sig_flag:>3} {row['Indicateur']:<45} "
                        f"p={row[col_p_brut]:.4f}  p_FDR={row[col_p]:.4f}  "
                        f"d={row[col_d]:.3f}  η²={row[f'eta2_{effet}']:.4f}\n")

        f.write(f"\n\n{'=' * 80}\n")
        f.write("POST-HOC (Interaction significative après FDR)\n")
        f.write(f"{'=' * 80}\n\n")
        inter_sig = df_resultats[df_resultats['Sig_Interaction_FDR'].isin(['*', '**', '***'])]
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
        else:
            for _, row in inter_sig.iterrows():
                f.write(f"  {row['Indicateur']}:\n")
                f.write(f"    Interaction: p_FDR={row['p_Interaction_FDR']:.4f}, "
                        f"d={row['d_Interaction']:.3f}\n")
                bike_sig = row.get('PostHoc_Bike_sig', False)
                ebike_sig = row.get('PostHoc_eBike_sig', False)
                f.write(f"    Vélo: p_corr={row['PostHoc_Bike_p_corr']:.4f}, "
                        f"d={row['PostHoc_Bike_d']:.3f} "
                        f"{'* ' + str(row['PostHoc_Bike_direction']) if bike_sig else 'ns'}\n")
                f.write(f"    VAE:  p_corr={row['PostHoc_eBike_p_corr']:.4f}, "
                        f"d={row['PostHoc_eBike_d']:.3f} "
                        f"{'* ' + str(row['PostHoc_eBike_direction']) if ebike_sig else 'ns'}\n\n")

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


def generer_rapport_global(familles_res, titre, output_dir):
    """Synthèse d'une stratégie (familles = liste de (label, df))."""
    rapport_path = os.path.join(output_dir, 'Rapport_Synthese.txt')
    with open(rapport_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"SYNTHÈSE - {titre}\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")
        for label, df_res in familles_res:
            f.write(f"\n{'═' * 80}\n{label}  ({len(df_res)} tests)\n{'═' * 80}\n\n")
            f.write(f"  Tâches incluses : {', '.join(sorted(df_res['Tache'].unique()))}\n\n")
            for effet in ['Mode', 'Trajet', 'Interaction']:
                col_sig = f'Sig_{effet}_FDR'
                col_p_fdr = f'p_{effet}_FDR'
                col_d = f'd_{effet}'
                n_sig = df_res[col_sig].isin(['*', '**', '***']).sum()
                f.write(f"  {effet}: {n_sig} significatifs\n")
                for _, row in df_res[df_res[col_sig].isin(['*', '**', '***'])].sort_values(col_p_fdr).iterrows():
                    f.write(f"    ✓ {row['Indicateur']:<40} [{row['Tache']}] "
                            f"p_FDR={row[col_p_fdr]:.4f}  d={row[col_d]:.3f}\n")
                for _, row in df_res[(df_res[col_p_fdr] < 0.10) &
                                     (~df_res[col_sig].isin(['*', '**', '***']))].sort_values(col_p_fdr).iterrows():
                    f.write(f"    ~ {row['Indicateur']:<40} [{row['Tache']}] "
                            f"p_FDR={row[col_p_fdr]:.4f}  d={row[col_d]:.3f}  (tendance)\n")
            f.write("\n")
    print(f"  OK: {rapport_path}")


# ==============================================================================
# ORCHESTRATION : 1 ajustement → 2 stratégies FDR → feuille comparative
# ==============================================================================

def appliquer_strategie(df_brut, cle_strategie):
    """Applique une partition FDR ; écrit Excel + rapports ; renvoie df concaténé taggé."""
    spec = PARTITIONS[cle_strategie]
    out_dir = os.path.join(OUTPUT_DIR, spec['sous_dossier'])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 70}\n  STRATÉGIE : {spec['titre']}\n{'═' * 70}")
    familles_res = []
    for label, predicat in spec['familles']:
        mask = df_brut.apply(predicat, axis=1)
        df_fam = df_brut[mask].copy()
        df_fam = appliquer_correction_fdr(df_fam, label)

        sous = os.path.join(out_dir, label)
        Path(sous).mkdir(parents=True, exist_ok=True)
        cols = [c for c in COLONNES_EXPORT if c in df_fam.columns]
        df_fam[cols].to_excel(os.path.join(sous, f'Resultats_{label}.xlsx'), index=False)
        generer_rapport_famille(df_fam, label, sous)

        df_tag = df_fam.copy()
        df_tag['Famille'] = label
        df_tag['Strategie'] = cle_strategie
        familles_res.append((label, df_fam))

    generer_rapport_global(familles_res, spec['titre'], out_dir)
    df_concat = pd.concat([d.assign(Famille=lbl, Strategie=cle_strategie)
                           for lbl, d in familles_res], ignore_index=True)
    df_concat[[c for c in COLONNES_EXPORT if c in df_concat.columns] + ['Famille', 'Strategie']
              ].to_excel(os.path.join(out_dir, 'Resultats_Combines.xlsx'), index=False)
    return df_concat


def generer_feuille_comparative(df_strategies, output_dir):
    """
    Côte-à-côte : pour chaque indicateur/effet, p brut + q sous chaque
    stratégie (le p brut est IDENTIQUE entre stratégies — ajustement unique).
    """
    base_cols = ['Indicateur', 'Tache', 'Domaine', 'Categorie']
    effets = ['Mode', 'Trajet', 'Interaction']

    lignes = {}
    for _, r in df_strategies.iterrows():
        ind = r['Indicateur']
        if ind not in lignes:
            ligne = {c: r[c] for c in base_cols}
            for e in effets:
                ligne[f'p_{e}_brut'] = r[f'p_{e}']
            lignes[ind] = ligne
        strat = 'A_2niveaux' if r['Strategie'] == 'FDR_2niveaux' else 'B_domaine'
        for e in effets:
            lignes[ind][f'q_{e}_{strat}'] = r[f'p_{e}_FDR']
            lignes[ind][f'Sig_{e}_{strat}'] = r[f'Sig_{e}_FDR']

    df_cmp = pd.DataFrame(list(lignes.values()))

    # Indicateur de divergence : la significativité change-t-elle de stratégie ?
    for e in effets:
        a = df_cmp.get(f'Sig_{e}_A_2niveaux')
        b = df_cmp.get(f'Sig_{e}_B_domaine')

        def _sig(x):
            return x in ('*', '**', '***')

        df_cmp[f'Divergence_{e}'] = [
            'DIVERGENT' if _sig(x) != _sig(y) else ''
            for x, y in zip(a, b)
        ]

    ordre = base_cols.copy()
    for e in effets:
        ordre += [f'p_{e}_brut', f'q_{e}_A_2niveaux', f'Sig_{e}_A_2niveaux',
                  f'q_{e}_B_domaine', f'Sig_{e}_B_domaine', f'Divergence_{e}']
    df_cmp = df_cmp[[c for c in ordre if c in df_cmp.columns]]

    for c in df_cmp.columns:
        if c.startswith(('p_', 'q_')):
            df_cmp[c] = df_cmp[c].round(4)

    fichier = os.path.join(output_dir, 'Comparaison_Strategies_FDR.xlsx')
    df_cmp.to_excel(fichier, index=False)
    print(f"\n  OK: feuille comparative → {fichier}")

    n_div = sum((df_cmp[f'Divergence_{e}'] == 'DIVERGENT').sum() for e in effets)
    print(f"  Décisions de significativité divergentes entre stratégies : {n_div}")
    return df_cmp


def main():
    print("\n" + "=" * 80)
    print("ANALYSE LMM - ANALYSE D'INTENTION (fusionné : 2 stratégies FDR)")
    print("=" * 80)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\nDossier: {OUTPUT_DIR}")

    catalogue = generer_catalogue()
    n_nback = sum(1 for i in catalogue if i['categorie'] == 'N-Back')
    n_ant = sum(1 for i in catalogue if i['categorie'] == 'ANT')
    n_eeg = sum(1 for i in catalogue if i['domaine'] == 'EEG')
    n_comp = sum(1 for i in catalogue if i['domaine'] == 'Comportement')
    print(f"\nCatalogue : {len(catalogue)} indicateurs")
    print(f"  Axe 2 niveaux  : N-Back={n_nback}, ANT={n_ant}")
    print(f"  Axe domaine    : EEG={n_eeg}, Comportement={n_comp}")

    df = charger_donnees(DATA_FILE)
    df_sel = filtrer_selection(df, SELECTION_4)
    n_bike = len(df_sel[df_sel['Mode_transport'] == 'Bike'])
    n_ebike = len(df_sel[df_sel['Mode_transport'] == 'e-Bike'])
    print(f"\nSélection 4: {len(df_sel)} participants (Vélo: {n_bike}, VAE: {n_ebike})")

    # ── Ajustement LMM UNIQUE (indépendant du découpage FDR) ──
    print(f"\n{'═' * 70}\n  AJUSTEMENT LMM ({len(catalogue)} indicateurs, une seule fois)\n{'═' * 70}\n")
    resultats = [r for r in (analyser_indicateur(df_sel, c) for c in catalogue) if r is not None]
    if not resultats:
        print("ERREUR: aucun indicateur analysable")
        return
    df_brut = pd.DataFrame(resultats)
    df_brut.to_excel(os.path.join(OUTPUT_DIR, 'Resultats_LMM_bruts_sans_FDR.xlsx'), index=False)
    print(f"\nOK - {len(df_brut)}/{len(catalogue)} indicateurs ajustés")

    # ── Application des deux stratégies FDR ──
    df_a = appliquer_strategie(df_brut, 'FDR_2niveaux')
    df_b = appliquer_strategie(df_brut, 'FDR_domaine')

    # ── Feuille comparative ──
    df_strategies = pd.concat([df_a, df_b], ignore_index=True)
    generer_feuille_comparative(df_strategies, OUTPUT_DIR)

    # ── Résumé console ──
    print(f"\n{'=' * 80}\nTERMINÉ\n{'=' * 80}")
    for cle in ('FDR_2niveaux', 'FDR_domaine'):
        sub = df_strategies[df_strategies['Strategie'] == cle]
        print(f"\n  {PARTITIONS[cle]['titre']}")
        for fam in sub['Famille'].unique():
            d = sub[sub['Famille'] == fam]
            bilan = ", ".join(
                f"{e}={d[f'Sig_{e}_FDR'].isin(['*', '**', '***']).sum()}"
                for e in ['Mode', 'Trajet', 'Interaction']
            )
            print(f"    {fam:<22} ({len(d)} tests) : {bilan} significatifs")
    print(f"\n  Sorties dans : {OUTPUT_DIR}")
    print("    FDR_2niveaux/  (Principale, Secondaire)")
    print("    FDR_domaine/   (Electrophysiologique, Comportemental)")
    print("    Comparaison_Strategies_FDR.xlsx")
    print("    Resultats_LMM_bruts_sans_FDR.xlsx")


if __name__ == "__main__":
    main()
