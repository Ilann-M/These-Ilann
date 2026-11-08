"""
================================================================================
ANALYSE DE REGRESSION CONTINUE - EXPOSITION -> PERFORMANCE COGNITIVE
Version FDR par TACHE (N-Back : 1B + 2B  /  ANT)

Principe :
  Regression lineaire OLS (et non LMM) pour etablir un lien dose-reponse
  entre niveau d'exposition continu (BC880, Number, PM2.5) et performance cognitive.

  Trois volets par couple (polluant x indicateur) :
    - Volet 1A : Perf_A ~ Exposition_A + Mode_transport
    - Volet 1C : Perf_C ~ Exposition_C + Mode_transport
    - Volet 2D : (Perf_C - Perf_A) ~ (Exposition_C - Exposition_A) + Mode_transport

  Transformations d'exposition en parallele (sensibilite) :
    - sans_log : exposition brute
    - log_expo : log(exposition)
  Pour la difference avec log : log(Expo_C) - log(Expo_A) = log(Expo_C/Expo_A).

  Variables RT log-transformees comme dans les LMM existants.

Trois polluants (modeles separes, pas de multiple) :
    1) BC880_Mean_A_EXP / BC880_Mean_C_EXP
    2) Number_Mean_A_EXP / Number_Mean_C_EXP

Deux taches :
  N-BACK (8 tests)  : N1/P3 Valides x 1B/2B + RT_moyen(log)/d' x 1B/2B
  ANT (18 tests)    : N1/P3 fleche x 5 reseaux + N1/P2 sonore + N1 indice + RT x 5

FDR Benjamini-Hochberg applique par lot independant :
  (polluant x transformation x volet x tache) = 36 lots (24 avant PM2.5)

Selection : SELECTION_4 (N=51, sujets ayant fait les deux trajets).
Listwise deletion des NaN par modele.
Covariable : Mode_transport (Bike / e-Bike) en C(Mode_transport).

Sorties locales :
  <racine>/Data/Traitement/Resultats_regression_continue/<date>_FDR_taches/
    <polluant>/<transformation>/<tache>/Resultats_<tache>.xlsx
                                  /Diagnostics/
    <polluant>/Synthese.xlsx
    Synthese_globale.xlsx
================================================================================
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
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import het_breuschpagan


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_FILE = D(r'Data\Table_combine\Tab_sujet_20260205.xlsx')
OUTPUT_DIR = S(r'Resultats_regression_continue\26-10-16_concentration_sans134')

VERSION_DONNEES = 'filtre'

SELECTION_4 = [101, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
               201, 202, 203, 204, 205, 206, 209, 210, 211, 212, 215, 216,
               121, 122, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 139,
               221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]

ALPHA = 0.05
METHODE_CORRECTION = 'fdr_bh'

POLLUANTS = [
    {'nom': 'BC880_Mean',  'col_A': 'BC880_Mean_A_EXP',  'col_C': 'BC880_Mean_C_EXP'},
    {'nom': 'Number_Mean', 'col_A': 'Number_Mean_A_EXP', 'col_C': 'Number_Mean_C_EXP'},
    # PM2.5 ajoute le 03/09. Les niveaux 2 et 3 l'ecartent comme critere, un
    # filtrage a 1,1 ne retenant que 30 participants sur 51 et un reetiquetage
    # en deplacant 18 ; le niveau 4 ne filtre ni ne reetiquette, il n'y perd
    # donc aucun effectif et peut porter le marqueur reglementaire. La
    # correction reste par lot (polluant x transformation x volet x tache,
    # 3.9.6) : ces 12 lots nouveaux ne modifient aucun q deja publie.
    {'nom': 'PM25_Mean',   'col_A': 'PM2.5_Mean_A_EXP',  'col_C': 'PM2.5_Mean_C_EXP'},
]

TRANSFORMATIONS = ['sans_log', 'log_expo']

# A1 (03/09) : ordre des trajets en covariable, pour aligner ce niveau sur les
# trois autres, qui portent un terme de session. Sur les volets par trajet,
# l'ordre dit si le trajet observe etait la 1re ou la 2e session de test ; sur
# le volet en differences, il dit le signe qu'y prend un effet de pratique.
AVEC_ORDRE = True
BASE_ORDRE = D(r'Data\Base.xlsx')


# ==============================================================================
# DEFINITION DES INDICATEURS (copie des scripts LMM existants)
# ==============================================================================

def generer_indicateurs_nback():
    """N-BACK : 8 indicateurs (1B + 2B, EEG + Comportement)."""
    indicateurs = []
    for tache in ['1B', '2B']:
        for composante in ['N1', 'P3']:
            indicateurs.append({
                'nom': f'{composante}_Moy_Valide_{tache}',
                'tache': 'NBack',
                'domaine': 'EEG',
                'col_away': f'{composante}_Moy_Valide_A_{tache}',
                'col_close': f'{composante}_Moy_Valide_C_{tache}',
                'transformation': None,
            })
    for tache in ['1B', '2B']:
        indicateurs.append({
            'nom': f'RT_moyen_{tache}',
            'tache': 'NBack',
            'domaine': 'Comportement',
            'col_away': f'RT_moyen_{VERSION_DONNEES}_A_{tache}',
            'col_close': f'RT_moyen_{VERSION_DONNEES}_C_{tache}',
            'transformation': 'log',
        })
        indicateurs.append({
            'nom': f'd_prime_{tache}',
            'tache': 'NBack',
            'domaine': 'Comportement',
            'col_away': f'd_prime_{VERSION_DONNEES}_A_{tache}',
            'col_close': f'd_prime_{VERSION_DONNEES}_C_{tache}',
            'transformation': None,
        })
    return indicateurs


def generer_indicateurs_ant():
    """ANT : 18 indicateurs (EEG + Comportement)."""
    indicateurs = []
    reseaux_eeg_fleche = {
        'Alerte_Standard': 'Alerte_Standard',
        'Alerte_Auditive': 'Alerte_Auditive',
        'Alerte_Visuelle': 'Alerte_Visuelle',
        'Orientation': 'Orientation',
        'Controle_Executif': 'Controle',
    }
    for reseau_label, condition_eeg in reseaux_eeg_fleche.items():
        for composante in ['N1', 'P3']:
            indicateurs.append({
                'nom': f'{composante}_Moy_fleche_{reseau_label}_ANT',
                'tache': 'ANT',
                'domaine': 'EEG',
                'col_away': f'{composante}_{condition_eeg}_Moy_fleche_A_ANT',
                'col_close': f'{composante}_{condition_eeg}_Moy_fleche_C_ANT',
                'transformation': None,
            })
    for composante in ['N1', 'P2']:
        indicateurs.append({
            'nom': f'{composante}_Moy_sonore_Alerte_Standard_ANT',
            'tache': 'ANT',
            'domaine': 'EEG',
            'col_away': f'{composante}_Alerte_Standard_Moy_sonore_A_ANT',
            'col_close': f'{composante}_Alerte_Standard_Moy_sonore_C_ANT',
            'transformation': None,
        })
    indicateurs.append({
        'nom': 'N1_Moy_indice_Alerte_Visuelle_ANT',
        'tache': 'ANT',
        'domaine': 'EEG',
        'col_away': 'N1_Alerte_Visuelle_Moy_indice_A_ANT',
        'col_close': 'N1_Alerte_Visuelle_Moy_indice_C_ANT',
        'transformation': None,
    })
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
# UTILITAIRES
# ==============================================================================

def charger_donnees(fichier_path):
    print(f"Chargement: {fichier_path}...")
    df = pd.read_excel(fichier_path)
    print(f"  Total: N={len(df)} x {df.shape[1]} colonnes")
    # Session 2-back du trajet eloigne du participant 134 : boitier de reponse defaillant (0 reponse).
    # Retiree des indicateurs comportementaux du 2-back, comme au niveau 1 (LMM_intention.py, chapitre 3).
    df.loc[df['Participant'] == 134, [c for c in df.columns if c.startswith('d_prime_') and c.endswith('_A_2B')]] = float('nan')
    return df


def charger_ordre():
    """Participant -> 'A_first' / 'C_first', d'apres Base.xlsx.

    Meme logique que LMM_intention.py, pour que les quatre niveaux lisent
    l'ordre de la meme source et de la meme facon.
    """
    if not AVEC_ORDRE:
        return None
    try:
        b = pd.read_excel(BASE_ORDRE)
    except Exception as e:
        print(f"  [!] ordre indisponible ({type(e).__name__}) : covariable omise")
        return None
    b.columns = [str(c).strip() for c in b.columns]
    col = 'Passation-2' if 'Passation-2' in b.columns else 'Passation_2'
    if col not in b.columns or 'Participant' not in b.columns:
        print("  [!] colonne d'ordre absente : covariable omise")
        return None
    b['Participant'] = b['Participant'].astype(str).str.extract(r'[Pp](\d+)')[0]
    b = b.dropna(subset=['Participant'])
    b['Participant'] = b['Participant'].astype(int)
    p2 = b[col].astype(str).str.strip().str.upper()
    b['Ordre'] = np.where(p2 == 'A', 'A_first',
                          np.where(p2 == 'C', 'C_first', np.nan))
    out = b[['Participant', 'Ordre']].dropna()
    print(f"  Ordre charge pour {len(out)} participants "
          f"({(out.Ordre == 'A_first').sum()} A_first / "
          f"{(out.Ordre == 'C_first').sum()} C_first)")
    return out


def _covariables(sub):
    """Termes de controle disponibles et informatifs sur ce sous-echantillon."""
    termes = ['C(Mode_transport)']
    if 'Ordre' in sub.columns and sub['Ordre'].nunique() > 1:
        termes.append('C(Ordre)')
    return ' + '.join(termes)


def filtrer_selection(df, selection, col_participant='Participant'):
    df_sel = df[df[col_participant].isin(selection)].copy()
    print(f"  SELECTION_4: N={len(df_sel)} / {len(selection)}")
    return df_sel


def appliquer_log(valeurs):
    """Log naturel sur valeurs strictement positives; NaN sinon."""
    arr = np.asarray(valeurs, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(arr > 0, np.log(arr), np.nan)
    return out


def diagnostics_residus(residus, exog_design=None):
    """Normalite (Shapiro, D'Agostino) + homoscedasticite (Breusch-Pagan)."""
    diag = {}
    r = np.asarray(residus, dtype=float)
    r = r[np.isfinite(r)]
    try:
        diag['shapiro_p'] = float(stats.shapiro(r).pvalue) if len(r) >= 3 else np.nan
    except Exception:
        diag['shapiro_p'] = np.nan
    try:
        diag['dagostino_p'] = float(stats.normaltest(r).pvalue) if len(r) >= 8 else np.nan
    except Exception:
        diag['dagostino_p'] = np.nan
    try:
        if exog_design is not None and len(r) == exog_design.shape[0]:
            bp = het_breuschpagan(residus, exog_design)
            diag['breuschpagan_p'] = float(bp[1])
        else:
            diag['breuschpagan_p'] = np.nan
    except Exception:
        diag['breuschpagan_p'] = np.nan
    return diag


# ==============================================================================
# REGRESSIONS
# ==============================================================================

def _resultat_vide(n=0, motif='insuffisant'):
    return {
        'N': n, 'beta_expo': np.nan, 'SE': np.nan, 't': np.nan,
        'p_brut': np.nan, 'IC95_low': np.nan, 'IC95_high': np.nan,
        'R2': np.nan, 'R2_adj': np.nan,
        'shapiro_p': np.nan, 'dagostino_p': np.nan, 'breuschpagan_p': np.nan,
        'statut': motif,
    }


def ajuster_regression_par_trajet(df, col_perf, col_expo, log_expo,
                                  transformation_perf):
    """
    Regression : Perf_X ~ Expo_X + C(Mode_transport)
    Retour : dict de stats.
    """
    colonnes = [col_perf, col_expo, 'Mode_transport']
    if 'Ordre' in df.columns:
        colonnes.append('Ordre')
    sub = df[colonnes].copy()
    sub.columns = ['Perf', 'Expo', 'Mode_transport'] + (['Ordre'] if len(colonnes) == 4 else [])
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if transformation_perf == 'log':
        sub['Perf'] = appliquer_log(sub['Perf'])
    if log_expo:
        sub['Expo'] = appliquer_log(sub['Expo'])
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5 or sub['Mode_transport'].nunique() < 2:
        return _resultat_vide(len(sub), 'N insuffisant')
    try:
        modele = ols(f'Perf ~ Expo + {_covariables(sub)}', data=sub).fit()
        beta = float(modele.params.get('Expo', np.nan))
        se = float(modele.bse.get('Expo', np.nan))
        tval = float(modele.tvalues.get('Expo', np.nan))
        pval = float(modele.pvalues.get('Expo', np.nan))
        ci = modele.conf_int().loc['Expo'] if 'Expo' in modele.conf_int().index else (np.nan, np.nan)
        diag = diagnostics_residus(modele.resid, modele.model.exog)
        return {
            'N': int(len(sub)),
            'beta_expo': beta, 'SE': se, 't': tval, 'p_brut': pval,
            'IC95_low': float(ci[0]), 'IC95_high': float(ci[1]),
            'R2': float(modele.rsquared), 'R2_adj': float(modele.rsquared_adj),
            **diag, 'statut': 'OK',
        }
    except Exception as e:
        res = _resultat_vide(len(sub), f'erreur:{type(e).__name__}')
        return res


def ajuster_regression_differences(df, col_perf_A, col_perf_C,
                                   col_expo_A, col_expo_C,
                                   log_expo, transformation_perf):
    """
    Regression : (Perf_C - Perf_A) ~ (Expo_C - Expo_A) + C(Mode_transport)
    Log sur perf (RT) et/ou exposition applique AVANT differenciation.
    """
    colonnes = [col_perf_A, col_perf_C, col_expo_A, col_expo_C, 'Mode_transport']
    if 'Ordre' in df.columns:
        colonnes.append('Ordre')
    sub = df[colonnes].copy()
    sub.columns = (['PerfA', 'PerfC', 'ExpoA', 'ExpoC', 'Mode_transport']
                   + (['Ordre'] if len(colonnes) == 6 else []))
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if transformation_perf == 'log':
        sub['PerfA'] = appliquer_log(sub['PerfA'])
        sub['PerfC'] = appliquer_log(sub['PerfC'])
    if log_expo:
        sub['ExpoA'] = appliquer_log(sub['ExpoA'])
        sub['ExpoC'] = appliquer_log(sub['ExpoC'])
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5 or sub['Mode_transport'].nunique() < 2:
        return _resultat_vide(len(sub), 'N insuffisant')
    sub['DeltaPerf'] = sub['PerfC'] - sub['PerfA']
    sub['DeltaExpo'] = sub['ExpoC'] - sub['ExpoA']
    try:
        modele = ols(f'DeltaPerf ~ DeltaExpo + {_covariables(sub)}', data=sub).fit()
        beta = float(modele.params.get('DeltaExpo', np.nan))
        se = float(modele.bse.get('DeltaExpo', np.nan))
        tval = float(modele.tvalues.get('DeltaExpo', np.nan))
        pval = float(modele.pvalues.get('DeltaExpo', np.nan))
        ci = modele.conf_int().loc['DeltaExpo'] if 'DeltaExpo' in modele.conf_int().index else (np.nan, np.nan)
        diag = diagnostics_residus(modele.resid, modele.model.exog)
        return {
            'N': int(len(sub)),
            'beta_expo': beta, 'SE': se, 't': tval, 'p_brut': pval,
            'IC95_low': float(ci[0]), 'IC95_high': float(ci[1]),
            'R2': float(modele.rsquared), 'R2_adj': float(modele.rsquared_adj),
            **diag, 'statut': 'OK',
        }
    except Exception as e:
        return _resultat_vide(len(sub), f'erreur:{type(e).__name__}')


# ==============================================================================
# PIPELINE
# ==============================================================================

def appliquer_fdr_par_domaine(df_combine, col_p='p_brut',
                              col_out_p='p_FDR_domaine',
                              col_out_sig='significatif_FDR_domaine'):
    """
    FDR BH appliquee au sein de chaque domaine (EEG / Comportement),
    en poolant les indicateurs des deux taches (NBack + ANT).
    """
    df_combine[col_out_p] = np.nan
    df_combine[col_out_sig] = False
    if 'domaine' not in df_combine.columns:
        return df_combine
    for _domaine, grp in df_combine.groupby('domaine'):
        mask = grp[col_p].notna()
        p_brut = grp.loc[mask, col_p].values
        if len(p_brut) == 0:
            continue
        _, p_corr, _, _ = multipletests(p_brut, alpha=ALPHA, method=METHODE_CORRECTION)
        p_fdr = np.full(len(grp), np.nan)
        p_fdr[mask.values] = p_corr
        df_combine.loc[grp.index, col_out_p] = p_fdr
        df_combine.loc[grp.index, col_out_sig] = df_combine.loc[grp.index, col_out_p] < ALPHA
    return df_combine


def appliquer_fdr(df_resultats, col_p='p_brut', col_out='p_FDR'):
    """FDR Benjamini-Hochberg sur une colonne de p-valeurs."""
    mask = df_resultats[col_p].notna()
    p_brut = df_resultats.loc[mask, col_p].values
    p_fdr = np.full(len(df_resultats), np.nan)
    if len(p_brut) > 0:
        _, p_corr, _, _ = multipletests(p_brut, alpha=ALPHA, method=METHODE_CORRECTION)
        p_fdr_masked = np.full(len(df_resultats), np.nan)
        p_fdr_masked[mask.values] = p_corr
        df_resultats[col_out] = p_fdr_masked
    else:
        df_resultats[col_out] = np.nan
    df_resultats['significatif_FDR'] = df_resultats[col_out] < ALPHA
    return df_resultats


def traiter_une_tache(df, indicateurs_tache, col_expo_A, col_expo_C, log_expo):
    """
    Pour une tache (N-Back ou ANT) et un polluant/transformation donnes :
    - Ajuste les 3 regressions par indicateur
    - Renvoie 3 DataFrames (trajet_A, trajet_C, differences) avec FDR applique par volet.
    """
    lignes_A, lignes_C, lignes_D = [], [], []
    for ind in indicateurs_tache:
        base = {'indicateur': ind['nom'], 'domaine': ind['domaine'],
                'tache': ind['tache'], 'transfo_perf': ind['transformation']}
        # Volet A
        res_A = ajuster_regression_par_trajet(
            df, ind['col_away'], col_expo_A, log_expo, ind['transformation'])
        lignes_A.append({**base, **res_A})
        # Volet C
        res_C = ajuster_regression_par_trajet(
            df, ind['col_close'], col_expo_C, log_expo, ind['transformation'])
        lignes_C.append({**base, **res_C})
        # Volet Delta
        res_D = ajuster_regression_differences(
            df, ind['col_away'], ind['col_close'],
            col_expo_A, col_expo_C, log_expo, ind['transformation'])
        lignes_D.append({**base, **res_D})
    df_A = appliquer_fdr(pd.DataFrame(lignes_A))
    df_C = appliquer_fdr(pd.DataFrame(lignes_C))
    df_D = appliquer_fdr(pd.DataFrame(lignes_D))
    return df_A, df_C, df_D


def ecrire_resultats_tache(path_xlsx, df_A, df_C, df_D):
    with pd.ExcelWriter(path_xlsx, engine='openpyxl') as writer:
        df_A.to_excel(writer, sheet_name='Trajet_A', index=False)
        df_C.to_excel(writer, sheet_name='Trajet_C', index=False)
        df_D.to_excel(writer, sheet_name='Differences', index=False)


def ecrire_rapport_diagnostics(path_txt, df_A, df_C, df_D, polluant,
                                transformation, tache, col_expo_A, col_expo_C,
                                df_source, log_expo):
    with open(path_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RAPPORT DIAGNOSTICS - {polluant} / {transformation} / {tache}\n")
        f.write("=" * 80 + "\n\n")
        expo_A = df_source[col_expo_A].dropna()
        expo_C = df_source[col_expo_C].dropna()
        f.write("Distribution de l'exposition :\n")
        for label, arr in [('A (Away)', expo_A), ('C (Close)', expo_C)]:
            if len(arr) == 0:
                continue
            sk = float(stats.skew(arr))
            sk_log = float(stats.skew(np.log(arr[arr > 0]))) if (arr > 0).any() else np.nan
            f.write(f"  {label}: N={len(arr)} min={arr.min():.4g} "
                    f"median={arr.median():.4g} max={arr.max():.4g} "
                    f"skew={sk:.2f} skew(log)={sk_log:.2f}\n")
        f.write(f"  Transformation appliquee : {'log(expo)' if log_expo else 'brute'}\n\n")
        for label, df_res in [('Trajet A', df_A), ('Trajet C', df_C),
                              ('Differences', df_D)]:
            n_sig = int(df_res['significatif_FDR'].sum())
            f.write(f"--- {label} ---\n")
            f.write(f"  Indicateurs testes : {len(df_res)}, significatifs apres FDR : {n_sig}\n")
            if n_sig > 0:
                sig = df_res[df_res['significatif_FDR']]
                for _, row in sig.iterrows():
                    f.write(f"    * {row['indicateur']} : beta={row['beta_expo']:.4g} "
                            f"p_FDR={row['p_FDR']:.4f} R2={row['R2']:.3f}\n")
            # Warnings diagnostics
            warn_norm = df_res[(df_res['shapiro_p'] < 0.05) & df_res['shapiro_p'].notna()]
            warn_hetero = df_res[(df_res['breuschpagan_p'] < 0.05)
                                 & df_res['breuschpagan_p'].notna()]
            if len(warn_norm):
                f.write(f"  /!\\ Residus non-normaux (Shapiro p<0.05) : "
                        f"{warn_norm['indicateur'].tolist()}\n")
            if len(warn_hetero):
                f.write(f"  /!\\ Heteroscedasticite (BP p<0.05) : "
                        f"{warn_hetero['indicateur'].tolist()}\n")
            f.write("\n")


def ecrire_resultats_par_domaine(dir_transfo, df_A_nb, df_C_nb, df_D_nb,
                                 df_A_ant, df_C_ant, df_D_ant):
    """
    Pool NBack + ANT par volet, applique FDR par domaine (EEG / Comportement),
    et ecrit par_domaine/Resultats_EEG.xlsx + Resultats_Comportement.xlsx
    (chacun avec 3 feuilles : Trajet_A / Trajet_C / Differences).
    """
    dir_domaine = os.path.join(dir_transfo, 'par_domaine')
    Path(dir_domaine).mkdir(parents=True, exist_ok=True)

    df_A = pd.concat([df_A_nb, df_A_ant], ignore_index=True)
    df_C = pd.concat([df_C_nb, df_C_ant], ignore_index=True)
    df_D = pd.concat([df_D_nb, df_D_ant], ignore_index=True)

    df_A = appliquer_fdr_par_domaine(df_A)
    df_C = appliquer_fdr_par_domaine(df_C)
    df_D = appliquer_fdr_par_domaine(df_D)

    for domaine, suffixe in [('EEG', 'EEG'), ('Comportement', 'Comportement')]:
        sub_A = df_A[df_A['domaine'] == domaine].copy()
        sub_C = df_C[df_C['domaine'] == domaine].copy()
        sub_D = df_D[df_D['domaine'] == domaine].copy()
        if len(sub_A) == 0 and len(sub_C) == 0 and len(sub_D) == 0:
            continue
        fichier = os.path.join(dir_domaine, f'Resultats_{suffixe}.xlsx')
        with pd.ExcelWriter(fichier, engine='openpyxl') as writer:
            sub_A.to_excel(writer, sheet_name='Trajet_A', index=False)
            sub_C.to_excel(writer, sheet_name='Trajet_C', index=False)
            sub_D.to_excel(writer, sheet_name='Differences', index=False)
        n_sigA = int(sub_A['significatif_FDR_domaine'].sum()) if len(sub_A) else 0
        n_sigC = int(sub_C['significatif_FDR_domaine'].sum()) if len(sub_C) else 0
        n_sigD = int(sub_D['significatif_FDR_domaine'].sum()) if len(sub_D) else 0
        print(f"    par_domaine/{suffixe}: sig_A={n_sigA} sig_C={n_sigC} sig_D={n_sigD}")

    return df_A, df_C, df_D


def construire_synthese_polluant(resultats_polluant):
    """
    resultats_polluant : dict {(transfo, tache_ou_par_domaine): (df_A, df_C, df_D)}
    Retourne un DataFrame recapitulatif.
    - Pour les taches NBack/ANT : comptage sur significatif_FDR (par tache).
    - Pour 'par_domaine' : comptage sur significatif_FDR_domaine + detail EEG/Comport.
    """
    lignes = []
    for (transfo, tache), (df_A, df_C, df_D) in resultats_polluant.items():
        if tache == 'par_domaine':
            col_sig = 'significatif_FDR_domaine'
            col_ind = 'indicateur'
            ligne = {
                'transformation': transfo,
                'tache': 'par_domaine',
                'N_tests': len(df_A),
                'N_sig_Trajet_A': int(df_A[col_sig].sum()),
                'N_sig_Trajet_C': int(df_C[col_sig].sum()),
                'N_sig_Differences': int(df_D[col_sig].sum()),
                'Sig_Trajet_A': ', '.join(df_A.loc[df_A[col_sig], col_ind].tolist()),
                'Sig_Trajet_C': ', '.join(df_C.loc[df_C[col_sig], col_ind].tolist()),
                'Sig_Differences': ', '.join(df_D.loc[df_D[col_sig], col_ind].tolist()),
            }
            for dom in ['EEG', 'Comportement']:
                for lab, d in [('A', df_A), ('C', df_C), ('D', df_D)]:
                    sub = d[d['domaine'] == dom]
                    ligne[f'N_sig_{dom}_Trajet_{lab}' if lab != 'D'
                          else f'N_sig_{dom}_Differences'] = int(sub[col_sig].sum())
            lignes.append(ligne)
        else:
            lignes.append({
                'transformation': transfo,
                'tache': tache,
                'N_tests': len(df_A),
                'N_sig_Trajet_A': int(df_A['significatif_FDR'].sum()),
                'N_sig_Trajet_C': int(df_C['significatif_FDR'].sum()),
                'N_sig_Differences': int(df_D['significatif_FDR'].sum()),
                'Sig_Trajet_A': ', '.join(df_A.loc[df_A['significatif_FDR'], 'indicateur'].tolist()),
                'Sig_Trajet_C': ', '.join(df_C.loc[df_C['significatif_FDR'], 'indicateur'].tolist()),
                'Sig_Differences': ', '.join(df_D.loc[df_D['significatif_FDR'], 'indicateur'].tolist()),
            })
    return pd.DataFrame(lignes)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 80}\nANALYSE DE REGRESSION CONTINUE\n{'=' * 80}\n")
    df = charger_donnees(DATA_FILE)
    df_sel = filtrer_selection(df, SELECTION_4)
    ordre = charger_ordre()
    if ordre is not None:
        avant = len(df_sel)
        df_sel = df_sel.merge(ordre, on='Participant', how='left')
        manquants = df_sel['Ordre'].isna().sum()
        print(f"  Ordre fusionne : {avant - manquants}/{avant} participants "
              f"({manquants} sans ordre connu)")

    indicateurs_nback = generer_indicateurs_nback()
    indicateurs_ant = generer_indicateurs_ant()
    print(f"\nIndicateurs : {len(indicateurs_nback)} N-Back + {len(indicateurs_ant)} ANT "
          f"= {len(indicateurs_nback) + len(indicateurs_ant)}\n")

    synth_globale = []

    for polluant in POLLUANTS:
        print(f"\n{'=' * 80}\nPOLLUANT : {polluant['nom']}\n{'=' * 80}")
        dossier_polluant = os.path.join(OUTPUT_DIR, polluant['nom'])
        Path(dossier_polluant).mkdir(parents=True, exist_ok=True)
        resultats_pol = {}

        for transfo in TRANSFORMATIONS:
            log_expo = (transfo == 'log_expo')
            print(f"\n[{polluant['nom']} / {transfo}]")
            dossier_transfo = os.path.join(dossier_polluant, transfo)
            Path(dossier_transfo).mkdir(parents=True, exist_ok=True)

            for nom_tache, indicateurs in [('NBack', indicateurs_nback),
                                            ('ANT', indicateurs_ant)]:
                dossier_tache = os.path.join(dossier_transfo, nom_tache)
                dossier_diag = os.path.join(dossier_transfo, 'Diagnostics')
                Path(dossier_tache).mkdir(parents=True, exist_ok=True)
                Path(dossier_diag).mkdir(parents=True, exist_ok=True)

                df_A, df_C, df_D = traiter_une_tache(
                    df_sel, indicateurs, polluant['col_A'], polluant['col_C'], log_expo)

                path_xlsx = os.path.join(dossier_tache, f'Resultats_{nom_tache}.xlsx')
                ecrire_resultats_tache(path_xlsx, df_A, df_C, df_D)

                path_diag = os.path.join(dossier_diag, f'Rapport_{nom_tache}.txt')
                ecrire_rapport_diagnostics(path_diag, df_A, df_C, df_D,
                                           polluant['nom'], transfo, nom_tache,
                                           polluant['col_A'], polluant['col_C'],
                                           df_sel, log_expo)

                resultats_pol[(transfo, nom_tache)] = (df_A, df_C, df_D)
                print(f"  {nom_tache} : N_ind={len(df_A)}  "
                      f"sig_A={int(df_A['significatif_FDR'].sum())}  "
                      f"sig_C={int(df_C['significatif_FDR'].sum())}  "
                      f"sig_D={int(df_D['significatif_FDR'].sum())}")

            # FDR par DOMAINE (EEG / Comportement) - pool NBack + ANT
            df_A_nb, df_C_nb, df_D_nb = resultats_pol[(transfo, 'NBack')]
            df_A_ant, df_C_ant, df_D_ant = resultats_pol[(transfo, 'ANT')]
            print(f"  [{polluant['nom']} / {transfo}] FDR par domaine (pool NBack+ANT) :")
            df_A_pool, df_C_pool, df_D_pool = ecrire_resultats_par_domaine(
                dossier_transfo, df_A_nb, df_C_nb, df_D_nb,
                df_A_ant, df_C_ant, df_D_ant)
            resultats_pol[(transfo, 'par_domaine')] = (df_A_pool, df_C_pool, df_D_pool)

        df_synth = construire_synthese_polluant(resultats_pol)
        path_synth = os.path.join(dossier_polluant, 'Synthese.xlsx')
        df_synth.to_excel(path_synth, index=False)
        df_synth.insert(0, 'polluant', polluant['nom'])
        synth_globale.append(df_synth)

    df_globale = pd.concat(synth_globale, ignore_index=True)
    df_globale.to_excel(os.path.join(OUTPUT_DIR, 'Synthese_globale.xlsx'), index=False)
    print(f"\n{'=' * 80}\nTERMINE - sortie : {OUTPUT_DIR}\n{'=' * 80}\n")


if __name__ == '__main__':
    main()
