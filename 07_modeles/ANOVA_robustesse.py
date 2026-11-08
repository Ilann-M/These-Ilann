# === CORRECTION DU F DE L'EFFET PRINCIPAL INTRA — 2026-09-09 ===
# SS_Trajet etait calcule sur l'echelle de la variable et SS_TxS sur celle des scores de
# difference : le F de l'effet principal du Trajet etait divise par deux. Chemins passes
# en local (le partage n'est plus la source d'execution).
# Sauvegarde de la version anterieure dans Data\Codes\_Archives\.
"""
════════════════════════════════════════════════════════════════════════════════
ANALYSE DE ROBUSTESSE - ANOVA MIXTE

Objectif : Vérifier la concordance des résultats LMM avec une approche
ANOVA mixte classique (split-plot) sur les mêmes indicateurs.

Design : 2×2 mixte
  - Mode de transport (Vélo vs VAE) : facteur inter-sujets
  - Trajet (Away vs Close) : facteur intra-sujets

Indicateurs : identiques à l'Approche 3 (FDR 2 niveaux)
  - Principale (8) : N-Back (RT, d', N1, P3 × 1B/2B)
  - Secondaire (17) : ANT (RT × 5 réseaux + EEG)

Correction : FDR Benjamini-Hochberg par niveau (identique au LMM)

Sélection : Selection_4 (N=51)
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

from statsmodels.stats.multitest import multipletests

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_FILE = D(r'Data\Table_combine\Tab_sujet_20260205.xlsx')
OUTPUT_DIR = S(r'Resultats_LMM_intention\26-10-07_ANOVA_codage_somme_sans134')

VERSION_DONNEES = 'filtre'

SELECTION_4 = [101, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
               201, 202, 203, 204, 205, 206, 209, 210, 211, 212, 215, 216,
               121, 122, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 139,
               221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]

ALPHA = 0.05
METHODE_CORRECTION = 'fdr_bh'


# ==============================================================================
# DÉFINITION DES INDICATEURS (identique à l'Approche 3)
# ==============================================================================

def generer_indicateurs_principaux():
    """Analyse principale : 8 indicateurs N-Back"""
    indicateurs = []
    for tache in ['1B', '2B']:
        indicateurs.append({
            'nom': f'RT_moyen_{tache}', 'tache': tache, 'domaine': 'Comportement',
            'col_away': f'RT_moyen_{VERSION_DONNEES}_A_{tache}',
            'col_close': f'RT_moyen_{VERSION_DONNEES}_C_{tache}',
            'transformation': 'log'
        })
        indicateurs.append({
            'nom': f'd_prime_{tache}', 'tache': tache, 'domaine': 'Comportement',
            'col_away': f'd_prime_{VERSION_DONNEES}_A_{tache}',
            'col_close': f'd_prime_{VERSION_DONNEES}_C_{tache}',
            'transformation': None
        })
        for comp in ['N1', 'P3']:
            indicateurs.append({
                'nom': f'{comp}_Moy_Valide_{tache}', 'tache': tache, 'domaine': 'EEG',
                'col_away': f'{comp}_Moy_Valide_A_{tache}',
                'col_close': f'{comp}_Moy_Valide_C_{tache}',
                'transformation': None
            })
    return indicateurs


def generer_indicateurs_secondaires():
    """Analyse secondaire : 17 indicateurs ANT"""
    indicateurs = []

    reseaux_rt = ['Alerte_Standard', 'Alerte_Auditive', 'Alerte_Visuelle',
                  'Orientation', 'Controle_Executif']
    for reseau in reseaux_rt:
        indicateurs.append({
            'nom': f'RT_{reseau}_ANT', 'tache': 'ANT', 'domaine': 'Comportement',
            'col_away': f'RT_{reseau}_A_ANT', 'col_close': f'RT_{reseau}_C_ANT',
            'transformation': 'log'
        })

    reseaux_eeg = {
        'Alerte_Standard': 'Alerte_Standard', 'Alerte_Auditive': 'Alerte_Auditive',
        'Alerte_Visuelle': 'Alerte_Visuelle', 'Orientation': 'Orientation',
        'Controle_Executif': 'Controle'
    }
    for reseau_label, cond in reseaux_eeg.items():
        for comp in ['N1', 'P3']:
            indicateurs.append({
                'nom': f'{comp}_Moy_fleche_{reseau_label}_ANT', 'tache': 'ANT', 'domaine': 'EEG',
                'col_away': f'{comp}_{cond}_Moy_fleche_A_ANT',
                'col_close': f'{comp}_{cond}_Moy_fleche_C_ANT',
                'transformation': None
            })

    for comp in ['N1', 'P2']:
        indicateurs.append({
            'nom': f'{comp}_Moy_sonore_Alerte_Standard_ANT', 'tache': 'ANT', 'domaine': 'EEG',
            'col_away': f'{comp}_Alerte_Standard_Moy_sonore_A_ANT',
            'col_close': f'{comp}_Alerte_Standard_Moy_sonore_C_ANT',
            'transformation': None
        })

    indicateurs.append({
        'nom': 'N1_Moy_indice_Alerte_Visuelle_ANT', 'tache': 'ANT', 'domaine': 'EEG',
        'col_away': 'N1_Alerte_Visuelle_Moy_indice_A_ANT',
        'col_close': 'N1_Alerte_Visuelle_Moy_indice_C_ANT',
        'transformation': None
    })

    return indicateurs


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def charger_et_filtrer(fichier, selection):
    df = pd.read_excel(fichier)
    # === 07/10/2026 (comme LMM_intention.py) : session 2-back du trajet éloigné du participant 134
    # retirée des indicateurs comportementaux du 2-back (aucun appui enregistré) ; ERP conservés.
    df.loc[df['Participant'] == 134, [c for c in df.columns if c.startswith('d_prime_') and c.endswith('_A_2B')]] = float('nan')
    return df[df['Participant'].isin(selection)].copy()


def appliquer_transformation(valeurs, transformation):
    if transformation is None or transformation == 'none':
        return valeurs
    v = valeurs.copy()
    if transformation == 'log':
        if (v <= 0).any():
            min_pos = v[v > 0].min() if (v > 0).any() else 0.001
            v = v.clip(lower=min_pos * 0.001)
        return np.log(v)
    return v


# ==============================================================================
# ANOVA MIXTE 2×2
# ==============================================================================

def anova_mixte(df, col_away, col_close, transformation=None):
    """
    ANOVA mixte 2×2 calculée manuellement.

    Design :
      - Facteur A (Mode) : inter-sujets (Bike vs e-Bike)
      - Facteur B (Trajet) : intra-sujets (Away vs Close)

    Méthode : décomposition des sommes de carrés selon le modèle split-plot.
    """

    # Paires complètes
    mask = df[col_away].isna() | df[col_close].isna()
    df_clean = df[~mask].copy()

    if len(df_clean) < 10:
        return None

    # Appliquer transformation
    away = appliquer_transformation(df_clean[col_away], transformation).values
    close = appliquer_transformation(df_clean[col_close], transformation).values
    mode = df_clean['Mode_transport'].values
    participants = df_clean['Participant'].values

    # Groupes
    bike_mask = mode == 'Bike'
    ebike_mask = mode == 'e-Bike'

    n_bike = bike_mask.sum()
    n_ebike = ebike_mask.sum()
    N = n_bike + n_ebike

    if n_bike < 5 or n_ebike < 5:
        return None

    # 4 cellules
    bike_away = away[bike_mask]
    bike_close = close[bike_mask]
    ebike_away = away[ebike_mask]
    ebike_close = close[ebike_mask]

    # Moyennes des cellules
    m_ba = np.mean(bike_away)
    m_bc = np.mean(bike_close)
    m_ea = np.mean(ebike_away)
    m_ec = np.mean(ebike_close)

    # Moyennes marginales
    m_bike = (m_ba + m_bc) / 2
    m_ebike = (m_ea + m_ec) / 2
    m_away = (m_ba + m_ea) / 2
    m_close = (m_bc + m_ec) / 2
    m_grand = (m_ba + m_bc + m_ea + m_ec) / 4

    # ── Effet inter-sujets : Mode ──
    # Moyenne par sujet (collapsed across Trajet)
    subj_means = (away + close) / 2

    # SS_Mode (between-groups on subject means)
    SS_Mode = n_bike * (np.mean(subj_means[bike_mask]) - np.mean(subj_means)) ** 2 + \
              n_ebike * (np.mean(subj_means[ebike_mask]) - np.mean(subj_means)) ** 2

    # SS_Subjects_within_Mode (error term for between-subjects)
    SS_subj_bike = np.sum((subj_means[bike_mask] - np.mean(subj_means[bike_mask])) ** 2)
    SS_subj_ebike = np.sum((subj_means[ebike_mask] - np.mean(subj_means[ebike_mask])) ** 2)
    SS_subj_within = SS_subj_bike + SS_subj_ebike

    df_Mode = 1
    df_subj_within = N - 2

    MS_Mode = SS_Mode / df_Mode
    MS_subj_within = SS_subj_within / df_subj_within

    F_Mode = MS_Mode / MS_subj_within if MS_subj_within > 0 else np.nan
    p_Mode = 1 - stats.f.cdf(F_Mode, df_Mode, df_subj_within) if not np.isnan(F_Mode) else np.nan

    # ── Effet intra-sujets : Trajet ──
    diff = close - away  # score de différence

    # SS_Trajet
    # CORRECTION 2026-09-09 : SS_Trajet doit etre sur la MEME echelle que son terme
    # d'erreur SS_TxS, calcule plus bas sur les SCORES DE DIFFERENCE. La division par 2
    # (echelle de la variable) divisait F par 2 : 3,16 au lieu de 6,32 pour la N1 1-back.
    # Verifie contre pingouin.mixed_anova. Voir Retours\PLAN_REVISION_ENVADV.md, section 1.3.
    mean_diff = np.mean(diff)
    SS_Trajet = N * (mean_diff ** 2)

    # SS_Trajet×Sujets (error for within-subjects main effect)
    diff_bike = diff[bike_mask]
    diff_ebike = diff[ebike_mask]

    # Interaction Trajet × Mode
    mean_diff_bike = np.mean(diff_bike)
    mean_diff_ebike = np.mean(diff_ebike)

    SS_Interaction = (n_bike * n_ebike / N) * (mean_diff_bike - mean_diff_ebike) ** 2

    # Error: Trajet × Subjects(Mode)
    SS_TxS_bike = np.sum((diff_bike - mean_diff_bike) ** 2)
    SS_TxS_ebike = np.sum((diff_ebike - mean_diff_ebike) ** 2)
    SS_TxS = SS_TxS_bike + SS_TxS_ebike

    df_Trajet = 1
    df_Interaction = 1
    df_TxS = N - 2

    MS_Trajet = SS_Trajet / df_Trajet
    MS_Interaction = SS_Interaction / df_Interaction
    MS_TxS = SS_TxS / df_TxS

    F_Trajet = MS_Trajet / MS_TxS if MS_TxS > 0 else np.nan
    F_Interaction = MS_Interaction / MS_TxS if MS_TxS > 0 else np.nan

    p_Trajet = 1 - stats.f.cdf(F_Trajet, df_Trajet, df_TxS) if not np.isnan(F_Trajet) else np.nan
    p_Interaction = 1 - stats.f.cdf(F_Interaction, df_Interaction, df_TxS) if not np.isnan(F_Interaction) else np.nan

    # ── Tailles d'effet (eta² partiel) ──
    eta2_Mode = SS_Mode / (SS_Mode + SS_subj_within) if (SS_Mode + SS_subj_within) > 0 else np.nan
    eta2_Trajet = SS_Trajet / (SS_Trajet + SS_TxS) if (SS_Trajet + SS_TxS) > 0 else np.nan
    eta2_Interaction = SS_Interaction / (SS_Interaction + SS_TxS) if (SS_Interaction + SS_TxS) > 0 else np.nan

    # ── Cohen's d ──
    pooled_sd = np.sqrt(MS_subj_within)
    d_Mode = (m_bike - m_ebike) / pooled_sd if pooled_sd > 0 else np.nan

    sd_diff = np.std(diff, ddof=1)
    d_Trajet = mean_diff / sd_diff if sd_diff > 0 else np.nan

    d_Interaction = (mean_diff_bike - mean_diff_ebike) / sd_diff if sd_diff > 0 else np.nan

    # ── Vérification de sphéricité (pas nécessaire avec 2 niveaux) ──
    # Avec seulement 2 niveaux intra-sujets, la sphéricité est automatiquement satisfaite.

    # ── Test de Levene (homogénéité inter-groupes) ──
    try:
        _, p_levene = stats.levene(subj_means[bike_mask], subj_means[ebike_mask])
    except:
        p_levene = np.nan

    # ── Post-hoc : t-tests appariés par Mode ──
    post_hoc = {}

    # Vélo : Close vs Away
    t_bike, p_bike = stats.ttest_rel(bike_close, bike_away)
    d_ph_bike = np.mean(bike_close - bike_away) / np.std(bike_close - bike_away, ddof=1)
    post_hoc['Bike'] = {
        'p': p_bike, 'p_corr': min(p_bike * 2, 1.0),
        't': t_bike, 'd': d_ph_bike,
        'direction': 'Close > Away' if np.mean(bike_close) > np.mean(bike_away) else 'Away > Close',
        'sig': min(p_bike * 2, 1.0) < ALPHA
    }

    # VAE : Close vs Away
    t_ebike, p_ebike = stats.ttest_rel(ebike_close, ebike_away)
    d_ph_ebike = np.mean(ebike_close - ebike_away) / np.std(ebike_close - ebike_away, ddof=1)
    post_hoc['e-Bike'] = {
        'p': p_ebike, 'p_corr': min(p_ebike * 2, 1.0),
        't': t_ebike, 'd': d_ph_ebike,
        'direction': 'Close > Away' if np.mean(ebike_close) > np.mean(ebike_away) else 'Away > Close',
        'sig': min(p_ebike * 2, 1.0) < ALPHA
    }

    # ── Moyennes descriptives ──
    descriptives = {
        'Bike_Away_M': m_ba, 'Bike_Away_SD': np.std(bike_away, ddof=1),
        'Bike_Close_M': m_bc, 'Bike_Close_SD': np.std(bike_close, ddof=1),
        'eBike_Away_M': m_ea, 'eBike_Away_SD': np.std(ebike_away, ddof=1),
        'eBike_Close_M': m_ec, 'eBike_Close_SD': np.std(ebike_close, ddof=1),
    }

    return {
        'N': N, 'n_bike': n_bike, 'n_ebike': n_ebike,
        'n_nan': mask.sum(), 'pct_perte': mask.sum() / len(df) * 100,

        'F_Mode': F_Mode, 'p_Mode': p_Mode, 'df_Mode': (df_Mode, df_subj_within),
        'eta2_Mode': eta2_Mode, 'd_Mode': d_Mode,

        'F_Trajet': F_Trajet, 'p_Trajet': p_Trajet, 'df_Trajet': (df_Trajet, df_TxS),
        'eta2_Trajet': eta2_Trajet, 'd_Trajet': d_Trajet,

        'F_Interaction': F_Interaction, 'p_Interaction': p_Interaction,
        'df_Interaction': (df_Interaction, df_TxS),
        'eta2_Interaction': eta2_Interaction, 'd_Interaction': d_Interaction,

        'p_levene': p_levene,
        'post_hoc': post_hoc,
        'descriptives': descriptives,
    }


# ==============================================================================
# CORRECTION FDR
# ==============================================================================

def appliquer_fdr(df_res, label):
    """Applique FDR sur un dataframe de résultats"""
    n = len(df_res)
    print(f"\n  Correction FDR - {label} ({n} tests)...")

    for effet in ['Mode', 'Trajet', 'Interaction']:
        col_p = f'p_{effet}'
        col_fdr = f'p_{effet}_FDR'
        col_sig = f'Sig_{effet}_FDR'

        p_vals = df_res[col_p].values
        mask_valid = ~np.isnan(p_vals)
        p_fdr = np.full_like(p_vals, np.nan)

        if mask_valid.sum() > 0:
            _, p_corr, _, _ = multipletests(p_vals[mask_valid], alpha=ALPHA, method=METHODE_CORRECTION)
            p_fdr[mask_valid] = p_corr

        df_res[col_fdr] = p_fdr
        df_res[col_sig] = df_res[col_fdr].apply(
            lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns'
            if not np.isnan(x) else 'N/A'
        )

        n_sig = df_res[col_sig].isin(['*', '**', '***']).sum()
        print(f"    {effet}: {n_sig}/{mask_valid.sum()} significatifs")

    return df_res


# ==============================================================================
# RAPPORT
# ==============================================================================

def generer_rapport(df_princ, df_sec, output_dir):
    """Rapport comparatif ANOVA vs LMM"""
    path = os.path.join(output_dir, 'Rapport_ANOVA_Robustesse.txt')

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ANALYSE DE ROBUSTESSE - ANOVA MIXTE 2×2\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Design : Mode (Vélo vs VAE, inter) × Trajet (Away vs Close, intra)\n")
        f.write("Correction : FDR Benjamini-Hochberg par niveau\n")
        f.write("Post-hoc : t-tests appariés par Mode (Bonferroni ×2)\n")
        f.write("Note : Avec 2 niveaux intra-sujets, la sphéricité est\n")
        f.write("automatiquement satisfaite (pas de correction nécessaire).\n\n")

        for label, df_res in [('PRINCIPALE (N-Back, 8 tests)', df_princ),
                               ('SECONDAIRE (ANT)', df_sec)]:
            f.write(f"\n{'═' * 80}\n")
            f.write(f"  {label}\n")
            f.write(f"{'═' * 80}\n")

            for effet in ['Mode', 'Trajet', 'Interaction']:
                col_p = f'p_{effet}'
                col_fdr = f'p_{effet}_FDR'
                col_sig = f'Sig_{effet}_FDR'
                col_F = f'F_{effet}'
                col_d = f'd_{effet}'
                col_eta = f'eta2_{effet}'
                col_df = f'df_{effet}'

                f.write(f"\n  ── {effet} ──\n")

                sub = df_res.sort_values(col_p)
                for _, row in sub.iterrows():
                    sig = row[col_sig] if row[col_sig] in ['*', '**', '***'] else '  '
                    df_str = row.get(col_df, '')
                    f.write(f"    {sig:>3} {row['Indicateur']:<45} "
                            f"F={row[col_F]:.2f}  p={row[col_p]:.4f}  "
                            f"p_FDR={row[col_fdr]:.4f}  η²p={row[col_eta]:.4f}  "
                            f"d={row[col_d]:.3f}\n")

            # Post-hoc pour interactions significatives/tendances
            f.write(f"\n  ── Post-hoc (Interactions p_FDR < 0.10) ──\n")
            tendances = df_res[df_res['p_Interaction_FDR'] < 0.10]
            if len(tendances) == 0:
                f.write("    (aucune)\n")
            else:
                for _, row in tendances.iterrows():
                    f.write(f"\n    {row['Indicateur']}:\n")
                    f.write(f"      Interaction: F={row['F_Interaction']:.2f}, "
                            f"p_FDR={row['p_Interaction_FDR']:.4f}\n")
                    f.write(f"      Vélo:  t={row['PostHoc_Bike_t']:.2f}, "
                            f"p_corr={row['PostHoc_Bike_p_corr']:.4f}, "
                            f"d={row['PostHoc_Bike_d']:.3f} "
                            f"{'* ' + str(row['PostHoc_Bike_dir']) if row['PostHoc_Bike_sig'] else 'ns'}\n")
                    f.write(f"      VAE:   t={row['PostHoc_eBike_t']:.2f}, "
                            f"p_corr={row['PostHoc_eBike_p_corr']:.4f}, "
                            f"d={row['PostHoc_eBike_d']:.3f} "
                            f"{'* ' + str(row['PostHoc_eBike_dir']) if row['PostHoc_eBike_sig'] else 'ns'}\n")

            # Levene
            f.write(f"\n  ── Homogénéité (Levene) ──\n")
            for _, row in df_res.iterrows():
                if row.get('p_levene', 1.0) < 0.05:
                    f.write(f"    ⚠ {row['Indicateur']}: p_Levene={row['p_levene']:.4f}\n")
            n_levene_fail = (df_res['p_levene'] < 0.05).sum()
            if n_levene_fail == 0:
                f.write("    Tous les indicateurs satisfont l'homogénéité.\n")

        # ── Comparaison ANOVA vs LMM ──
        f.write(f"\n\n{'═' * 80}\n")
        f.write("COMPARAISON ANOVA vs LMM (indicateurs clés)\n")
        f.write(f"{'═' * 80}\n\n")

        f.write("  Note : les p-values brutes ANOVA et LMM sont comparées ci-dessous.\n")
        f.write("  Une concordance forte indique que les résultats ne dépendent pas\n")
        f.write("  du choix de la méthode statistique.\n\n")

        f.write(f"  {'Indicateur':<40} {'ANOVA p':>10} {'LMM p':>10} {'Δ':>8}  Concordance\n")
        f.write(f"  {'─' * 80}\n")

        # On compare sur les indicateurs principaux pour Trajet et Interaction
        for _, row in df_princ.iterrows():
            for effet in ['Trajet', 'Interaction']:
                p_anova = row[f'p_{effet}']
                p_lmm = row.get(f'p_{effet}_LMM', np.nan)
                if np.isnan(p_lmm):
                    continue
                delta = abs(p_anova - p_lmm)
                if (p_anova < 0.05 and p_lmm < 0.05) or (p_anova >= 0.05 and p_lmm >= 0.05):
                    concord = '✓ Concordant'
                else:
                    concord = '✗ Divergent'
                f.write(f"  {row['Indicateur'] + ' (' + effet + ')':<40} "
                        f"{p_anova:>10.4f} {p_lmm:>10.4f} {delta:>8.4f}  {concord}\n")

    print(f"\n  OK: {path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 80)
    print("ANALYSE DE ROBUSTESSE - ANOVA MIXTE")
    print("=" * 80)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Charger
    df = charger_et_filtrer(DATA_FILE, SELECTION_4)
    n_bike = len(df[df['Mode_transport'] == 'Bike'])
    n_ebike = len(df[df['Mode_transport'] == 'e-Bike'])
    print(f"\nSélection 4: {len(df)} participants (Vélo: {n_bike}, VAE: {n_ebike})")

    # Charger résultats LMM pour comparaison
    lmm_combine_path = os.path.join(
        os.path.dirname(OUTPUT_DIR), '26-10-07_codage_somme_sans134',
        'FDR_2niveaux', 'Resultats_Combines.xlsx'
    )
    df_lmm = None
    if os.path.exists(lmm_combine_path):
        df_lmm = pd.read_excel(lmm_combine_path)
        print(f"  Résultats LMM chargés pour comparaison ({len(df_lmm)} indicateurs)")

    # ── Analyser les deux niveaux ──
    for niveau_label, indicateurs, niveau_tag in [
        ('Principale (N-Back)', generer_indicateurs_principaux(), 'Principale'),
        ('Secondaire (ANT)', generer_indicateurs_secondaires(), 'Secondaire'),
    ]:
        print(f"\n{'═' * 70}")
        print(f"  {niveau_label} ({len(indicateurs)} indicateurs)")
        print(f"{'═' * 70}\n")

        resultats = []

        for config in indicateurs:
            nom = config['nom']
            print(f"  {nom}...", end=" ")

            if config['col_away'] not in df.columns or config['col_close'] not in df.columns:
                print("SKIP (colonnes manquantes)")
                continue

            res = anova_mixte(df, config['col_away'], config['col_close'], config['transformation'])

            if res is None:
                print("SKIP (N insuffisant)")
                continue

            row = {
                'Indicateur': nom,
                'Tache': config['tache'],
                'Domaine': config['domaine'],
                'Transformation': config['transformation'] or 'aucune',
                'N': res['N'], 'n_bike': res['n_bike'], 'n_ebike': res['n_ebike'],
                'N_NaN': res['n_nan'], 'Pct_perte': res['pct_perte'],

                'F_Mode': res['F_Mode'], 'p_Mode': res['p_Mode'],
                'df_Mode': f"({res['df_Mode'][0]}, {res['df_Mode'][1]})",
                'eta2_Mode': res['eta2_Mode'], 'd_Mode': res['d_Mode'],

                'F_Trajet': res['F_Trajet'], 'p_Trajet': res['p_Trajet'],
                'df_Trajet': f"({res['df_Trajet'][0]}, {res['df_Trajet'][1]})",
                'eta2_Trajet': res['eta2_Trajet'], 'd_Trajet': res['d_Trajet'],

                'F_Interaction': res['F_Interaction'], 'p_Interaction': res['p_Interaction'],
                'df_Interaction': f"({res['df_Interaction'][0]}, {res['df_Interaction'][1]})",
                'eta2_Interaction': res['eta2_Interaction'], 'd_Interaction': res['d_Interaction'],

                'p_levene': res['p_levene'],

                'PostHoc_Bike_t': res['post_hoc']['Bike']['t'],
                'PostHoc_Bike_p': res['post_hoc']['Bike']['p'],
                'PostHoc_Bike_p_corr': res['post_hoc']['Bike']['p_corr'],
                'PostHoc_Bike_d': res['post_hoc']['Bike']['d'],
                'PostHoc_Bike_dir': res['post_hoc']['Bike']['direction'],
                'PostHoc_Bike_sig': res['post_hoc']['Bike']['sig'],

                'PostHoc_eBike_t': res['post_hoc']['e-Bike']['t'],
                'PostHoc_eBike_p': res['post_hoc']['e-Bike']['p'],
                'PostHoc_eBike_p_corr': res['post_hoc']['e-Bike']['p_corr'],
                'PostHoc_eBike_d': res['post_hoc']['e-Bike']['d'],
                'PostHoc_eBike_dir': res['post_hoc']['e-Bike']['direction'],
                'PostHoc_eBike_sig': res['post_hoc']['e-Bike']['sig'],
            }

            # Ajouter p-values LMM pour comparaison
            if df_lmm is not None:
                lmm_row = df_lmm[df_lmm['Indicateur'] == nom]
                if len(lmm_row) > 0:
                    lmm_row = lmm_row.iloc[0]
                    row['p_Mode_LMM'] = lmm_row.get('p_Mode', np.nan)
                    row['p_Trajet_LMM'] = lmm_row.get('p_Trajet', np.nan)
                    row['p_Interaction_LMM'] = lmm_row.get('p_Interaction', np.nan)

            resultats.append(row)
            print(f"OK  F_Trajet={res['F_Trajet']:.2f} p={res['p_Trajet']:.4f}  "
                  f"F_Inter={res['F_Interaction']:.2f} p={res['p_Interaction']:.4f}")

        if not resultats:
            continue

        df_res = pd.DataFrame(resultats)

        # Correction FDR
        df_res = appliquer_fdr(df_res, niveau_label)

        # Sauvegarder
        fichier = os.path.join(OUTPUT_DIR, f'ANOVA_{niveau_tag}.xlsx')
        df_res.to_excel(fichier, index=False)
        print(f"\n  OK: {fichier}")

        # Stocker pour le rapport
        if niveau_tag == 'Principale':
            df_princ = df_res
        else:
            df_sec = df_res

    # ── Rapport ──
    print(f"\n{'=' * 60}")
    print("RAPPORT")
    print(f"{'=' * 60}")

    generer_rapport(df_princ, df_sec, OUTPUT_DIR)

    # ── Tableau de concordance ANOVA vs LMM ──
    if df_lmm is not None:
        print(f"\n{'=' * 60}")
        print("CONCORDANCE ANOVA vs LMM")
        print(f"{'=' * 60}\n")

        print(f"  {'Indicateur':<40} {'Effet':<15} {'ANOVA p':>10} {'LMM p':>10} {'Concord.':>10}")
        print(f"  {'─' * 90}")

        for _, row in df_princ.iterrows():
            for effet in ['Trajet', 'Interaction']:
                p_anova = row[f'p_{effet}']
                p_lmm = row.get(f'p_{effet}_LMM', np.nan)
                if np.isnan(p_lmm):
                    continue
                same = (p_anova < 0.05) == (p_lmm < 0.05)
                mark = '✓' if same else '✗'
                print(f"  {row['Indicateur']:<40} {effet:<15} {p_anova:>10.4f} {p_lmm:>10.4f} {mark:>10}")

    # ── Résumé ──
    print(f"\n{'=' * 80}")
    print("ANALYSE TERMINÉE")
    print(f"{'=' * 80}")

    print(f"\n  PRINCIPALE ANOVA:")
    for effet in ['Mode', 'Trajet', 'Interaction']:
        n_sig = df_princ[f'Sig_{effet}_FDR'].isin(['*', '**', '***']).sum()
        print(f"    {effet}: {n_sig} significatifs après FDR")

    print(f"\n  SECONDAIRE ANOVA:")
    for effet in ['Mode', 'Trajet', 'Interaction']:
        n_sig = df_sec[f'Sig_{effet}_FDR'].isin(['*', '**', '***']).sum()
        print(f"    {effet}: {n_sig} significatifs après FDR")

    print(f"\n  Fichiers dans: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()