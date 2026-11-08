"""
════════════════════════════════════════════════════════════════════════════════
STATISTIQUES DESCRIPTIVES (N, Mean, SD) PAR TRAJET × MODE DE TRANSPORT

Extrait les statistiques descriptives de tous les indicateurs d'intérêt
N-BACK et ANT, différenciées :
  - par trajet  : Away (loin du trafic) / Close (proche du trafic)
  - par mode    : Vélo (Bike) / VAE (e-Bike)
sur la sélection fixe de 51 participants (SELECTION_4).

Descriptif uniquement (pas de FDR). RT en millisecondes brutes (aucun log).
Deux échantillons produits à chaque exécution (un jeu de fichiers chacun) :
  - 'par_cellule'      : non-NA par colonne, N variable (descriptif inclusif)
  - 'paires_completes' : listwise A–C par indicateur, N = N de l'échantillon
                         LMM (logique reprise de Descriptives_Correctes.py,
                         désormais intégrée ici — ce script le remplace).

Patterns de colonnes calqués sur :
  Analyse_expositionR\\LMM_expoR_FDR_2niveau.py  (indicateurs N-BACK / ANT)
Structure d'export calquée sur :
  Analyse_intention\\Descriptives_Correctes.py    (export Excel multi-feuilles)
════════════════════════════════════════════════════════════════════════════════
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402

import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Console UTF-8 (textes français avec accents)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_FILE = D(r'Data\Table_combine\Tab_sujet_20260205.xlsx')
OUTPUT_DIR = S(r'Descriptives_Trajet_Mode')
DATE_STR = datetime.now().strftime('%Y%m%d')

VERSION_DONNEES = 'filtre'

SELECTION_4 = [101, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
               201, 202, 203, 204, 205, 206, 209, 210, 211, 212, 215, 216,
               121, 122, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 139,
               221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]

# Modes de transport (valeurs de la colonne 'Mode_transport') -> libellé thèse
MODES = [('Bike', 'Vélo'), ('e-Bike', 'VAE')]

# Modes d'échantillon calculés (les deux sont produits à chaque exécution) :
#   'par_cellule'      : toutes les valeurs non-NA de la cellule (N variable)
#   'paires_completes' : listwise A–C par indicateur (N = N de l'échantillon
#                        LMM ; logique reprise de Descriptives_Correctes.py)
ECHANTILLONS = ['par_cellule', 'paires_completes']
ECHANTILLON_LIB = {
    'par_cellule': 'données disponibles par cellule (non-NA, pas de listwise A–C)',
    'paires_completes': 'paires complètes A–C par indicateur (listwise, N = N du LMM)',
}

# Trajets : (libellé, clé de colonne dans l'indicateur)
TRAJETS = [('Away', 'col_away'), ('Close', 'col_close')]


# ==============================================================================
# DÉFINITION DES INDICATEURS (transformation=None partout : valeurs brutes)
# ==============================================================================

def generer_indicateurs_nback():
    """N-BACK : 8 indicateurs (1B + 2B, EEG + Comportement). Feuille 'NBack'."""
    indicateurs = []

    # EEG : N1 + P3 (Valides) × 1B / 2B  (4)
    for tache in ['1B', '2B']:
        for composante in ['N1', 'P3']:
            indicateurs.append({
                'nom': f'{composante}_Moy_Valide_{tache}',
                'label': f'{composante} Valide {tache}',
                'domaine': 'EEG',
                'tache': tache,
                'reseau': '',
                'unite': 'µV',
                'categorie': 'NBack',
                'col_away': f'{composante}_Moy_Valide_A_{tache}',
                'col_close': f'{composante}_Moy_Valide_C_{tache}',
            })

    # Comportement : RT_moyen (ms brutes) + d_prime × 1B / 2B  (4)
    for tache in ['1B', '2B']:
        indicateurs.append({
            'nom': f'RT_moyen_{tache}',
            'label': f'RT moyen {tache} (ms)',
            'domaine': 'Comportement',
            'tache': tache,
            'reseau': '',
            'unite': 'ms',
            'categorie': 'NBack',
            'col_away': f'RT_moyen_{VERSION_DONNEES}_A_{tache}',
            'col_close': f'RT_moyen_{VERSION_DONNEES}_C_{tache}',
        })
        indicateurs.append({
            'nom': f'd_prime_{tache}',
            'label': f"d' {tache}",
            'domaine': 'Comportement',
            'tache': tache,
            'reseau': '',
            'unite': '',
            'categorie': 'NBack',
            'col_away': f'd_prime_{VERSION_DONNEES}_A_{tache}',
            'col_close': f'd_prime_{VERSION_DONNEES}_C_{tache}',
        })

    return indicateurs


def generer_indicateurs_ant():
    """ANT : EEG (13, feuille 'ANT_EEG') + Comportement RT/ER (feuille 'ANT_Comportement')."""
    indicateurs = []

    # ── EEG flèche : N1 + P3 × 5 réseaux (10) ──
    # ⚠ "Contrôle exécutif" = 'Controle' dans les colonnes flèche
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
                'label': f'{composante} flèche {reseau_label}',
                'domaine': 'EEG',
                'tache': 'ANT',
                'reseau': reseau_label,
                'unite': 'µV',
                'categorie': 'ANT_EEG',
                'col_away': f'{composante}_{cond}_Moy_fleche_A_ANT',
                'col_close': f'{composante}_{cond}_Moy_fleche_C_ANT',
            })

    # ── EEG sonore : N1 + P2 (Alerte Standard) (2) ──
    for composante in ['N1', 'P2']:
        indicateurs.append({
            'nom': f'{composante}_Moy_sonore_Alerte_Standard_ANT',
            'label': f'{composante} sonore Alerte Standard',
            'domaine': 'EEG',
            'tache': 'ANT',
            'reseau': 'Alerte_Standard',
            'unite': 'µV',
            'categorie': 'ANT_EEG',
            'col_away': f'{composante}_Alerte_Standard_Moy_sonore_A_ANT',
            'col_close': f'{composante}_Alerte_Standard_Moy_sonore_C_ANT',
        })

    # ── EEG indice : N1 (Alerte Visuelle) (1) ──
    indicateurs.append({
        'nom': 'N1_Moy_indice_Alerte_Visuelle_ANT',
        'label': 'N1 indice Alerte Visuelle',
        'domaine': 'EEG',
        'tache': 'ANT',
        'reseau': 'Alerte_Visuelle',
        'unite': 'µV',
        'categorie': 'ANT_EEG',
        'col_away': 'N1_Alerte_Visuelle_Moy_indice_A_ANT',
        'col_close': 'N1_Alerte_Visuelle_Moy_indice_C_ANT',
    })

    # ── Comportement RT (ms brutes) × 5 réseaux (5) — noms LONGS ──
    reseaux_rt = ['Alerte_Standard', 'Alerte_Auditive', 'Alerte_Visuelle',
                  'Orientation', 'Controle_Executif']
    for reseau in reseaux_rt:
        indicateurs.append({
            'nom': f'RT_{reseau}_ANT',
            'label': f'RT {reseau} (ms)',
            'domaine': 'Comportement',
            'tache': 'ANT',
            'reseau': reseau,
            'unite': 'ms',
            'categorie': 'ANT_Comportement',
            'col_away': f'RT_{reseau}_A_ANT',
            'col_close': f'RT_{reseau}_C_ANT',
        })

    # ── Comportement ER (taux d'erreur) × 5 réseaux (5) — noms LONGS (idem RT) ──
    # Vérifié sur Tab_sujet_20260205.xlsx : ER_<reseau>_{A|C}_ANT, mêmes
    # libellés réseau que le RT (les noms courts d'anciens scripts sont obsolètes).
    for reseau in reseaux_rt:
        indicateurs.append({
            'nom': f'ER_{reseau}_ANT',
            'label': f'ER {reseau}',
            'domaine': 'Comportement',
            'tache': 'ANT',
            'reseau': reseau,
            'unite': 'taux',
            'categorie': 'ANT_Comportement',
            'col_away': f'ER_{reseau}_A_ANT',
            'col_close': f'ER_{reseau}_C_ANT',
        })

    return indicateurs


# ==============================================================================
# CALCUL DES DESCRIPTIVES (par cellule Mode × Trajet, données disponibles)
# ==============================================================================

def calculer_descriptives(df_sel, indicateurs, echantillon='par_cellule'):
    """
    Pour chaque indicateur × Mode × Trajet : N, Mean, SD, Min, Max.

    echantillon :
      'par_cellule'      -> valeurs non-NA de chaque colonne, indépendamment
                            (N variable Away/Close ; pas de couplage A–C)
      'paires_completes' -> par indicateur, on ne garde que les participants
                            ayant Away ET Close non-NA, puis stats par cellule
                            (N identique Away/Close = N de l'échantillon LMM)

    Retourne (df_long, colonnes_manquantes).
    """
    colonnes_dispo = set(df_sel.columns)
    lignes = []
    colonnes_manquantes = []

    for ind in indicateurs:
        col_a, col_c = ind['col_away'], ind['col_close']
        present_a = col_a in colonnes_dispo
        present_c = col_c in colonnes_dispo

        if not present_a:
            colonnes_manquantes.append({
                'Indicateur': ind['nom'], 'Trajet': 'Away',
                'Colonne': col_a, 'Categorie': ind['categorie'],
            })
        if not present_c:
            colonnes_manquantes.append({
                'Indicateur': ind['nom'], 'Trajet': 'Close',
                'Colonne': col_c, 'Categorie': ind['categorie'],
            })

        if echantillon == 'paires_completes':
            # Les deux colonnes sont nécessaires pour former des paires
            if not (present_a and present_c):
                continue
            base = df_sel[df_sel[col_a].notna() & df_sel[col_c].notna()]
        else:
            base = df_sel

        for trajet_label, cle_col in TRAJETS:
            col = ind[cle_col]
            if col not in colonnes_dispo:
                continue

            for mode_val, mode_lib in MODES:
                serie = base.loc[base['Mode_transport'] == mode_val, col].dropna()
                n = len(serie)
                lignes.append({
                    'Echantillon': echantillon,
                    'Categorie': ind['categorie'],
                    'Domaine': ind['domaine'],
                    'Tache': ind['tache'],
                    'Reseau': ind['reseau'],
                    'Indicateur': ind['nom'],
                    'Label': ind['label'],
                    'Unite': ind['unite'],
                    'Mode': mode_lib,
                    'Mode_transport': mode_val,
                    'Trajet': trajet_label,
                    'Colonne': col,
                    'N': n,
                    'Mean': serie.mean() if n > 0 else np.nan,
                    'SD': serie.std(ddof=1) if n > 1 else np.nan,
                    'Min': serie.min() if n > 0 else np.nan,
                    'Max': serie.max() if n > 0 else np.nan,
                })

    return pd.DataFrame(lignes), colonnes_manquantes


def construire_format_these(df_long):
    """
    Tableau large lisible : 1 ligne / indicateur,
    colonnes N + 'Mean (SD)' pour chaque cellule Vélo/VAE × Away/Close.
    """
    cols_ordre = ['Categorie', 'Domaine', 'Tache', 'Reseau', 'Indicateur', 'Label', 'Unite']
    base = df_long[cols_ordre].drop_duplicates('Indicateur').reset_index(drop=True)

    lignes = []
    for _, meta in base.iterrows():
        indicateur = meta['Indicateur']
        ligne = {c: meta[c] for c in cols_ordre}
        for mode_val, mode_lib in MODES:
            for trajet_label, _ in TRAJETS:
                sel = df_long[
                    (df_long['Indicateur'] == indicateur)
                    & (df_long['Mode'] == mode_lib)
                    & (df_long['Trajet'] == trajet_label)
                ]
                prefixe = f'{mode_lib}_{trajet_label}'
                if len(sel) == 0:
                    ligne[f'{prefixe}_N'] = 0
                    ligne[f'{prefixe}_Mean_SD'] = ''
                    continue
                r = sel.iloc[0]
                if r['N'] > 0:
                    # Les taux d'erreur sont des contrastes en proportion : a
                    # trois decimales, plusieurs cellules s'ecrasent sur 0,000
                    # (dont une en -0,000). Une decimale de plus les separe.
                    nd = 4 if meta['Unite'] == 'taux' else 3
                    sd_txt = f"{r['SD']:.{nd}f}" if pd.notna(r['SD']) else 'NA'
                    mean_sd = f"{r['Mean']:.{nd}f} ({sd_txt})"
                else:
                    mean_sd = ''
                ligne[f'{prefixe}_N'] = int(r['N'])
                ligne[f'{prefixe}_Mean_SD'] = mean_sd
        lignes.append(ligne)

    return pd.DataFrame(lignes)


# ==============================================================================
# EXPORT
# ==============================================================================

def exporter_excel(df_long, df_these, output_dir, echantillon):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fichier = os.path.join(
        output_dir, f'Descriptives_Trajet_Mode_{echantillon}_{DATE_STR}.xlsx')

    cols_long = ['Echantillon', 'Categorie', 'Domaine', 'Tache', 'Reseau',
                 'Indicateur', 'Label', 'Unite', 'Mode', 'Trajet', 'Colonne',
                 'N', 'Mean', 'SD', 'Min', 'Max']

    def _fmt(df):
        d = df[[c for c in cols_long if c in df.columns]].copy()
        taux = d['Unite'].eq('taux') if 'Unite' in d.columns else False
        for c in ['Mean', 'SD', 'Min', 'Max']:
            d[c] = d[c].where(taux, d[c].round(3)).mask(taux, d[c].round(4))
        return d

    feuilles = {
        'NBack': df_long[df_long['Categorie'] == 'NBack'],
        'ANT_EEG': df_long[df_long['Categorie'] == 'ANT_EEG'],
        'ANT_Comportement': df_long[df_long['Categorie'] == 'ANT_Comportement'],
    }

    with pd.ExcelWriter(fichier, engine='openpyxl') as writer:
        for nom, sous in feuilles.items():
            if len(sous) > 0:
                _fmt(sous).to_excel(writer, sheet_name=nom, index=False)
        _fmt(df_long).to_excel(writer, sheet_name='Synthese_long', index=False)
        df_these.to_excel(writer, sheet_name='Format_These', index=False)

    print(f"\nOK - Excel exporté : {fichier}")
    return fichier


def exporter_rapport(df_long, colonnes_manquantes, n_total, n_bike, n_ebike,
                     output_dir, echantillon):
    fichier = os.path.join(
        output_dir, f'Descriptives_Trajet_Mode_{echantillon}_{DATE_STR}.txt')

    with open(fichier, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTIQUES DESCRIPTIVES PAR TRAJET × MODE DE TRANSPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date          : {DATE_STR}\n")
        f.write(f"Source        : {DATA_FILE}\n")
        f.write(f"Sélection     : SELECTION_4 ({len(SELECTION_4)} participants visés)\n")
        f.write(f"Participants  : {n_total} (Vélo={n_bike}, VAE={n_ebike})\n")
        f.write(f"Échantillon   : {ECHANTILLON_LIB[echantillon]}\n")
        f.write("RT            : millisecondes brutes (aucune transformation log)\n\n")

        # ── Colonnes ER réellement présentes / absentes ──
        f.write("-" * 80 + "\n")
        f.write("COLONNES ER (vérification du naming réel)\n")
        f.write("-" * 80 + "\n")
        er = df_long[df_long['Indicateur'].str.startswith('ER_')]
        er_ok = sorted(er['Colonne'].unique())
        f.write(f"ER colonnes TROUVÉES ({len(er_ok)}): {er_ok}\n")
        er_manq = [c['Colonne'] for c in colonnes_manquantes if c['Indicateur'].startswith('ER_')]
        f.write(f"ER colonnes MANQUANTES ({len(er_manq)}): {sorted(set(er_manq))}\n\n")

        if colonnes_manquantes:
            f.write("-" * 80 + "\n")
            f.write(f"TOUTES COLONNES MANQUANTES ({len(colonnes_manquantes)})\n")
            f.write("-" * 80 + "\n")
            for c in colonnes_manquantes:
                f.write(f"  [{c['Categorie']}] {c['Indicateur']} ({c['Trajet']}) -> {c['Colonne']}\n")
            f.write("\n")

        # ── Détail par catégorie ──
        for cat in ['NBack', 'ANT_EEG', 'ANT_Comportement']:
            sous = df_long[df_long['Categorie'] == cat]
            if len(sous) == 0:
                continue
            f.write("=" * 80 + "\n")
            f.write(f"{cat}  ({sous['Indicateur'].nunique()} indicateurs)\n")
            f.write("=" * 80 + "\n")
            for indicateur in sous['Indicateur'].unique():
                di = sous[sous['Indicateur'] == indicateur]
                f.write(f"\n{indicateur}  [{di.iloc[0]['Label']}]\n")
                for _, r in di.iterrows():
                    mean = f"{r['Mean']:8.3f}" if pd.notna(r['Mean']) else "      NA"
                    sd = f"{r['SD']:7.3f}" if pd.notna(r['SD']) else "     NA"
                    f.write(f"  {r['Mode']:5s} {r['Trajet']:6s} : "
                            f"N={int(r['N']):2d}  M={mean}  SD={sd}\n")
            f.write("\n")

    print(f"OK - Rapport exporté : {fichier}")
    return fichier


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 80)
    print("STATISTIQUES DESCRIPTIVES PAR TRAJET × MODE DE TRANSPORT")
    print("=" * 80)

    print(f"\nChargement : {DATA_FILE}")
    df = pd.read_excel(DATA_FILE)
    print(f"  OK : {df.shape[0]} participants, {df.shape[1]} colonnes")
    # === 07/10/2026 (comme LMM_intention.py) : la session 2-back du trajet éloigné du
    # participant 134 (aucun appui enregistré, boîtier défaillant) est retirée des indicateurs comportementaux du 2-back ;
    # les ERP sont conservés. Le d' passe à NaN, le TR y était déjà indéfini ; les paires complètes s'ajustent d'elles-mêmes.
    df.loc[df['Participant'] == 134, [c for c in df.columns if c.startswith('d_prime_') and c.endswith('_A_2B')]] = float('nan')

    df_sel = df[df['Participant'].isin(SELECTION_4)].copy()
    n_total = len(df_sel)
    n_bike = len(df_sel[df_sel['Mode_transport'] == 'Bike'])
    n_ebike = len(df_sel[df_sel['Mode_transport'] == 'e-Bike'])
    print(f"\nSélection 4 : {n_total} participants (visés : {len(SELECTION_4)})")
    print(f"  Vélo (Bike)   : {n_bike}")
    print(f"  VAE  (e-Bike) : {n_ebike}")

    indicateurs = generer_indicateurs_nback() + generer_indicateurs_ant()
    print(f"\nIndicateurs définis : {len(indicateurs)}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for echantillon in ECHANTILLONS:
        print("\n" + "=" * 80)
        print(f"ÉCHANTILLON : {echantillon}  ({ECHANTILLON_LIB[echantillon]})")
        print("=" * 80)

        df_long, colonnes_manquantes = calculer_descriptives(
            df_sel, indicateurs, echantillon)

        if len(df_long) == 0:
            print("ERREUR : aucune donnée calculée (toutes colonnes manquantes ?)")
            continue

        n_ind_ok = df_long['Indicateur'].nunique()
        print(f"OK - Statistiques calculées pour {n_ind_ok} indicateurs")
        if colonnes_manquantes:
            print(f"  ⚠ {len(colonnes_manquantes)} colonne(s) manquante(s) (voir rapport .txt)")
            er_manq = sorted({c['Colonne'] for c in colonnes_manquantes
                              if c['Indicateur'].startswith('ER_')})
            if er_manq:
                print(f"  ⚠ ER manquantes : {er_manq}")

        df_these = construire_format_these(df_long)

        exporter_excel(df_long, df_these, OUTPUT_DIR, echantillon)
        exporter_rapport(df_long, colonnes_manquantes, n_total, n_bike, n_ebike,
                         OUTPUT_DIR, echantillon)

        print(f"\nRésumé par catégorie ({echantillon}) :")
        for cat in ['NBack', 'ANT_EEG', 'ANT_Comportement']:
            sous = df_long[df_long['Categorie'] == cat]
            if len(sous) > 0:
                print(f"  {cat:18s} : {sous['Indicateur'].nunique():2d} indicateurs, "
                      f"{len(sous)} lignes (Mode × Trajet)")

    print("\nTerminé.\n")


if __name__ == "__main__":
    main()
