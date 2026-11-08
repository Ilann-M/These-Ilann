# === CODAGE SOMME DES FACTEURS — corrige le 2026-09-09 ===
# Le modele Mode x Trajet etait ajuste en codage treatment (reference 'Bike'), de sorte que
# le terme lu comme 'effet principal du Trajet' etait l'effet du trajet CHEZ LES CYCLISTES
# CLASSIQUES — un effet simple. Le codage somme C(..., Sum) rend les termes marginaux.
# Les post-hocs par mode et les tailles d'effet (calculees sur les donnees) sont inchanges.
# ATTENTION : C(Trajet, Sum)[S.Away] a le signe OPPOSE a Trajet[T.Close].
# Sauvegarde de la version anterieure dans Data\Codes\_Archives\.
# -*- coding: utf-8 -*-
"""
Contrôles de robustesse annoncés au chapitre 3, section 3.9.2.

Les facteurs testés ici sont constants au sein d'un participant et ne peuvent
donc pas confondre le contraste intra-sujet Trajet. Ce que l'on teste est la
STABILITÉ de l'effet d'exposition selon leurs niveaux, c'est-à-dire l'interaction
entre chaque facteur et le facteur Trajet :

  1. Campagne    (automne 2024 / printemps 2025) — la qualité d'horodatage EEG
                  diffère entre campagnes (section 3.7.3)
  2. Ordre       (trajet éloigné d'abord / trajet proche d'abord) — effet de
                  report que l'intervalle entre sessions vise à écarter
  3. Creneau     (matin / après-midi) — niveaux d'exposition et état circadien
  4. Ecart_log   intervalle réel entre les deux sessions de test, en logarithme
  5. Ecart_48h   conforme aux 48 h nominales, ou non

Les deux derniers facteurs répondent à un constat de vérification : l'intervalle
de 48 h annoncé au protocole n'est respecté que chez 30 participants sur 51,
l'étendue observée allant de 24 heures à 29 jours.

Modèle : Valeur ~ Mode_transport * Trajet + <Facteur> * Trajet
         groups=Participant, ordonnée à l'origine aléatoire
         (la pente aléatoire n'est pas identifiable avec une observation par
          cellule ; vérifié sans effet sur les estimations, section 3.9.2)

S'y ajoute une analyse de sensibilité restreinte à la campagne de 2025, seule
équipée de l'horodatage matériel, pour les composantes verrouillées sur le son
et sur l'indice visuel.

Sortie : Data\\Traitement\\Resultats_LMM_intention\\26-08-26_robustesse_nuisance_session
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from LMM_intention import (SELECTION_4, DATA_FILE, generer_catalogue,
                           charger_donnees, filtrer_selection,
                           preparer_donnees_long, _formule)

warnings.filterwarnings('ignore')

BASE_XLSX = D(r'Data\Base.xlsx')
OUTPUT_DIR = S(r'Resultats_LMM_intention'
              r'\26-10-07_nuisance_codage_somme_sans134')
ALPHA = 0.05

FACTEURS = ['Campagne', 'Ordre', 'Creneau', 'Ecart_log', 'Ecart_48h']

# indicateurs concernés par le décalage résiduel entre campagnes (section 3.7.3)
INDIC_HORODATAGE = ['Moy_sonore', 'Moy_indice']

_RX_DATE = re.compile(r'\s*(\d{1,2})\s*/\s*(\d{1,2})\s*:?\s*(\d{1,2})\s*h\s*(\d{0,2})')


# ---------------------------------------------------------------- covariables
def _horodate(txt):
    """'30/09 : 17h15' -> Timestamp. Annee deduite du mois (campagnes disjointes)."""
    m = _RX_DATE.match(str(txt))
    if not m:
        return pd.NaT
    jour, mois, heure = int(m.group(1)), int(m.group(2)), int(m.group(3))
    minute = int(m.group(4)) if m.group(4) else 0
    try:
        return pd.Timestamp(2024 if mois >= 9 else 2025, mois, jour, heure, minute)
    except ValueError:
        return pd.NaT


def construire_covariables():
    """Renvoie Participant / Campagne / Ordre / Creneau / Ecart_h / Ecart_log / Ecart_48h."""
    b = pd.read_excel(BASE_XLSX)
    b.columns = [str(c).strip() for c in b.columns]
    b['Participant'] = (b['Participant'].astype(str)
                        .str.extract(r'[Pp](\d+)')[0].astype(float))
    b = b.dropna(subset=['Participant'])
    b['Participant'] = b['Participant'].astype(int)

    col_p2 = 'Passation-2' if 'Passation-2' in b.columns else 'Passation_2'

    t2, t3 = b['Date_P2'].map(_horodate), b['Date_P3'].map(_horodate)
    b['Ecart_h'] = (t3 - t2).dt.total_seconds() / 3600.0
    b['Ecart_log'] = np.log(b['Ecart_h'])
    b['Ecart_48h'] = np.where(b['Ecart_h'].isna(), np.nan,
                              np.where(b['Ecart_h'].between(47, 49),
                                       'conforme', 'ecart'))
    b['Creneau'] = np.where(t2.dt.hour < 12, 'matin',
                            np.where(t2.notna(), 'aprem', np.nan))
    b['Campagne'] = np.where(t2.dt.year == 2024, 'aut24',
                             np.where(t2.notna(), 'print25', np.nan))

    # ordre = condition realisee lors de la premiere session de test
    p2 = b[col_p2].astype(str).str.strip().str.upper()
    b['Ordre'] = np.where(p2 == 'A', 'A_first',
                          np.where(p2 == 'C', 'C_first', np.nan))

    return b[['Participant', 'Campagne', 'Ordre', 'Creneau',
              'Ecart_h', 'Ecart_log', 'Ecart_48h']]


# ------------------------------------------------------------------ ajustement
def tester_interaction(df_long, facteur):
    """Ajuste Valeur ~ Mode*Trajet + facteur*Trajet ; renvoie le p de l'interaction.

    Gere indifferemment un facteur categoriel (terme 'facteur[T.x]:Trajet[T.y]')
    et une covariable continue (terme 'Trajet[T.y]:facteur').
    """
    if df_long[facteur].nunique() < 2:
        return None
    # Le terme de session est ajoute pour tous les facteurs SAUF l'ordre des
    # trajets : la session est par construction une fonction de l'ordre et du
    # trajet reel, si bien que l'interaction Ordre x Trajet lui est aliasee et
    # devient non estimable. Le chapitre 3 (section 3.9.2) releve ce recoupement.
    base = f"Valeur ~ Mode_transport * Trajet + {facteur} * Trajet"
    formule = base if facteur == 'Ordre' else _formule(df_long, base)
    try:
        res = mixedlm(formule, df_long, groups=df_long['Participant']).fit(reml=False)
    except Exception:
        return None
    cible = [k for k in res.params.index
             if facteur in k and 'Trajet' in k and ':' in k]
    if not cible:
        return None
    k = cible[0]
    return dict(terme=k, coef=res.params[k], p=res.pvalues[k])


def effet_trajet(df_long):
    """Effet Trajet du modele de reference (ordonnee a l'origine aleatoire seule)."""
    try:
        res = mixedlm(_formule(df_long, "Valeur ~ C(Mode_transport, Sum) * C(Trajet, Sum)"),
                      df_long,
                      groups=df_long['Participant']).fit(reml=True)
    except Exception:
        return None
    k = [x for x in res.params.index if 'Trajet' in x and ':' not in x]
    if not k:
        return None
    return dict(coef=res.params[k[0]], p=res.pvalues[k[0]],
                n=df_long['Participant'].nunique())


# ------------------------------------------------------------------------ main
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cov = construire_covariables()
    df = filtrer_selection(charger_donnees(DATA_FILE), SELECTION_4)
    # charger_donnees fusionne desormais sa propre colonne Ordre (terme de
    # session) : on la retire avant la fusion des covariables pour eviter
    # les suffixes _x / _y. Les deux colonnes suivent la meme regle.
    doublons = [c for c in cov.columns if c != 'Participant' and c in df.columns]
    if doublons:
        df = df.drop(columns=doublons)
    df = df.merge(cov, on='Participant', how='left')

    print("Covariables sur les participants analyses :")
    for c in ['Campagne', 'Ordre', 'Creneau', 'Ecart_48h']:
        print(f"  {c:10s} {df[c].value_counts(dropna=False).to_dict()}")
    e = df['Ecart_h'].dropna()
    print(f"  Ecart_h    mediane {e.median():.0f} h | etendue {e.min():.0f}-{e.max():.0f} h "
          f"| conforme 48 h : {(df['Ecart_48h'] == 'conforme').sum()}/{len(df)}")
    print(f"\n{len(df)} participants, "
          f"{df[FACTEURS].isna().any(axis=1).sum()} avec une covariable manquante\n")

    catalogue = generer_catalogue()
    lignes, lignes_2025 = [], []
    cols_cov = ['Participant'] + FACTEURS

    for indic in catalogue:
        nom, ca, cc = indic['nom'], indic['col_away'], indic['col_close']
        if ca not in df.columns or cc not in df.columns:
            continue
        # preparer_donnees_long renvoie (df_long, stats) et ne conserve que
        # Participant / Mode_transport / Valeur / Trajet : les covariables sont
        # rattachees ensuite depuis le tableau large.
        lg, _ = preparer_donnees_long(df, ca, cc, indic.get('transformation'))
        if lg is None or lg['Participant'].nunique() < 10:
            continue
        # preparer_donnees_long emporte deja Ordre (terme de session) : on
        # evite le doublon avant de rattacher les covariables.
        lg = lg.drop(columns=[c for c in cols_cov
                              if c != 'Participant' and c in lg.columns])
        lg = lg.merge(df[cols_cov], on='Participant', how='left')

        ref = effet_trajet(lg)
        ligne = {'Indicateur': nom, 'Tache': indic['tache'],
                 'Famille': indic['categorie'],   # N-Back / ANT : partition FDR principale
                 'Domaine': indic['domaine'],
                 'N_participants': lg['Participant'].nunique(),
                 'p_Trajet_reference': ref['p'] if ref else np.nan}

        for facteur in FACTEURS:
            r = tester_interaction(lg.dropna(subset=[facteur]), facteur)
            ligne[f'p_{facteur}xTrajet'] = r['p'] if r else np.nan
            ligne[f'coef_{facteur}xTrajet'] = r['coef'] if r else np.nan
        lignes.append(ligne)

        # sensibilite restreinte a 2025
        s25 = lg[lg['Campagne'] == 'print25']
        r25 = effet_trajet(s25) if s25['Participant'].nunique() >= 10 else None
        lignes_2025.append({
            'Indicateur': nom, 'Tache': indic['tache'],
            'Horodatage_concerne': any(k in nom for k in INDIC_HORODATAGE),
            'N_total': ligne['N_participants'],
            'p_Trajet_total': ligne['p_Trajet_reference'],
            'N_2025': s25['Participant'].nunique(),
            'p_Trajet_2025': r25['p'] if r25 else np.nan,
            'coef_Trajet_total': ref['coef'] if ref else np.nan,
            'coef_Trajet_2025': r25['coef'] if r25 else np.nan,
        })
        print(f"  {nom[:40]:40s} ok")

    res = pd.DataFrame(lignes)
    res25 = pd.DataFrame(lignes_2025)

    # correction FDR sur la partition principale du chapitre : N-Back (8) / ANT (23)
    for facteur in FACTEURS:
        col, qcol = f'p_{facteur}xTrajet', f'q_{facteur}xTrajet'
        res[qcol] = np.nan
        for famille in res['Famille'].unique():
            m = res['Famille'].eq(famille) & res[col].notna()
            if m.sum():
                res.loc[m, qcol] = multipletests(res.loc[m, col], method='fdr_bh')[1]

    res.to_excel(os.path.join(OUTPUT_DIR, 'Robustesse_facteurs_nuisance.xlsx'), index=False)
    res25.to_excel(os.path.join(OUTPUT_DIR, 'Sensibilite_campagne_2025.xlsx'), index=False)

    # ------------------------------------------------------------------ rapport
    with open(os.path.join(OUTPUT_DIR, 'Rapport_robustesse.txt'), 'w',
              encoding='utf-8') as f:
        f.write("CONTROLES DE ROBUSTESSE - FACTEURS DE NUISANCE\n")
        f.write("=" * 78 + "\n\n")
        f.write("Modele : Valeur ~ Mode_transport * Trajet + <Facteur> * Trajet\n")
        f.write("Chaque facteur est constant au sein d'un participant : son interaction\n")
        f.write("avec Trajet teste la stabilite de l'effet d'exposition selon ses niveaux.\n")
        f.write("Correction FDR (Benjamini-Hochberg) sur la partition principale du\n")
        f.write(f"chapitre 3 : famille N-Back (8 tests) / famille ANT (23). Seuil {ALPHA}.\n\n")
        f.write(f"Intervalle entre sessions : mediane {e.median():.0f} h, "
                f"etendue {e.min():.0f}-{e.max():.0f} h, "
                f"conforme aux 48 h chez {(df['Ecart_48h'] == 'conforme').sum()}/{len(df)}.\n\n")

        for facteur in FACTEURS:
            col, qcol = f'p_{facteur}xTrajet', f'q_{facteur}xTrajet'
            sig = res[res[qcol] < ALPHA]
            brut = res[res[col] < ALPHA]
            f.write(f"--- {facteur} x Trajet ---\n")
            f.write(f"  significatifs apres FDR : {len(sig)} / {res[col].notna().sum()}\n")
            f.write(f"  p bruts < .05           : {len(brut)}\n")
            for _, r in brut.sort_values(col).iterrows():
                etoile = ' *' if r[qcol] < ALPHA else ''
                f.write(f"    {r['Indicateur'][:40]:40s} p={r[col]:.4f}  q={r[qcol]:.4f}{etoile}\n")
            f.write("\n")

        f.write("--- Sensibilite restreinte a la campagne 2025 ---\n")
        for _, r in res25[res25['Horodatage_concerne']].iterrows():
            f.write(f"  {r['Indicateur'][:40]:40s} "
                    f"p(total, N={int(r['N_total'])}) = {r['p_Trajet_total']:.4f}   "
                    f"p(2025, N={int(r['N_2025'])}) = {r['p_Trajet_2025']:.4f}\n")

    print(f"\nSorties : {OUTPUT_DIR}")
    for facteur in FACTEURS:
        print(f"  {facteur:10s} x Trajet : "
              f"{(res[f'q_{facteur}xTrajet'] < ALPHA).sum()} significatifs apres FDR, "
              f"{(res[f'p_{facteur}xTrajet'] < ALPHA).sum()} en p brut")


if __name__ == "__main__":
    main()
