# -*- coding: utf-8 -*-
"""
Comparabilité initiale des deux groupes de mode de transport.

Le mode de transport est le facteur inter-participants du plan, et son
attribution ne relève pas d'un tirage aléatoire mais d'une séquence fixée
d'avance (chapitre 3, section 3.8.3). Il importe donc de montrer que les deux
groupes ne diffèrent pas sur les caractéristiques susceptibles de peser sur la
performance cognitive ou sur la réponse à l'effort — ce que le chapitre ne
rapportait jusqu'ici que pour l'âge et le genre.

Les valeurs de p sont fournies à titre descriptif : sur un échantillon de cette
taille, l'absence de significativité ne démontre pas l'équivalence. Ce sont les
tailles d'effet, et leur intervalle de confiance, qui portent la lecture.

Sortie : Data\\Traitement\\Descriptives\\Comparabilite_groupes_mode
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402
import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

TAB_SUJET = D(r'Data\Table_combine\Tab_sujet_20260205.xlsx')
# Les questionnaires unifies de Data\Cognition\Questionnaire, produits par
# Q_traitement_vie_all.py, portent le correctif « age Q2 2024 -> 2025 ». Les
# copies restees dans les dossiers de campagne lui sont anterieures et donnent
# aux 25 participants du printemps 2025 un an de moins (age moyen Q2 : 23,18
# contre 24,18).
QUESTIONNAIRES = [
    D(r'Data\Cognition\Questionnaire\resultats_Q*-vie.xlsx'),
]
OUTPUT_DIR = S(r'Descriptives\Comparabilite_groupes_mode')

SELECTION_4 = [101, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
               201, 202, 203, 204, 205, 206, 209, 210, 211, 212, 215, 216,
               121, 122, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 139,
               221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]

# libelle -> nom de colonne
CONTINUES = [
    ("Âge (ans)", 'Âge'),
    ("Indice de masse corporelle", 'IMC'),
    ("Activité physique totale (ONAPS-PAQ)", 'APtot'),
    ("Stress perçu (PSS)", 'Score_PSS'),
    ("Symptomatologie dépressive (PHQ-9)", 'Score_PHQ'),
    ("Qualité du sommeil (PSQI)", 'Score_PSQI'),
    ("Efficacité du sommeil", 'score_efficacite'),
    ("Alimentation (score nutrition)", 'Score_Nutrition'),
    ("Nicotine (ASSIST)", 'score_nicotine'),
    ("Alcool (ASSIST)", 'score_alcool'),
    ("Cannabis (ASSIST)", 'score_cannabis'),
    ("Exposition chronique déclarée", 'expo_total'),
]
CATEGORIELLES = [("Genre", 'Sexe')]

# L'age de Tab_sujet precede le correctif « age Q2 2024 -> 2025 » porte par
# Q_traitement_vie_all.py : les 25 participants du printemps 2025 y sont plus
# jeunes d'un an. La source questionnaire fait donc foi pour cette colonne.
PRIORITE_QUESTIONNAIRE = ['Âge']


def charger():
    d = pd.read_excel(TAB_SUJET)
    d = d[d['Participant'].isin(SELECTION_4)].copy()

    # variables presentes seulement dans les sorties questionnaires
    blocs = []
    for motif in QUESTIONNAIRES:
        for f in glob.glob(motif):
            q = pd.read_excel(f)
            col_id = 'ID' if 'ID' in q.columns else q.columns[0]
            q = q.rename(columns={col_id: 'Participant'})
            q['Participant'] = (q['Participant'].astype(str)
                                .str.extract(r'(\d+)')[0].astype(float))
            blocs.append(q.dropna(subset=['Participant']))
    if blocs:
        q = pd.concat(blocs, ignore_index=True)
        q['Participant'] = q['Participant'].astype(int)
        q = q.drop_duplicates(subset='Participant', keep='first')
        manquantes = [c for _, c in CONTINUES if c not in d.columns and c in q.columns]
        if manquantes:
            d = d.merge(q[['Participant'] + manquantes], on='Participant', how='left')
        # colonnes presentes des deux cotes dont la version questionnaire fait foi
        for col in PRIORITE_QUESTIONNAIRE:
            if col in q.columns and col in d.columns:
                ref = q.set_index('Participant')[col]
                d[col] = d['Participant'].map(ref).fillna(d[col])
    return d


def hedges_g(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan, (np.nan, np.nan)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if s == 0:
        return np.nan, (np.nan, np.nan)
    d = (a.mean() - b.mean()) / s
    j = 1 - 3 / (4 * (na + nb) - 9)      # correction de biais pour petits effectifs
    g = d * j
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
    return g, (g - 1.96 * se, g + 1.96 * se)


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    d = charger()
    groupes = sorted(d['Mode_transport'].dropna().unique())
    assert len(groupes) == 2, f"groupes trouves : {groupes}"
    g1, g2 = groupes
    print(f"{len(d)} participants : {g1} n={int((d['Mode_transport'] == g1).sum())}, "
          f"{g2} n={int((d['Mode_transport'] == g2).sum())}\n")

    lignes, rapport = [], []

    def ecrire(t=''):
        print(t)
        rapport.append(t)

    ecrire("=" * 96)
    ecrire("COMPARABILITE INITIALE DES DEUX GROUPES DE MODE DE TRANSPORT")
    ecrire("=" * 96)
    ecrire(f"\n{g1} : n = {int((d['Mode_transport'] == g1).sum())}    "
           f"{g2} : n = {int((d['Mode_transport'] == g2).sum())}")
    ecrire("\nLes p sont descriptifs : une absence de significativite ne demontre pas")
    ecrire("l'equivalence. La lecture repose sur g de Hedges et son intervalle a 95 %.\n")
    ecrire(f"{'Variable':38s} {g1:>16s} {g2:>16s} {'g [IC 95%]':>22s} {'p':>7s}")
    ecrire("-" * 96)

    for libelle, col in CONTINUES:
        if col not in d.columns:
            ecrire(f"{libelle:38s} {'colonne absente':>16s}")
            continue
        a = pd.to_numeric(d.loc[d['Mode_transport'] == g1, col], errors='coerce').dropna()
        b = pd.to_numeric(d.loc[d['Mode_transport'] == g2, col], errors='coerce').dropna()
        if len(a) < 3 or len(b) < 3:
            ecrire(f"{libelle:38s} effectif insuffisant ({len(a)} / {len(b)})")
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        g, (lo, hi) = hedges_g(a, b)
        ecrire(f"{libelle:38s} {a.mean():7.1f} ({a.std():4.1f}) {b.mean():7.1f} "
               f"({b.std():4.1f}) {g:+6.2f} [{lo:+5.2f};{hi:+5.2f}] {p:7.3f}")
        lignes.append(dict(Variable=libelle, colonne=col,
                           n_1=len(a), moy_1=a.mean(), et_1=a.std(),
                           n_2=len(b), moy_2=b.mean(), et_2=b.std(),
                           g_Hedges=g, IC_bas=lo, IC_haut=hi, p=p))

    ecrire("")
    for libelle, col in CATEGORIELLES:
        if col not in d.columns:
            continue
        tab = pd.crosstab(d[col], d['Mode_transport'])
        chi2, p, ddl, _ = stats.chi2_contingency(tab)
        ecrire(f"{libelle} :")
        ecrire(tab.to_string())
        ecrire(f"  chi2({ddl}) = {chi2:.2f}, p = {p:.3f}")
        lignes.append(dict(Variable=libelle, colonne=col, p=p))

    res = pd.DataFrame(lignes)
    res.to_excel(os.path.join(OUTPUT_DIR, 'Comparabilite_groupes.xlsx'), index=False)

    ecrire("\n" + "-" * 96)
    cont = res[res['g_Hedges'].notna()] if 'g_Hedges' in res.columns else pd.DataFrame()
    if len(cont):
        gmax = cont.loc[cont['g_Hedges'].abs().idxmax()]
        ecrire(f"Ecart le plus marque : {gmax['Variable']} (g = {gmax['g_Hedges']:+.2f}, "
               f"p = {gmax['p']:.3f})")
        ecrire(f"Variables a |g| >= 0,50 : "
               f"{list(cont.loc[cont['g_Hedges'].abs() >= 0.5, 'Variable'])}")
        ecrire(f"Variables a p < 0,05    : "
               f"{list(cont.loc[cont['p'] < 0.05, 'Variable'])}")

    with open(os.path.join(OUTPUT_DIR, 'Rapport_comparabilite.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(rapport))
    print(f"\nSorties : {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
