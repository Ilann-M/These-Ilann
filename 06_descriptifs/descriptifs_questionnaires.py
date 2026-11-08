# -*- coding: utf-8 -*-
"""
Descriptifs des questionnaires de vie par mode de transport (Vélo vs VAE),
sur SELECTION_4, pour la caractérisation de l'échantillon (tableau 4.4 du chapitre 4 : moyennes,
tests de Mann-Whitney et test exact de Fisher pour le sexe ; les g de Hedges et leurs
intervalles viennent de Comparabilite_groupes_mode.py).
Sortie : tableau console + Data/Traitement/Descriptifs/Table_caracterisation_echantillon.csv
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402
import sys, os, re
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')

QDIR = D(r'Data/Cognition/Questionnaire')
TAB = D(r'Data/Table_combine/Tab_sujet_20260205.xlsx')
OUT = S(r'Descriptifs')
os.makedirs(OUT, exist_ok=True)

SELECTION_4 = [101,104,106,107,108,109,110,111,112,113,114,115,116,
               201,202,203,204,205,206,209,210,211,212,215,216,
               121,122,127,128,130,131,132,133,134,135,136,137,139,
               221,222,224,226,227,228,229,230,231,232,233,234,235]

q = pd.concat([pd.read_excel(f'{QDIR}/resultats_Q1-vie.xlsx'),
               pd.read_excel(f'{QDIR}/resultats_Q2-vie.xlsx')], ignore_index=True)
q['Participant'] = q['ID'].map(lambda x: int(re.search(r'(\d+)', str(x)).group(1)))
tab = pd.read_excel(TAB)[['Participant', 'Mode_transport']]
d = q.merge(tab, on='Participant', how='inner')
d = d[d['Participant'].isin(SELECTION_4)].copy()
d['Groupe'] = d['Mode_transport'].map({'Bike': 'Vélo', 'e-Bike': 'VAE'})

nB = (d['Groupe'] == 'Vélo').sum(); nE = (d['Groupe'] == 'VAE').sum()
print(f'N SELECTION_4 avec questionnaires : {len(d)}  (Vélo {nB}, VAE {nE})')
print(f"Sexe uniques: {sorted(d['Sexe'].dropna().unique())}")

CONT = [('Âge', 'Âge (ans)'), ('IMC', 'IMC (kg/m²)'), ('IMG', 'Masse grasse (%)'),
        ('APtot', 'Activité physique (MET-min/sem)'), ('MVPA', 'MVPA (min/sem)'),
        ('SB', 'Comportement sédentaire (min/j)'),
        ('Score_PSQI', 'PSQI (qualité sommeil)'), ('Dure_sommeil', 'Durée de sommeil (h)'),
        ('Score_PSS', 'PSS (stress perçu)'), ('Score_PHQ', 'PHQ-9 (dépression)'),
        ('Extraversion', 'TIPI Extraversion'), ('Agréabilité', 'TIPI Agréabilité'),
        ('Conscienciosité', 'TIPI Conscienciosité'), ('Stabilité_émotionnelle', 'TIPI Stabilité émot.'),
        ('Ouverture', 'TIPI Ouverture'),
        ('Score_Nutrition', 'Nutrition (score global)'),
        ('score_alcool', 'ASSIST alcool'), ('score_nicotine', 'ASSIST nicotine'),
        ('score_cannabis', 'ASSIST cannabis'),
        ('expo_total', 'Exposition chronique (score)')]

rows = []
for col, lbl in CONT:
    if col not in d.columns: continue
    b = d.loc[d['Groupe'] == 'Vélo', col].dropna()
    e = d.loc[d['Groupe'] == 'VAE', col].dropna()
    if len(b) < 3 or len(e) < 3: continue
    try: p = stats.mannwhitneyu(b, e, alternative='two-sided').pvalue
    except Exception: p = np.nan
    rows.append({'Variable': lbl,
                 'Vélo': f'{b.mean():.1f} ({b.std(ddof=1):.1f})',
                 'VAE': f'{e.mean():.1f} ({e.std(ddof=1):.1f})',
                 'p': f'{p:.3f}', 'p_num': p})

# Sexe (catégoriel) — Fisher, sur les 51 et non sur les 50 répondants
# Le participant 228 n'a rempli aucun questionnaire de mode de vie : il est absent
# de « d », qui ne connaît que les 50 répondants. Son sexe est en revanche connu et
# consigné dans Tab_sujet (report documenté dans Data/Codes/format_all3.py). La
# répartition par sexe est donc établie depuis Tab_sujet, sur les 51 participants de
# SELECTION_4 ; toutes les autres lignes du tableau restent sur les 50 répondants.
sexe_src = pd.read_excel(TAB)[['Participant', 'Mode_transport', 'Sexe']]
sexe_src = sexe_src[sexe_src['Participant'].isin(SELECTION_4)].copy()
sexe_src['Groupe'] = sexe_src['Mode_transport'].map({'Bike': 'Vélo', 'e-Bike': 'VAE'})
assert sexe_src['Sexe'].notna().all(), (
    'sexe manquant pour : '
    + str(sexe_src.loc[sexe_src['Sexe'].isna(), 'Participant'].tolist()))
sx = pd.crosstab(sexe_src['Groupe'], sexe_src['Sexe'])
try:
    if sx.shape == (2, 2): pf = stats.fisher_exact(sx.values)[1]
    else: pf = stats.chi2_contingency(sx.values)[1]
except Exception: pf = np.nan

# La ligne Sexe entre dans le CSV : aucune valeur du tableau 4.4 n'est recopiée à la main.
ligne_sexe = {'Variable': 'Sexe (femmes / hommes), n',
              'Vélo': f"{sx.loc['Vélo', 'Femme']} / {sx.loc['Vélo', 'Homme']}",
              'VAE': f"{sx.loc['VAE', 'Femme']} / {sx.loc['VAE', 'Homme']}",
              'p': f'{pf:.3f}', 'p_num': pf}
res = pd.DataFrame([ligne_sexe] + rows)
print("\n=== Caractérisation de l'échantillon (moyenne (ET)) ===")
print(f'{"Variable":34} {"Vélo":>14} {"VAE":>14} {"p":>7}')
for _, r in res.iterrows():
    flag = ' *' if (pd.notna(r['p_num']) and r['p_num'] < 0.05) else ''
    print(f'{r["Variable"]:34} {r["Vélo"]:>14} {r["VAE"]:>14} {r["p"]:>7}{flag}')
print(f'\nSexe (effectifs par groupe, sur les {len(sexe_src)} de SELECTION_4) :'
      f'\n{sx}\nFisher p = {pf:.3f}')
print(f'\nVariables avec p<0.05 (différence de groupe) : '
      f'{[r["Variable"] for _,r in res.iterrows() if pd.notna(r["p_num"]) and r["p_num"]<0.05] or "aucune"}')

res.drop(columns='p_num').to_csv(f'{OUT}/Table_caracterisation_echantillon.csv', index=False, encoding='utf-8-sig')
print(f'\nCSV -> {OUT}/Table_caracterisation_echantillon.csv')
