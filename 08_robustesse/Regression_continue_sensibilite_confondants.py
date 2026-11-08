# -*- coding: utf-8 -*-
"""
Analyse de sensibilite du niveau 4 (volet en differences) aux facteurs
confondants variant entre trajets (chapitre 4, niveau 4).

Modele de reference (script principal, 3.9.6) :
    M0 : DeltaPerf ~ DeltaExpo + C(Mode_transport) + C(Ordre)
Modeles ajustes :
    M1 : M0 + C(Campagne) + DeltaFC            (FC moyenne du trajet, Garmin)
    M2 : M1 + DeltaL90                         (bruit de fond, campagne 2024 seule)
    M3 : M1 + DeltaStressConduite + DeltaAnxTrajet   (stress du trajet, 0-10 ; roue emotionnelle)
    M4 : M3 + DeltaFatigue + DeltaDiffConduite + DeltaDuree
Les scores de ressenti viennent du cahier de passation (Q_traitement_passation_all.py),
en difference Proche - Loin comme l'exposition.
M0 est aussi reajuste sur l'echantillon exact de M1 (M0_echM1) afin de separer
l'effet de l'ajustement de celui de la perte d'effectif.

Memes 31 indicateurs, 3 polluants, 2 transformations que le script principal
(import direct de sa configuration). FDR BH par famille polluant x
transformation x tache, comme au niveau 4.
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import Regression_continue_FDR_2niveau as R

OUTPUT_DIR = S(r'Resultats_regression_continue\26-10-26_confondants')
FC_CSV = D(r'Data\Traitement\Descriptives\FC_par_trajet_et_mode_codage_somme\FC_par_trajet.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(R.DATA_FILE)
df = df[df['Participant'].isin(R.SELECTION_4)].copy()
df = df.merge(R.charger_ordre(), on='Participant', how='left')
df['Campagne'] = np.where(df['Participant'] < 200, 'aut24', 'print25')
fc = pd.read_csv(FC_CSV, sep=';')
fc = fc.pivot_table(index='Participant', columns='Trajet', values='FC_moyenne')
fc['DeltaFC'] = fc['Close'] - fc['Away']
df = df.merge(fc[['DeltaFC']], left_on='Participant', right_index=True, how='left')
df['DeltaL90'] = df['Leq_90_C'] - df['Leq_90_A']
RESSENTI = {'DeltaStressConduite': 'stress_conduite', 'DeltaAnxTrajet': 'roue_diff_impact_trajet_Anxiete',
            'DeltaFatigue': 'fatigue', 'DeltaDiffConduite': 'difficulte_conduite',
            'DeltaDuree': 'Duree_trajet_minutes'}
for new, col in RESSENTI.items():
    df[new] = pd.to_numeric(df[f'{col}_C'], errors='coerce') - pd.to_numeric(df[f'{col}_A'], errors='coerce')
    print(f"{new} dispo {df[new].notna().sum()}")
COVS_EXTRA = list(RESSENTI)
print(f"N={len(df)} ; DeltaFC dispo {df.DeltaFC.notna().sum()} ; DeltaL90 dispo {df.DeltaL90.notna().sum()}")

inds = R.generer_indicateurs_nback() + R.generer_indicateurs_ant()
MODELES = {
    'M0': 'DeltaPerf ~ DeltaExpo + C(Mode_transport) + C(Ordre)',
    'M1': 'DeltaPerf ~ DeltaExpo + C(Mode_transport) + C(Ordre) + C(Campagne) + DeltaFC',
    'M2': 'DeltaPerf ~ DeltaExpo + C(Mode_transport) + C(Ordre) + C(Campagne) + DeltaFC + DeltaL90',
    'M3': 'DeltaPerf ~ DeltaExpo + C(Mode_transport) + C(Ordre) + C(Campagne) + DeltaFC + DeltaStressConduite + DeltaAnxTrajet',
    'M4': 'DeltaPerf ~ DeltaExpo + C(Mode_transport) + C(Ordre) + C(Campagne) + DeltaFC + DeltaStressConduite + DeltaAnxTrajet + DeltaFatigue + DeltaDiffConduite + DeltaDuree',
}


def fit(sub, formule):
    m = ols(formule, data=sub).fit()
    return dict(n=int(m.nobs), coef=m.params['DeltaExpo'], p=m.pvalues['DeltaExpo'], r2=m.rsquared)


rows = []
for pol in R.POLLUANTS:
    for ind in inds:
        for trans in R.TRANSFORMATIONS:
            sub = df[['Participant', ind['col_away'], ind['col_close'], pol['col_A'], pol['col_C'],
                      'Mode_transport', 'Ordre', 'Campagne', 'DeltaFC', 'DeltaL90'] + COVS_EXTRA].copy()
            sub.columns = ['Participant', 'PerfA', 'PerfC', 'ExpoA', 'ExpoC',
                           'Mode_transport', 'Ordre', 'Campagne', 'DeltaFC', 'DeltaL90'] + COVS_EXTRA
            if ind['transformation'] == 'log':
                sub['PerfA'] = R.appliquer_log(sub['PerfA'])
                sub['PerfC'] = R.appliquer_log(sub['PerfC'])
            if trans == 'log_expo':
                sub['ExpoA'] = R.appliquer_log(sub['ExpoA'])
                sub['ExpoC'] = R.appliquer_log(sub['ExpoC'])
            sub['DeltaPerf'] = sub['PerfC'] - sub['PerfA']
            sub['DeltaExpo'] = sub['ExpoC'] - sub['ExpoA']
            base = sub.dropna(subset=['DeltaPerf', 'DeltaExpo', 'Mode_transport', 'Ordre'])
            s1 = base.dropna(subset=['DeltaFC'])
            s2 = s1.dropna(subset=['DeltaL90'])
            s3 = s1.dropna(subset=['DeltaStressConduite', 'DeltaAnxTrajet'])
            s4 = s3.dropna(subset=['DeltaFatigue', 'DeltaDiffConduite', 'DeltaDuree'])
            row = dict(polluant=pol['nom'], indicateur=ind['nom'], tache=ind['tache'],
                       domaine=ind['domaine'], transformation=trans)
            for nom, s, formule in [('M0', base, MODELES['M0']), ('M0_echM1', s1, MODELES['M0']),
                                    ('M1', s1, MODELES['M1']), ('M2', s2, MODELES['M2']),
                                    ('M3', s3, MODELES['M3']), ('M4', s4, MODELES['M4'])]:
                try:
                    r = fit(s, formule)
                except Exception:
                    r = dict(n=len(s), coef=np.nan, p=np.nan, r2=np.nan)
                for k, v in r.items():
                    row[f'{nom}_{k}'] = v
            rows.append(row)

res = pd.DataFrame(rows)
for nom in ['M0', 'M0_echM1', 'M1', 'M2', 'M3', 'M4']:
    res[f'{nom}_q'] = np.nan
    for _, g in res.groupby(['polluant', 'transformation', 'tache']):
        ok = g[f'{nom}_p'].notna()
        if ok.sum():
            res.loc[g.index[ok], f'{nom}_q'] = multipletests(g.loc[ok, f'{nom}_p'], method='fdr_bh')[1]

lignes = []


def w(s=''):
    print(s)
    lignes.append(s)


w("SENSIBILITE DU VOLET EN DIFFERENCES AUX CONFONDANTS VARIANT ENTRE TRAJETS (26/10/2026)")
w(f"{len(res)} regressions par modele (31 indicateurs x 3 polluants x 2 transformations)")
w()
for nom, lib in [('M0', "M0 reference (mode + ordre), echantillon complet"),
                 ('M0_echM1', "M0 sur l'echantillon de M1"),
                 ('M1', "M1 = M0 + campagne + DeltaFC"),
                 ('M2', "M2 = M1 + DeltaL90 (bruit de fond, 2024 seule)"),
                 ('M3', "M3 = M1 + stress conduite + anxiete trajet"),
                 ('M4', "M4 = M3 + fatigue + difficulte conduite + duree")]:
    p, q, n = res[f'{nom}_p'], res[f'{nom}_q'], res[f'{nom}_n']
    w(f"{lib:55s} N {n.min():.0f}-{n.max():.0f} | p<0,05 : {(p < .05).sum():3d} | "
      f"0,05-0,10 : {((p >= .05) & (p < .10)).sum():3d} | q<0,05 : {(q < .05).sum()} | q<0,10 : {(q < .10).sum()}")
w()
for nom in ['M1', 'M3', 'M4']:
    w(f"Stabilite des p entre M0 (echantillon de M1) et {nom} :")
    w(f"  correlation de Spearman des p : {res['M0_echM1_p'].corr(res[f'{nom}_p'], method='spearman'):.3f}")
    w(f"  meme decision a 0,05 : {((res.M0_echM1_p < .05) == (res[f'{nom}_p'] < .05)).mean() * 100:.0f} %")
    w(f"  passent sous 0,05 : {((res.M0_echM1_p >= .05) & (res[f'{nom}_p'] < .05)).sum()} ; "
      f"sortent de 0,05 : {((res.M0_echM1_p < .05) & (res[f'{nom}_p'] >= .05)).sum()}")
    ratio = res[f'{nom}_coef'] / res.M0_echM1_coef
    w(f"  coefficient DeltaExpo : rapport {nom}/M0 median {ratio.median():.2f} "
      f"(quartiles {ratio.quantile(.25):.2f} - {ratio.quantile(.75):.2f})")
w()
w("Tests a p<0,05 sous M1 (tries) :")
cols = ['polluant', 'indicateur', 'transformation', 'M0_echM1_p', 'M1_n', 'M1_p', 'M1_q', 'M3_n', 'M3_p', 'M3_q', 'M4_n', 'M4_p', 'M4_q']
w(res[res.M1_p < .05].sort_values('M1_p')[cols].round(4).to_string(index=False))
w()
w("Tests a p<0,05 sous M4 et pas sous M0 (echantillon M1) :")
w(res[(res.M4_p < .05) & (res.M0_echM1_p >= .05)][cols].round(4).to_string(index=False))
w()
w("Tests a q<0,10 sous un modele quelconque :")
w(res[(res.M0_q < .10) | (res.M1_q < .10) | (res.M2_q < .10) | (res.M3_q < .10) | (res.M4_q < .10)][
    ['polluant', 'indicateur', 'transformation', 'M0_p', 'M0_q', 'M1_p', 'M1_q', 'M3_p', 'M3_q', 'M4_p', 'M4_q']
].round(4).to_string(index=False))
w()
w("Tendance P2 sonore / PM2.5 / log (q=0,076 au chapitre) :")
w(res[(res.indicateur == 'P2_Moy_sonore_Alerte_Standard_ANT') & (res.polluant == 'PM25_Mean')][
    ['transformation', 'M0_n', 'M0_p', 'M0_q', 'M1_n', 'M1_coef', 'M1_p', 'M1_q', 'M3_n', 'M3_coef', 'M3_p', 'M3_q', 'M4_n', 'M4_coef', 'M4_p', 'M4_q']
].round(4).to_string(index=False))

res.to_excel(os.path.join(OUTPUT_DIR, 'Sensibilite_confondants_voletD.xlsx'), index=False)
with open(os.path.join(OUTPUT_DIR, 'Rapport_sensibilite_confondants.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lignes))
print(f"\n-> {OUTPUT_DIR}")
