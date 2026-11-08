# -*- coding: utf-8 -*-
"""
Analyse de sensibilité de puissance (chapitre 3, section 3.2.4).

L'ampleur de l'effet d'une exposition de quarante minutes n'étant pas
caractérisée dans la littérature, aucun calcul de puissance a priori n'était
réalisable : le dimensionnement relève d'un cadre « resource-constrained »
(Lakens, 2022). Ce que l'on peut et doit rapporter dans ce cas, c'est la
SENSIBILITE du dispositif, c'est-à-dire le plus petit effet qu'il est capable
de détecter avec une puissance donnée.

Trois effets, trois structures de test distinctes :

  - Trajet (intra-participants)  -> test apparié sur N paires ; l'indice est le
                                    dz (écart-type des différences appariées)
  - Mode (inter-participants)    -> test à échantillons indépendants,
                                    26 vs 25 ; l'indice est le d de Cohen
  - Interaction Mode x Trajet    -> test à échantillons indépendants SUR LES
                                    SCORES DE DIFFERENCE intra-participants,
                                    26 vs 25 ; l'indice est la différence
                                    standardisée entre les effets de trajet des
                                    deux modes

Deux seuils sont considérés : le seuil nominal de 5 %, et le seuil effectif que
la correction de Benjamini-Hochberg impose au plus petit p d'une famille de huit
tests (0,05 / 8 = 0,00625), qui borne la sensibilité de l'analyse principale.

Sortie : Data\\Traitement\\Resultats_LMM_intention\\26-08-12_sensibilite_puissance
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestPower, TTestIndPower

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from LMM_intention import (SELECTION_4, DATA_FILE, generer_catalogue,
                           charger_donnees, filtrer_selection,
                           preparer_donnees_long)

warnings.filterwarnings('ignore')

OUTPUT_DIR = S(r'Resultats_LMM_intention'
              r'\26-08-12_sensibilite_puissance')

PUISSANCES = [0.80, 0.90]
ALPHAS = {'nominal 5 %': 0.05,
          'seuil BH famille N-Back (8 tests)': 0.05 / 8}
N_GROUPE_1, N_GROUPE_2 = 26, 25       # vélo classique / VAE


def effectifs_observes():
    """N de paires completes par indicateur, sur l'echantillon analyse."""
    df = filtrer_selection(charger_donnees(DATA_FILE), SELECTION_4)
    lignes = []
    for indic in generer_catalogue():
        ca, cc = indic['col_away'], indic['col_close']
        if ca not in df.columns or cc not in df.columns:
            continue
        lg, _ = preparer_donnees_long(df, ca, cc, indic.get('transformation'))
        if lg is None:
            continue
        n = lg['Participant'].nunique()
        # dz observe = moyenne des differences / ecart-type des differences
        large = lg.pivot_table(index='Participant', columns='Trajet', values='Valeur')
        dz = np.nan
        if {'Away', 'Close'} <= set(large.columns):
            d = (large['Close'] - large['Away']).dropna()
            if len(d) > 1 and d.std(ddof=1) > 0:
                dz = d.mean() / d.std(ddof=1)
        lignes.append(dict(Indicateur=indic['nom'], Famille=indic['categorie'],
                           N=n, dz_observe=dz))
    return pd.DataFrame(lignes)


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    obs = effectifs_observes()
    obs.to_excel(os.path.join(OUTPUT_DIR, 'Effectifs_et_dz_observes.xlsx'), index=False)

    n_med = int(obs['N'].median())
    n_min, n_max = int(obs['N'].min()), int(obs['N'].max())
    print(f"Effectifs par indicateur : mediane {n_med}, etendue {n_min}-{n_max}\n")

    ap, ai = TTestPower(), TTestIndPower()
    lignes = []

    for lib_alpha, alpha in ALPHAS.items():
        for puiss in PUISSANCES:
            # --- effet Trajet : test apparie, pour plusieurs effectifs ---
            for n in sorted({n_min, n_med, n_max, 51}):
                dz = ap.solve_power(nobs=n, alpha=alpha, power=puiss,
                                    alternative='two-sided')
                lignes.append(dict(Effet='Trajet (intra)', Indice='dz', Seuil=lib_alpha,
                                   alpha=alpha, Puissance=puiss, N=n,
                                   Effet_detectable=dz))

            # --- effet Mode : echantillons independants ---
            d = ai.solve_power(nobs1=N_GROUPE_1, ratio=N_GROUPE_2 / N_GROUPE_1,
                               alpha=alpha, power=puiss, alternative='two-sided')
            lignes.append(dict(Effet='Mode (inter)', Indice='d', Seuil=lib_alpha,
                               alpha=alpha, Puissance=puiss,
                               N=f'{N_GROUPE_1}+{N_GROUPE_2}', Effet_detectable=d))

            # --- interaction : test independant sur les scores de difference ---
            di = ai.solve_power(nobs1=N_GROUPE_1, ratio=N_GROUPE_2 / N_GROUPE_1,
                                alpha=alpha, power=puiss, alternative='two-sided')
            lignes.append(dict(Effet='Interaction Mode x Trajet', Indice='d sur differences',
                               Seuil=lib_alpha, alpha=alpha, Puissance=puiss,
                               N=f'{N_GROUPE_1}+{N_GROUPE_2}', Effet_detectable=di))

    res = pd.DataFrame(lignes)
    res.to_excel(os.path.join(OUTPUT_DIR, 'Sensibilite_puissance.xlsx'), index=False)

    # puissance atteinte pour l'effet principal observe
    principal = obs[obs['Indicateur'] == 'N1_Moy_Valide_1B']
    puiss_obs = {}
    if len(principal):
        n_p = int(principal['N'].iloc[0])
        dz_p = float(principal['dz_observe'].iloc[0])
        for lib_alpha, alpha in ALPHAS.items():
            puiss_obs[lib_alpha] = ap.solve_power(effect_size=abs(dz_p), nobs=n_p,
                                                  alpha=alpha, power=None,
                                                  alternative='two-sided')

    # ------------------------------------------------------------------ rapport
    with open(os.path.join(OUTPUT_DIR, 'Rapport_sensibilite_puissance.txt'), 'w',
              encoding='utf-8') as f:
        f.write("ANALYSE DE SENSIBILITE DE PUISSANCE\n")
        f.write("=" * 78 + "\n\n")
        f.write("Plus petit effet detectable, test bilateral.\n")
        f.write(f"Effectifs par indicateur : mediane {n_med}, etendue {n_min}-{n_max}.\n")
        f.write(f"Groupes de mode : {N_GROUPE_1} vs {N_GROUPE_2}.\n\n")
        for lib_alpha, alpha in ALPHAS.items():
            f.write(f"--- Seuil : {lib_alpha} (alpha = {alpha:.5f}) ---\n")
            for puiss in PUISSANCES:
                f.write(f"  Puissance {puiss:.0%} :\n")
                s = res[(res['Seuil'] == lib_alpha) & (res['Puissance'] == puiss)]
                for _, r in s.iterrows():
                    f.write(f"    {r['Effet']:26s} N={str(r['N']):7s} "
                            f"{r['Indice']:18s} = {r['Effet_detectable']:.3f}\n")
            f.write("\n")
        if puiss_obs:
            f.write("--- Puissance atteinte pour l'effet principal observe ---\n")
            f.write(f"  N1_Moy_Valide_1B : N={n_p}, dz observe = {dz_p:+.3f}\n")
            for lib_alpha, p in puiss_obs.items():
                f.write(f"    puissance sous {lib_alpha} : {p:.1%}\n")

    print(open(os.path.join(OUTPUT_DIR, 'Rapport_sensibilite_puissance.txt'),
               encoding='utf-8').read())
    print(f"Sorties : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
