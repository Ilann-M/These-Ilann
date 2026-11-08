# === CODAGE SOMME DES FACTEURS — corrige le 2026-09-09 ===
# Le modele Mode x Trajet etait ajuste en codage treatment (reference 'Bike'), de sorte que
# le terme lu comme 'effet principal du Trajet' etait l'effet du trajet CHEZ LES CYCLISTES
# CLASSIQUES — un effet simple. Le codage somme C(..., Sum) rend les termes marginaux.
# Les post-hocs par mode et les tailles d'effet (calculees sur les donnees) sont inchanges.
# ATTENTION : C(Trajet, Sum)[S.Away] a le signe OPPOSE a Trajet[T.Close].
# Sauvegarde de la version anterieure dans Data\Codes\_Archives\.
# -*- coding: utf-8 -*-
"""
Contrôle de la manipulation par les mesures subjectives (section 3.4.8).

Cinq indicateurs, recueillis à l'issue de chaque trajet :

  - difficulté ressentie du trajet   (échelle 0-10)
  - stress perçu pendant le trajet   (échelle 0-10)
  - fatigue                          (échelle 0-10)  — contrôle négatif attendu
  - affect négatif induit par le trajet   composite de 5 familles de la roue
  - affect positif induit par le trajet   composite de 5 familles de la roue

Les composites de la roue de Genève agrègent les différences « impact trajet »,
définies comme la cotation à l'arrivée moins la cotation après le trajet. Le
signe se lit donc : une valeur négative signifie que l'émotion a AUGMENTÉ au
cours du trajet.

Modèle identique au reste du chapitre : Valeur ~ Mode_transport * Trajet +
Session, ordonnée à l'origine aléatoire, participant en effet aléatoire. Les
échelles étant ordinales et bornées, un test des rangs signés de Wilcoxon
accompagne chaque effet. Correction FDR sur la famille des cinq indicateurs.

Sortie : Data\\Traitement\\Controles_manipulation
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
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

TABLE = D(r"Data\Table_combine\Tab_passation_20260205.xlsx")
OUTPUT_DIR = S(r"Controles_manipulation")
ALPHA = 0.05

SELECTION_4 = [101, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
               201, 202, 203, 204, 205, 206, 209, 210, 211, 212, 215, 216,
               121, 122, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 139,
               221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]

NEGATIVES = ['Anxiete', 'Colere', 'Ennui', 'Peur', 'Tristesse']
POSITIVES = ['Compassion', 'Energie', 'Interet', 'Joie', 'Serenite']

INDICATEURS = [
    ('difficulte_conduite', "difficulté ressentie du trajet"),
    ('stress_conduite', "stress perçu pendant le trajet"),
    ('fatigue', "fatigue"),
    ('affect_negatif', "affect négatif induit par le trajet"),
    ('affect_positif', "affect positif induit par le trajet"),
]


def charger():
    d = pd.read_excel(TABLE)
    d = d[d['Participant'].isin(SELECTION_4) & d['Passation'].isin([2, 3])].copy()

    for nom, familles in (('affect_negatif', NEGATIVES), ('affect_positif', POSITIVES)):
        cols = [f'roue_diff_impact_trajet_{f}' for f in familles]
        presentes = [c for c in cols if c in d.columns]
        if len(presentes) < len(cols):
            print(f"  [!] {nom} : {len(presentes)}/{len(cols)} familles trouvées")
        d[nom] = d[presentes].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    d['Trajet'] = d['Etat_Passation'].map({'A': 'Away', 'C': 'Close'})
    d['Session'] = np.where(d['Passation'] == 2, 'S2', 'S3')
    return d.dropna(subset=['Trajet'])


def analyser(d, col):
    sub = d[['Participant', 'Mode_transport', 'Trajet', 'Session', col]].dropna()
    sub[col] = pd.to_numeric(sub[col], errors='coerce')
    sub = sub.dropna()
    # paires completes uniquement
    paires = sub.groupby('Participant')['Trajet'].nunique()
    sub = sub[sub['Participant'].isin(paires[paires == 2].index)]
    if sub['Participant'].nunique() < 10:
        return None

    large = sub.pivot_table(index='Participant', columns='Trajet', values=col)
    diff = (large['Close'] - large['Away']).dropna()
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    try:
        p_wil = stats.wilcoxon(large['Close'], large['Away'])[1]
    except Exception:
        p_wil = np.nan

    sub = sub.rename(columns={col: 'Valeur'})
    p_lmm, coef = np.nan, np.nan
    try:
        res = mixedlm("Valeur ~ C(Mode_transport, Sum) * C(Trajet, Sum) + Session", sub,
                      groups=sub['Participant']).fit(reml=True)
        k = [x for x in res.params.index
             if 'Trajet' in x and ':' not in x and 'Session' not in x]
        if k:
            # C(Trajet, Sum)[S.Away] a le signe oppose a Trajet[T.Close] :
            # on le retourne pour conserver la convention 'Close - Away'.
            coef, p_lmm = -res.params[k[0]], res.pvalues[k[0]]
    except Exception:
        pass

    return dict(n=sub['Participant'].nunique(),
                moy_away=large['Away'].mean(), moy_close=large['Close'].mean(),
                et_away=large['Away'].std(ddof=1), et_close=large['Close'].std(ddof=1),
                diff=diff.mean(), dz=dz, coef=coef, p_lmm=p_lmm, p_wilcoxon=p_wil)


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    d = charger()
    print(f"{d['Participant'].nunique()} participants, {len(d)} sessions de test\n")

    lignes = []
    for col, libelle in INDICATEURS:
        if col not in d.columns:
            print(f"  [absent] {col}")
            continue
        r = analyser(d, col)
        if r is None:
            print(f"  [insuffisant] {col}")
            continue
        r.update(indicateur=col, libelle=libelle)
        lignes.append(r)
        print(f"  {libelle:38s} n={r['n']:3d}  "
              f"A={r['moy_away']:5.2f}  C={r['moy_close']:5.2f}  "
              f"dz={r['dz']:+.2f}  p={r['p_lmm']:.4f}")

    res = pd.DataFrame(lignes)
    for col_p in ('p_lmm', 'p_wilcoxon'):
        m = res[col_p].notna()
        res.loc[m, 'q_' + col_p] = multipletests(res.loc[m, col_p], method='fdr_bh')[1]
    res.to_excel(os.path.join(OUTPUT_DIR, 'Controles_manipulation.xlsx'), index=False)

    chemin = os.path.join(OUTPUT_DIR, 'Rapport_controles_manipulation.txt')
    with open(chemin, 'w', encoding='utf-8') as f:
        f.write("CONTROLE DE LA MANIPULATION PAR LES MESURES SUBJECTIVES\n")
        f.write("=" * 78 + "\n\n")
        f.write("Modele : Valeur ~ Mode_transport * Trajet + Session, intercept aleatoire.\n")
        f.write("Wilcoxon apparie en complement (echelles ordinales bornees).\n")
        f.write(f"Correction FDR sur la famille des {len(res)} indicateurs. Seuil {ALPHA}.\n\n")
        f.write("Roue de Genève : difference = cotation a l'arrivee moins cotation apres\n")
        f.write("le trajet ; une valeur negative signifie une AUGMENTATION sur le trajet.\n\n")
        for _, r in res.iterrows():
            f.write(f"--- {r['libelle']} (n = {int(r['n'])})\n")
            f.write(f"    éloigné : {r['moy_away']:.2f} ({r['et_away']:.2f})   "
                    f"proche : {r['moy_close']:.2f} ({r['et_close']:.2f})\n")
            f.write(f"    écart proche − éloigné : {r['diff']:+.2f}   dz = {r['dz']:+.2f}\n")
            f.write(f"    modèle mixte : p = {r['p_lmm']:.4f}  (q = {r['q_p_lmm']:.4f})"
                    f"   Wilcoxon : p = {r['p_wilcoxon']:.4f}  (q = {r['q_p_wilcoxon']:.4f})\n\n")

    print("\n" + open(chemin, encoding='utf-8').read())
    print(f"Sorties : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
