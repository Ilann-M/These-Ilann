# === CODAGE SOMME DES FACTEURS — corrige le 2026-09-09 ===
# Le modele Mode x Trajet etait ajuste en codage treatment (reference 'Bike'), de sorte que
# le terme lu comme 'effet principal du Trajet' etait l'effet du trajet CHEZ LES CYCLISTES
# CLASSIQUES — un effet simple. Le codage somme C(..., Sum) rend les termes marginaux.
# Les post-hocs par mode et les tailles d'effet (calculees sur les donnees) sont inchanges.
# ATTENTION : C(Trajet, Sum)[S.Away] a le signe OPPOSE a Trajet[T.Close].
# Sauvegarde de la version anterieure dans Data\Codes\_Archives\.
# -*- coding: utf-8 -*-
"""
Fréquence cardiaque par itinéraire et par mode de transport.

Contrôle de la manipulation de l'effort (chapitre 4, section 4.1.2).
Deux questions distinctes :

  1. L'effort diffère-t-il entre les deux ITINÉRAIRES ? Si oui, l'effort devient
     un facteur de confusion du contraste intra-participant, les deux trajets
     n'ayant ni la même vitesse moyenne ni le même profil d'arrêts.
  2. L'effort diffère-t-il entre les deux MODES ? C'est la validation du levier
     expérimental décrit en section 3.3.3, aujourd'hui étayée par les seules
     valeurs publiées dans Al Marj et al. (2026).

Source : les fichiers agrégés par trajet (un par session de test), qui portent la
colonne « Heart Rate » relevée à la seconde par la montre du participant. Chaque
fichier est apparié à un participant et à une session par sa date et son créneau,
via Base.xlsx — même clé (mois, jour, moment) que format_all3.py.

Sortie : Data\\Traitement\\Descriptives\\FC_par_trajet_et_mode
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402
import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm

warnings.filterwarnings('ignore')

DOSSIERS_AGGR = [
    D(r'Campagne_aut_24\Trajets\Aggregated'),
    D(r'Campagne_print_25\Trajets\Aggregated'),
]
BASE_XLSX = D(r'Data\Base.xlsx')
TAB_SUJET = D(r'Data\Table_combine\Tab_sujet_20260205.xlsx')
OUTPUT_DIR = S(r'Descriptives\FC_par_trajet_et_mode_codage_somme')

SELECTION_4 = [101, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
               201, 202, 203, 204, 205, 206, 209, 210, 211, 212, 215, 216,
               121, 122, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 139,
               221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]

_RX_FICHIER = re.compile(r'Data_(\d\d)-(\d\d)-(\d\d)_(\d+)h')
_RX_DATE = re.compile(r'\s*(\d{1,2})\s*/\s*(\d{1,2})\s*:?\s*(\d{1,2})\s*h')


def inventaire_trajets():
    """Un enregistrement par fichier agrege : statistiques de frequence cardiaque."""
    lignes = []
    for dossier in DOSSIERS_AGGR:
        for chemin in sorted(glob.glob(os.path.join(dossier, '*.xlsx'))):
            m = _RX_FICHIER.search(os.path.basename(chemin))
            if not m:
                continue
            mois, jour, heure = int(m.group(2)), int(m.group(3)), int(m.group(4))
            try:
                d = pd.read_excel(chemin)
            except Exception as e:
                print(f"  [illisible] {os.path.basename(chemin)} : {e}")
                continue
            col = next((c for c in d.columns if str(c).strip().lower() == 'heart rate'), None)
            if col is None:
                continue
            brut = pd.to_numeric(d[col], errors='coerce').dropna()
            # 255 est la valeur sentinelle « donnee invalide » du protocole Garmin ;
            # au-dela de 220 bpm la valeur n'est pas physiologiquement plausible.
            fc = brut[(brut > 0) & (brut < 220)]
            n_invalides = int(len(brut) - len(fc))
            # un enregistrement n'est retenu que s'il conserve au moins la moitie
            # de ses points et 100 mesures valides (environ 17 minutes)
            if len(fc) < 100 or (len(brut) and len(fc) / len(brut) < 0.5):
                print(f"  [ecarte] {os.path.basename(chemin)} : "
                      f"{len(fc)} points valides sur {len(brut)}")
                continue
            lignes.append(dict(
                fichier=os.path.basename(chemin), mois=mois, jour=jour,
                creneau='matin' if heure < 12 else 'aprem',
                FC_moyenne=fc.mean(), FC_mediane=float(np.median(fc)),
                FC_max=fc.max(), n_points=len(fc), n_invalides=n_invalides,
                duree_min=len(d) * 10 / 60.0))   # fichiers agreges au pas de 10 s
    return pd.DataFrame(lignes)


def inventaire_sessions():
    """Participant x session -> itineraire, date, creneau."""
    b = pd.read_excel(BASE_XLSX)
    b.columns = [str(c).strip() for c in b.columns]
    b['Participant'] = b['Participant'].astype(str).str.extract(r'[Pp](\d+)')[0]
    b = b.dropna(subset=['Participant'])
    b['Participant'] = b['Participant'].astype(int)
    col_p2 = 'Passation-2' if 'Passation-2' in b.columns else 'Passation_2'
    col_p3 = 'Passation-3' if 'Passation-3' in b.columns else 'Passation_3'

    lignes = []
    for _, r in b.iterrows():
        for passation, col_date, col_cond in (('P2', 'Date_P2', col_p2),
                                              ('P3', 'Date_P3', col_p3)):
            m = _RX_DATE.match(str(r[col_date]))
            if not m:
                continue
            jour, mois, heure = int(m.group(1)), int(m.group(2)), int(m.group(3))
            lignes.append(dict(
                Participant=int(r['Participant']), passation=passation,
                Trajet={'A': 'Away', 'C': 'Close'}.get(str(r[col_cond]).strip().upper()),
                mois=mois, jour=jour,
                creneau='matin' if heure < 12 else 'aprem'))
    return pd.DataFrame(lignes).dropna(subset=['Trajet'])


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    trajets = inventaire_trajets()
    print(f"{len(trajets)} fichiers agreges porteurs d'une frequence cardiaque")

    sessions = inventaire_sessions()
    print(f"{len(sessions)} sessions datees dans Base.xlsx")

    cles = ['mois', 'jour', 'creneau']
    dbl = int(trajets.duplicated(subset=cles, keep=False).sum())
    print(f"fichiers partageant date et creneau : {dbl}")

    d = sessions.merge(trajets, on=cles, how='left')
    print(f"sessions appariees : {d['FC_moyenne'].notna().sum()} / {len(d)}")

    sujets = pd.read_excel(TAB_SUJET)[['Participant', 'Mode_transport']]
    d = d.merge(sujets, on='Participant', how='left')
    d = d[d['Participant'].isin(SELECTION_4)].dropna(subset=['FC_moyenne', 'Mode_transport'])
    print(f"retenu : {len(d)} trajets chez {d['Participant'].nunique()} participants\n")
    d.to_csv(os.path.join(OUTPUT_DIR, 'FC_par_trajet.csv'), index=False, sep=';')

    rapport = []

    def ecrire(txt=''):
        print(txt)
        rapport.append(txt)

    ecrire("=" * 74)
    ecrire("FREQUENCE CARDIAQUE PAR ITINERAIRE ET PAR MODE")
    ecrire("=" * 74)
    ecrire(f"\n{len(d)} trajets, {d['Participant'].nunique()} participants "
           f"(selection de reference)\n")

    ecrire("-- Frequence cardiaque moyenne, en bpm : moyenne (ecart-type) --")
    ecrire(d.groupby(['Mode_transport', 'Trajet'])['FC_moyenne']
           .agg(['count', 'mean', 'std']).round(1).to_string())
    ecrire("\n-- Par itineraire, modes confondus --")
    ecrire(d.groupby('Trajet')['FC_moyenne'].agg(['count', 'mean', 'std']).round(1).to_string())
    ecrire("\n-- Par mode, itineraires confondus --")
    ecrire(d.groupby('Mode_transport')['FC_moyenne'].agg(['count', 'mean', 'std']).round(1).to_string())

    paires = d.groupby('Participant')['Trajet'].nunique()
    dd = d[d['Participant'].isin(paires[paires == 2].index)]
    ecrire(f"\n-- Modele mixte, {dd['Participant'].nunique()} participants "
           f"disposant des deux trajets --")
    ecrire("   FC_moyenne ~ C(Mode_transport, Sum) * C(Trajet, Sum), participant en effet aleatoire")
    ecrire("   (codage somme : les termes Trajet et Mode sont des effets principaux)")
    try:
        res = mixedlm("FC_moyenne ~ C(Mode_transport, Sum) * C(Trajet, Sum)", dd,
                      groups=dd['Participant']).fit(reml=True)
        for k in res.params.index:
            if k in ('Intercept', 'Group Var'):
                continue
            ecrire(f"   {k:44s} coef={res.params[k]:+8.2f}  p={res.pvalues[k]:.4f}")
    except Exception as e:
        ecrire(f"   ajustement impossible : {e}")

    large = dd.pivot_table(index='Participant', columns='Trajet',
                           values='FC_moyenne').dropna()
    diff = large['Close'] - large['Away']
    ecrire("\n-- Contraste intra-participant Close - Away --")
    ecrire(f"   n={len(diff)}  ecart moyen={diff.mean():+.2f} bpm  "
           f"ET={diff.std():.2f}  dz={diff.mean()/diff.std():+.3f}")
    # Ajout 2026-09-09 : le manuscrit citait le p du terme Trajet en codage treatment,
    # qui etait un effet simple. Le test apparie sur le contraste est la bonne reference.
    from scipy import stats as _st
    _t, _p = _st.ttest_rel(large["Close"], large["Away"])
    _w = _st.wilcoxon(large["Close"], large["Away"])[1]
    ecrire(f"   t apparie = {_t:+.3f}   p = {_p:.4f}   |   Wilcoxon p = {_w:.4f}")

    with open(os.path.join(OUTPUT_DIR, 'Rapport_FC.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rapport))
    print(f"\nSorties : {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
