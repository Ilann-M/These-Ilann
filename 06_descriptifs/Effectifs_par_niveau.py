# -*- coding: utf-8 -*-
r"""
EFFECTIFS ANALYSÉS PAR INDICATEUR ET PAR NIVEAU (tableau 4.1 du chapitre 4)

Section 4.1.1 « Effectifs analysés » : tableau du nombre de participants retenus (paires complètes), par tâche, domaine, niveau d'analyse et mode, plus les
mesures d'exposition et d'effort.

Règles reproduites (aucune analyse refaite, on compte seulement) :
  - sélection SELECTION_4 (N = 51), catalogue des 31 indicateurs de LMM_intention.py (paires complètes Loin / Proche),
    participant 134 retiré du TR et du d′ du 2-back (comme LMM_intention.py du 07/10) ;
  - niveau 2 : C ≥ 1,1 × A sur la concentration moyenne du marqueur (LMM_corr_FDR_2niveau_2.py, RATIO_MIN = 1.1) ;
  - niveaux 3 et volet en différences du niveau 4 : paires complètes du marqueur (réétiquetage, aucune égalité) ;
  - mode : 100-199 vélo classique, 200-299 VAE.
Le script vérifie qu'il retrouve les N publiés au chapitre (TR 2-back 50 / 43 / 37 / 48 / 46 ; N1 1-back 47 / 45 / 44 ;
P3 2-back 44) avant d'écrire quoi que ce soit.
Sortie : Data\Traitement\Descriptives\Effectifs_par_niveau_20261017.xlsx (détail par indicateur + synthèse par groupe).
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402
import os
import sys
import importlib.util

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
LMM = _os.path.join(DEPOT, '07_modeles', 'LMM_intention.py')
spec = importlib.util.spec_from_file_location('lmm_intention', LMM)
lmm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lmm)          # garde __main__ : rien n'est exécuté

TAB = D(r'Data\Table_combine\Tab_sujet_20260205.xlsx')
OUT = S(r'Descriptives\Effectifs_par_niveau_20261017.xlsx')
SEL = lmm.SELECTION_4 if hasattr(lmm, 'SELECTION_4') else None
if SEL is None:
    SEL = [101, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
           201, 202, 203, 204, 205, 206, 209, 210, 211, 212, 215, 216,
           121, 122, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 139,
           221, 222, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235]
assert len(SEL) == 51

t = pd.read_excel(TAB)
t = t[t['Participant'].isin(SEL)].set_index('Participant')
assert len(t) == 51
# 134 : TR et d′ du 2-back éloigné (boîtier défaillant)
for c in t.columns:
    if c.endswith('_A_2B') and (c.startswith('d_prime_') or c.startswith('RT_moyen')):
        t.loc[134, c] = np.nan


def paired(a, c):
    return set(t.index[t[a].notna() & t[c].notna()])


MARQ = {'BC': 'BC880', 'PUF': 'Number', 'PM25': 'PM2.5'}
expo_pairs, filt = {}, {}
for k, pre in MARQ.items():
    a, c = f'{pre}_Mean_A_EXP', f'{pre}_Mean_C_EXP'
    assert a in t.columns and c in t.columns, (a, c)
    ok = paired(a, c)
    expo_pairs[k] = ok
    filt[k] = {p for p in ok if t.loc[p, c] >= 1.1 * t.loc[p, a]}
print('paires exposition :', {k: len(v) for k, v in expo_pairs.items()}, '| filtre 1,1 :', {k: len(v) for k, v in filt.items()})

cat = lmm.generer_catalogue()
rows = []
for ind in cat:
    base = paired(ind['col_away'], ind['col_close'])
    r = {'indicateur': ind['nom'], 'tache': ind['tache'], 'domaine': ind['domaine'], 'N1': base}
    r['N2_BC'] = base & filt['BC']
    r['N2_PUF'] = base & filt['PUF']
    r['N34_BC'] = base & expo_pairs['BC']
    r['N34_PUF'] = base & expo_pairs['PUF']
    r['N4_PM25'] = base & expo_pairs['PM25']
    rows.append(r)

LEVELS = ['N1', 'N2_BC', 'N2_PUF', 'N34_BC', 'N34_PUF', 'N4_PM25']
fmt = lambda s: f"{len(s)} ({len([p for p in s if p < 200])} / {len([p for p in s if p >= 200])})"
det = pd.DataFrame([{**{k: r[k] for k in ('indicateur', 'tache', 'domaine')}, **{L: fmt(r[L]) for L in LEVELS}} for r in rows])

# contrôles contre le chapitre
def n(nom, L):
    return len([r for r in rows if r['indicateur'] == nom][0][L])
attendu = {('RT_moyen_2B', 'N1'): 50, ('RT_moyen_2B', 'N2_BC'): 43, ('RT_moyen_2B', 'N2_PUF'): 37,
           ('RT_moyen_2B', 'N34_BC'): 48, ('RT_moyen_2B', 'N34_PUF'): 46,
           ('N1_Moy_Valide_1B', 'N1'): 47, ('N1_Moy_Valide_1B', 'N34_BC'): 45, ('N1_Moy_Valide_1B', 'N34_PUF'): 44,
           ('P3_Moy_Valide_2B', 'N1'): 44}
for (nom, L), v in attendu.items():
    got = n(nom, L)
    print(f'  contrôle {nom} {L} : {got} (attendu {v})')
    assert got == v, (nom, L, got, v)

# synthèse par groupe (tâche × domaine) : min-max des N si les indicateurs d'un groupe diffèrent
GROUPES = [('1B', 'Comportement', 'N-back 1-back, comportement'), ('1B', 'EEG', 'N-back 1-back, potentiels évoqués'),
           ('2B', 'Comportement', 'N-back 2-back, comportement'), ('2B', 'EEG', 'N-back 2-back, potentiels évoqués'),
           ('ANT', 'Comportement', 'ANT-I, comportement'), ('ANT', 'EEG', 'ANT-I, potentiels évoqués')]
syn = []
for tache, dom, lab in GROUPES:
    sub = [r for r in rows if r['tache'] == tache and r['domaine'] == dom]
    ligne = {'groupe': lab, 'n_indicateurs': len(sub)}
    for L in LEVELS:
        tot = sorted({len(r[L]) for r in sub})
        if len(tot) == 1:
            ligne[L] = fmt(sub[0][L])
        else:
            ligne[L] = f'{tot[0]} à {tot[-1]}'
    syn.append(ligne)
syn = pd.DataFrame(syn)

# mesures d'exposition et d'effort (paires complètes)
env = []
for k, lab in (('BC', 'Carbone suie'), ('PUF', 'Particules ultrafines'), ('PM25', 'PM2.5')):
    env.append({'mesure': lab, 'N': fmt(expo_pairs[k])})
fc = [c for c in t.columns if c.lower().startswith(('fc_', 'hr_', 'fc')) and ('_A' in c or '_C' in c)]
env_df = pd.DataFrame(env)
print('\nSYNTHÈSE\n', syn.to_string(index=False))
print('\nEXPOSITION\n', env_df.to_string(index=False))
print('colonnes FC candidates :', fc[:10])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with pd.ExcelWriter(OUT) as w:
    syn.to_excel(w, sheet_name='synthese', index=False)
    det.to_excel(w, sheet_name='detail', index=False)
    env_df.to_excel(w, sheet_name='exposition', index=False)
print('écrit :', OUT)
