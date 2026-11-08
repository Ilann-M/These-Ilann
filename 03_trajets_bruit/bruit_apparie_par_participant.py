# -*- coding: utf-8 -*-
"""
Bruit (LAeq) par trajet, en test APPARIE par participant — Ch4 section 4.1.

Le descriptif existant (descriptif_bruit_par_trajet.py) compare les 76
enregistrements NoiseCapture comme deux groupes independants. Or le plan est
intra-participant : chaque participant parcourt les deux itineraires, et
l'article 1 conduit d'ailleurs le test sur des paires (Wilcoxon apparie).
Ce script reconstruit l'appariement.

Chaine : Base.xlsx donne, pour chaque participant, la date et le creneau de ses
deux passations et l'itineraire correspondant (Passation-2/3 = A/C). Le CSV
LAeq_par_track.csv donne le LAeq energetique de chaque enregistrement, date et
creneau. La jointure sur (date, creneau) attribue a chaque participant un LAeq
pour son trajet eloigne et un pour son trajet proche ; les participants dont les
deux trajets sont documentes forment les paires analysees.

Sortie : Data/Traitement/Descriptifs/bruit_apparie/
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402
import os
import re
import io
import numpy as np
import pandas as pd
from scipy import stats

BASE = D(r'Data\Base.xlsx')
LAEQ = D(r'Data\Traitement\Descriptifs\LAeq_par_track.csv')
LMM_SRC = _os.path.join(DEPOT, '07_modeles', 'LMM_intention.py')
OUT = S(r'Descriptifs\bruit_apparie')


def campagne_annee(pid):
    if (100 < pid < 121) or (200 < pid < 221):
        return 2024
    if (120 < pid < 141) or (220 < pid < 241):
        return 2025
    return None


def selection_4():
    src = io.open(LMM_SRC, encoding='utf-8').read()
    m = re.search(r'SELECTION_4\s*=\s*\[(.*?)\]', src, re.S)
    return set(int(x) for x in re.findall(r'\d+', m.group(1)))


def sessions_par_participant():
    """[(pid, annee, mois, jour, slot, trajet), ...] depuis Base.xlsx."""
    b = pd.read_excel(BASE)
    out = []
    for _, r in b.iterrows():
        m0 = re.search(r'(\d+)', str(r['Participant']))
        if not m0:
            continue
        pid = int(m0.group(1))
        yr = campagne_annee(pid)
        if yr is None:
            continue
        for dcol, pcol in [('Date_P2', 'Passation-2'), ('Date_P3', 'Passation-3')]:
            ds, pas = r.get(dcol), r.get(pcol)
            if pd.isna(ds) or pd.isna(pas):
                continue
            m = re.search(r'(\d+)\s*/\s*(\d+).*?(\d+)\s*h', str(ds))
            if not m:
                continue
            day, mon, hour = int(m.group(1)), int(m.group(2)), int(m.group(3))
            slot = 'AM' if hour < 12 else 'PM'
            traj = 'Proche' if str(pas).strip().upper() == 'C' else 'Loin'
            out.append((pid, yr, mon, day, slot, traj))
    return pd.DataFrame(out, columns=['Participant', 'annee', 'mois', 'jour', 'slot', 'trajet'])


def main():
    os.makedirs(OUT, exist_ok=True)
    lignes = []

    def p(t=''):
        print(t)
        lignes.append(str(t))

    laeq = pd.read_csv(LAEQ)
    d = pd.to_datetime(laeq['date'])
    laeq['annee'], laeq['mois'], laeq['jour'] = d.dt.year, d.dt.month, d.dt.day
    # un enregistrement par (date, creneau) : moyenne energetique si plusieurs
    par_session = (laeq.groupby(['annee', 'mois', 'jour', 'slot'])['laeq']
                        .apply(lambda s: 10 * np.log10(np.mean(10 ** (np.asarray(s) / 10))))
                        .reset_index())

    sess = sessions_par_participant()
    m = sess.merge(par_session, on=['annee', 'mois', 'jour', 'slot'], how='left')

    wide = m.pivot_table(index='Participant', columns='trajet', values='laeq')
    wide = wide.dropna(subset=[c for c in ('Loin', 'Proche') if c in wide.columns])

    sel = selection_4()
    wide_sel = wide[wide.index.isin(sel)]

    p('=' * 74)
    p('  BRUIT (LAeq) PAR TRAJET — TEST APPARIE PAR PARTICIPANT')
    p('=' * 74)
    p('  Enregistrements NoiseCapture      : %d' % len(laeq))
    p('  Sessions (date, creneau) distinctes : %d' % len(par_session))
    p('  Participants avec les deux trajets  : %d' % len(wide))
    p('  Dont dans SELECTION_4               : %d' % len(wide_sel))
    p()

    for lab, g in (('tous participants apparies', wide), ('SELECTION_4', wide_sel)):
        if len(g) < 3:
            continue
        a, c = g['Loin'].values, g['Proche'].values
        diff = c - a
        t, p_t = stats.ttest_rel(c, a)
        try:
            _, p_w = stats.wilcoxon(c, a)
        except Exception:
            p_w = float('nan')
        dz = diff.mean() / diff.std(ddof=1)
        p('  [%s] n = %d paires' % (lab, len(g)))
        p('    Loin   : %.1f dB(A)  (ET %.1f)' % (a.mean(), a.std(ddof=1)))
        p('    Proche : %.1f dB(A)  (ET %.1f)' % (c.mean(), c.std(ddof=1)))
        p('    Difference Proche - Loin : %+.2f dB(A)  (ET %.2f)' % (diff.mean(), diff.std(ddof=1)))
        p('    t apparie : t = %.3f, p = %.4f   |   Wilcoxon apparie : p = %.4f' % (t, p_t, p_w))
        p('    dz = %+.3f' % dz)
        p('    Participants ou le trajet Proche est MOINS bruyant : %d (%.0f %%)'
          % ((diff < 0).sum(), 100.0 * (diff < 0).mean()))
        p()

    # rappel du test non apparie actuellement rapporte dans le chapitre
    g1 = laeq[laeq['trajet'] == 'Loin']['laeq']
    g2 = laeq[laeq['trajet'] == 'Proche']['laeq']
    u, p_u = stats.mannwhitneyu(g2, g1, alternative='two-sided')
    p('  [rappel] test NON apparie sur les %d enregistrements (version actuelle du chapitre)' % len(laeq))
    p('    Loin n=%d : %.1f dB(A)   Proche n=%d : %.1f dB(A)   Mann-Whitney p = %.4f'
      % (len(g1), g1.mean(), len(g2), g2.mean(), p_u))

    wide.to_excel(os.path.join(OUT, 'LAeq_apparie_par_participant.xlsx'))
    io.open(os.path.join(OUT, 'Rapport_bruit_apparie.txt'), 'w', encoding='utf-8').write('\n'.join(lignes))
    print()
    print('  -> ' + OUT)


main()
