# -*- coding: utf-8 -*-
"""
Indicateurs percentiles du bruit (L90, L50, L10) par trajet, appari\u00e9s par participant.

L'article 1 caract\u00e9rise l'environnement sonore par deux percentiles plut\u00f4t que par le
seul niveau \u00e9quivalent : L90, niveau d\u00e9pass\u00e9 90 % du temps, qui sert d'estimateur du
bruit de fond, et L10, niveau d\u00e9pass\u00e9 10 % du temps, qui capte les pointes sonores
intrusives. Le chapitre 4 n'utilisait que le LAeq, ce qui interdisait tout
rapprochement avec la figure 2 de l'article. Ce script produit les trois indicateurs
sur la m\u00eame cha\u00eene que bruit_apparie_par_participant.py.

Convention : un niveau d\u00e9pass\u00e9 X % du temps correspond au percentile (100 - X) de la
distribution des Leq instantan\u00e9s. L90 est donc le 10e percentile et L10 le 90e.

Cha\u00eene : chaque archive NoiseCapture contient un track.geojson dont chaque point porte
un leq_mean (environ un point par seconde). Les percentiles sont calcul\u00e9s sur cette
distribution, le LAeq sur la moyenne \u00e9nerg\u00e9tique. Le rattachement (date, cr\u00e9neau) ->
trajet -> participant reprend Base.xlsx.

Sortie : Data/Traitement/Descriptifs/bruit_percentiles/
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402
import os
import re
import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

CAMPAGNES = [
    ('Campagne_print_25', D(r'Campagne_print_25\Trajets\Capteurs\NoiseCapture\data')),
    ('Campagne_aut_24', D(r'Campagne_aut_24\Trajets\Capteurs\NoiseCapture\data')),
]
BASE = D(r'Data\Base.xlsx')
LMM_SRC = _os.path.join(DEPOT, '07_modeles', 'LMM_intention.py')
OUT = S(r'Descriptifs\bruit_percentiles')

INDICATEURS = [('LAeq', 'niveau \u00e9quivalent'),
               ('L90', 'bruit de fond'),
               ('L50', 'niveau m\u00e9dian'),
               ('L10', 'pointes intrusives')]


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


def indicateurs_track(zpath):
    """LAeq energetique + percentiles sur les Leq instantanes d'un enregistrement."""
    z = zipfile.ZipFile(zpath)
    gj = json.loads(z.read('track.geojson').decode('utf-8', 'ignore'))
    leqs = [f['properties']['leq_mean'] for f in gj.get('features', [])
            if f.get('properties', {}).get('leq_mean') is not None]
    if len(leqs) < 60:
        return None
    a = np.asarray(leqs, dtype=float)
    meta = z.read('meta.properties').decode('utf-8', 'ignore')
    rec = next((l.split('=')[1] for l in meta.splitlines()
                if l.startswith('record_utc')), None)
    if rec is None:
        return None
    dt = datetime.fromtimestamp(int(rec) / 1000, tz=timezone.utc)
    return dict(
        n_points=len(a),
        LAeq=10 * np.log10(np.mean(10 ** (a / 10))),
        L90=float(np.percentile(a, 10)),
        L50=float(np.percentile(a, 50)),
        L10=float(np.percentile(a, 90)),
        annee=dt.year, mois=dt.month, jour=dt.day,
        slot='AM' if dt.hour < 12 else 'PM',
    )


def sessions_par_participant():
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
            out.append((pid, yr, mon, day, 'AM' if hour < 12 else 'PM',
                        'Proche' if str(pas).strip().upper() == 'C' else 'Loin'))
    return pd.DataFrame(out, columns=['Participant', 'annee', 'mois', 'jour', 'slot', 'trajet'])


def main():
    os.makedirs(OUT, exist_ok=True)
    lignes = []

    def p(t=''):
        print(t)
        lignes.append(str(t))

    rows = []
    for camp, d in CAMPAGNES:
        if not os.path.isdir(d):
            p('  !! dossier absent : %s' % d)
            continue
        for f in sorted(os.listdir(d)):
            if not (f.startswith('track_') and f.endswith('.zip')):
                continue
            r = indicateurs_track(os.path.join(d, f))
            if r:
                r['campagne'] = camp
                rows.append(r)
    tracks = pd.DataFrame(rows)

    p('=' * 78)
    p('  BRUIT : PERCENTILES L90 / L50 / L10, APPARI\u00c9S PAR PARTICIPANT')
    p('=' * 78)
    p('  Enregistrements exploit\u00e9s : %d' % len(tracks))
    p('  Points de mesure par enregistrement : m\u00e9diane %d (min %d, max %d)'
      % (tracks['n_points'].median(), tracks['n_points'].min(), tracks['n_points'].max()))
    p()

    par_session = tracks.groupby(['annee', 'mois', 'jour', 'slot']).agg(
        LAeq=('LAeq', lambda s: 10 * np.log10(np.mean(10 ** (np.asarray(s) / 10)))),
        L90=('L90', 'mean'), L50=('L50', 'mean'), L10=('L10', 'mean')).reset_index()

    sess = sessions_par_participant()
    m = sess.merge(par_session, on=['annee', 'mois', 'jour', 'slot'], how='left')
    sel = selection_4()

    res = []
    apparie = {}          # valeurs par participant, pour la figure 4.1 du chapitre 4
    for col, libelle in INDICATEURS:
        wide = m.pivot_table(index='Participant', columns='trajet', values=col)
        if 'Loin' not in wide.columns or 'Proche' not in wide.columns:
            continue
        wide = wide.dropna(subset=['Loin', 'Proche'])
        wide = wide[wide.index.isin(sel)]
        a, c = wide['Loin'].values, wide['Proche'].values
        apparie[col + '_Loin'] = wide['Loin']
        apparie[col + '_Proche'] = wide['Proche']
        diff = c - a
        t, p_t = stats.ttest_rel(c, a)
        try:
            _, p_w = stats.wilcoxon(c, a)
        except Exception:
            p_w = float('nan')
        dz = diff.mean() / diff.std(ddof=1)
        p('  %-5s (%-20s) n = %d paires' % (col, libelle, len(wide)))
        p('      Loin %.1f (ET %.1f)   Proche %.1f (ET %.1f)   diff %+.2f dB(A)'
          % (a.mean(), a.std(ddof=1), c.mean(), c.std(ddof=1), diff.mean()))
        p('      t appari\u00e9 p = %.4f   Wilcoxon p = %.4f   dz = %+.3f   inversions : %d/%d'
          % (p_t, p_w, dz, int((diff < 0).sum()), len(wide)))
        p()
        res.append(dict(indicateur=col, libelle=libelle, n=len(wide),
                        loin=a.mean(), loin_et=a.std(ddof=1),
                        proche=c.mean(), proche_et=c.std(ddof=1),
                        diff=diff.mean(), p_t=p_t, p_wilcoxon=p_w, dz=dz,
                        inversions=int((diff < 0).sum())))

    p('  Lecture : L90 mesure le fond sonore permanent, L10 les pointes li\u00e9es aux')
    p('  passages de v\u00e9hicules. Un contraste port\u00e9 surtout par le L10 signale une')
    p('  diff\u00e9rence d\u2019\u00e9v\u00e9nements ; port\u00e9 par le L90, une diff\u00e9rence de fond.')

    pd.DataFrame(res).to_excel(os.path.join(OUT, 'Percentiles_apparies.xlsx'), index=False)
    tracks.to_excel(os.path.join(OUT, 'Percentiles_par_track.xlsx'), index=False)
    # Valeurs appariees par participant : source unique des panneaux acoustiques
    # de la figure 4.1 du chapitre 4, qui doit reposer sur les memes 29 paires
    # que le tableau 4.2 et non sur la colonne Leq de Tab_passation.
    pd.DataFrame(apparie).rename_axis('Participant').reset_index().to_excel(
        os.path.join(OUT, 'Percentiles_par_participant.xlsx'), index=False)
    io.open(os.path.join(OUT, 'Rapport_percentiles.txt'), 'w', encoding='utf-8').write('\n'.join(lignes))
    print()
    print('  -> ' + OUT)


main()
