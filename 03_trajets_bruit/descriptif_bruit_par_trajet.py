# -*- coding: utf-8 -*-
"""
Descriptif du bruit (LAeq) par trajet (Loin/Proche), pour Ch4 §4.1.3.
- LAeq énergétique par track NoiseCapture (leq_mean par point).
- Mapping (année, mois, jour, créneau AM/PM) -> trajet, construit depuis Base.xlsx
  (Participant, Passation-2/3 = A/C, Date_P2/P3 = « JJ/MM : HHh »).
- LAeq moyen par trajet + test.
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D as DR, S, DEPOT  # noqa: E402
import sys, os, re, zipfile, json
from datetime import datetime, timezone
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')

def campagne_annee(pid):
    if (100 < pid < 121) or (200 < pid < 221): return 2024
    if (120 < pid < 141) or (220 < pid < 241): return 2025
    return None

# ---- mapping (annee, mois, jour, slot) -> trajet depuis Base.xlsx ----
b = pd.read_excel(DR(r'Data/Base.xlsx'))
sess = {}
for _, r in b.iterrows():
    pid = int(re.search(r'(\d+)', str(r['Participant'])).group(1))
    yr = campagne_annee(pid)
    if yr is None: continue
    for dcol, pcol in [('Date_P2', 'Passation-2'), ('Date_P3', 'Passation-3')]:
        ds, pas = r[dcol], r[pcol]
        if pd.isna(ds) or pd.isna(pas): continue
        m = re.search(r'(\d+)\s*/\s*(\d+).*?(\d+)\s*h', str(ds))
        if not m: continue
        day, mon, hour = int(m.group(1)), int(m.group(2)), int(m.group(3))
        slot = 'AM' if hour < 12 else 'PM'
        traj = 'Proche' if str(pas).strip().upper() == 'C' else 'Loin'   # C=Close=Proche, A=Away=Loin
        sess[(yr, mon, day, slot)] = traj
print('sessions (date,créneau)->trajet dans Base:', len(sess))

# ---- LAeq par track + assignation trajet ----
def laeq_track(zpath):
    z = zipfile.ZipFile(zpath)
    gj = json.loads(z.read('track.geojson').decode('utf-8', 'ignore'))
    leqs = [f['properties']['leq_mean'] for f in gj.get('features', [])
            if f.get('properties', {}).get('leq_mean') is not None]
    if not leqs: return None
    laeq = 10*np.log10(np.mean(10**(np.array(leqs)/10)))
    meta = z.read('meta.properties').decode('utf-8', 'ignore')
    rec = next((l.split('=')[1] for l in meta.splitlines() if l.startswith('record_utc')), None)
    dt = datetime.fromtimestamp(int(rec)/1000, tz=timezone.utc)
    return laeq, dt

rows = []
for camp in ['Campagne_print_25', 'Campagne_aut_24']:
    D = DR(rf'{camp}/Trajets/Capteurs/NoiseCapture/data')
    if not os.path.isdir(D): continue
    for t in [f for f in os.listdir(D) if f.startswith('track_') and f.endswith('.zip')]:
        r = laeq_track(os.path.join(D, t))
        if not r: continue
        laeq, dt = r
        slot = 'AM' if dt.hour < 12 else 'PM'
        traj = sess.get((dt.year, dt.month, dt.day, slot))
        rows.append({'campagne': camp, 'date': dt.strftime('%Y-%m-%d'), 'slot': slot,
                     'laeq': laeq, 'trajet': traj})

df = pd.DataFrame(rows)
n_match = df['trajet'].notna().sum()
print(f'tracks total: {len(df)} | appariés à un trajet: {n_match} | non appariés: {len(df)-n_match}')
print('\n=== LAeq par trajet (tous, dB(A)) ===')
for tj in ['Loin', 'Proche']:
    v = df.loc[df['trajet'] == tj, 'laeq']
    if len(v): print(f'  {tj:7} n={len(v):2d}  moy {v.mean():.1f}  ET {v.std(ddof=1):.1f}  médiane {v.median():.1f}  [{v.min():.1f}-{v.max():.1f}]')
lo, pr = df.loc[df['trajet']=='Loin','laeq'], df.loc[df['trajet']=='Proche','laeq']
if len(lo) >= 3 and len(pr) >= 3:
    print(f'  Δ Proche-Loin = {pr.mean()-lo.mean():+.1f} dB(A) | Mann-Whitney p = {stats.mannwhitneyu(pr,lo,alternative="two-sided").pvalue:.3f} | t p = {stats.ttest_ind(pr,lo,equal_var=False).pvalue:.3f}')
print('\n=== par campagne ===')
for camp in df['campagne'].unique():
    sub = df[df['campagne']==camp]
    for tj in ['Loin','Proche']:
        v = sub.loc[sub['trajet']==tj,'laeq']
        if len(v): print(f'  {camp} {tj:7} n={len(v):2d} moy {v.mean():.1f} dB(A)')
os.makedirs(S(r'Descriptifs'), exist_ok=True)
df.to_csv(S(r'Descriptifs/LAeq_par_track.csv'), index=False, encoding='utf-8-sig')
