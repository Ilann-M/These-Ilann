# -*- coding: utf-8 -*-
"""
Robustesse des effets du trajet (niveaux 1 a 3) au stress et au ressenti du
trajet (chapitre 4, section 4.2.3).

Statut conceptuel : le stress de conduite, la difficulte ressentie et l'affect
induit par le trajet sont des CONSEQUENCES de l'itineraire (controles de
manipulation, 4.1), mesurees a chaque session. Ils ne sont donc pas des
confondants du contraste Trajet au sens strict mais des mediateurs possibles.
Ajuster sur eux repond a la question : l'effet du trajet subsiste-t-il une fois
retiree la part qui passe par le ressenti ? Un effet qui disparait serait
compatible avec une mediation par le stress ; un effet qui subsiste ne l'est pas.

Trois tests par indicateur et par niveau :
  REF   : Valeur ~ Mode * Trajet + Session                 (modele du chapitre)
  ADJ1  : REF + stress_w                                    (stress de conduite,
          centre intra-participant : seule la variation entre les deux trajets
          d'un meme participant est utilisee)
  ADJ4  : REF + stress_w + difficulte_w + fatigue_w + anxiete_w
  INTER : REF + Trajet x DeltaStress  (DeltaStress = stress proche - stress
          eloigne, constant par participant ; teste si l'effet du trajet est
          plus fort chez ceux dont le stress a le plus augmente)
S'y ajoute la correlation de rang intra-participant entre la variation de
l'indicateur et la variation du stress (Delta vs Delta).

Niveau 1 : intention (51 participants). Niveau 2 : filtrage ratio >= 1,1 sur
BC et PUF. Niveau 3 : reetiquetage Fort/Faible sur BC et PUF, facteur
Exposition a la place de Trajet. Les regles sont celles des scripts
LMM_corr_FDR_2niveau_2.py et LMM_expoR_FDR_2niveau.py.
Ordonnee a l'origine et pente aleatoires (repli sur l'ordonnee seule), comme
dans LMM_intention.py. FDR BH par famille N-Back (8) / ANT (23).

Sortie : Data\\Traitement\\Resultats_LMM_intention\\26-10-26_stress
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

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from LMM_intention import (SELECTION_4, DATA_FILE, generer_catalogue,
                           charger_donnees, filtrer_selection,
                           preparer_donnees_long, _formule)

warnings.filterwarnings('ignore')

OUTPUT_DIR = S(r'Resultats_LMM_intention\26-10-26_stress')
ALPHA = 0.05
RATIO_MIN = 1.1
CRITERES = [
    {'nom': 'BC880_Mean', 'col_A': 'BC880_Mean_A_EXP', 'col_C': 'BC880_Mean_C_EXP'},
    {'nom': 'Number_Mean', 'col_A': 'Number_Mean_A_EXP', 'col_C': 'Number_Mean_C_EXP'},
]
RESSENTI = {
    'stress': 'stress_conduite',
    'difficulte': 'difficulte_conduite',
    'fatigue': 'fatigue',
    'anxiete': 'roue_diff_impact_trajet_Anxiete',
}


# ------------------------------------------------------------------ covariables
def table_ressenti(df):
    """Format long Participant x Trajet_reel des scores de ressenti, centres
    intra-participant (suffixe _w) ; DeltaStress constant par participant."""
    lignes = []
    for _, r in df.iterrows():
        for trajet, suf in (('Away', 'A'), ('Close', 'C')):
            d = {'Participant': r['Participant'], 'Trajet_reel': trajet}
            for k, col in RESSENTI.items():
                d[k] = pd.to_numeric(r.get(f'{col}_{suf}'), errors='coerce')
            lignes.append(d)
    lg = pd.DataFrame(lignes)
    for k in RESSENTI:
        lg[f'{k}_w'] = lg[k] - lg.groupby('Participant')[k].transform('mean')
    delta = (lg.pivot(index='Participant', columns='Trajet_reel', values='stress'))
    delta = (delta['Close'] - delta['Away']).rename('DeltaStress').reset_index()
    return lg, delta


# ------------------------------------------------------------------ modeles
def ajuster(formule, lg, facteur):
    """Mixedlm pente aleatoire sur le facteur, repli sur l'ordonnee seule.
    Renvoie coef, p du terme principal du facteur et le type d'effet aleatoire."""
    for re_formula, label in ((f'~{facteur}', 'pente'), (None, 'intercept')):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                kw = {'re_formula': re_formula} if re_formula else {}
                res = mixedlm(formule, lg, groups=lg['Participant'], **kw).fit(reml=True)
            if not np.isfinite(res.params).all() or not np.isfinite(res.bse).all():
                continue
            k = [x for x in res.params.index if facteur in x and ':' not in x][0]
            return dict(coef=res.params[k], p=res.pvalues[k], re=label)
        except Exception:
            continue
    return dict(coef=np.nan, p=np.nan, re='echec')


def interaction(formule, lg, facteur):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            res = mixedlm(formule, lg, groups=lg['Participant']).fit(reml=False)
        k = [x for x in res.params.index if facteur in x and 'DeltaStress' in x][0]
        return dict(coef=res.params[k], p=res.pvalues[k])
    except Exception:
        return dict(coef=np.nan, p=np.nan)


def tester(lg, facteur):
    """Enchaine REF / ADJ1 / ADJ4 / INTER sur un format long muni des covariables."""
    base = f"Valeur ~ C(Mode_transport, Sum) * C({facteur}, Sum)"
    out = {}
    ref = ajuster(_formule(lg, base), lg, facteur)
    out.update({'REF_coef': ref['coef'], 'REF_p': ref['p'], 'REF_re': ref['re'],
                'N': lg['Participant'].nunique()})
    l1 = lg.dropna(subset=['stress_w'])
    l1 = l1[l1.groupby('Participant')['Participant'].transform('size') == 2]
    a1 = ajuster(_formule(l1, base + ' + stress_w'), l1, facteur)
    r1 = ajuster(_formule(l1, base), l1, facteur)
    out.update({'N_ADJ1': l1['Participant'].nunique(), 'REF_echADJ1_p': r1['p'],
                'ADJ1_coef': a1['coef'], 'ADJ1_p': a1['p']})
    cols4 = [f'{k}_w' for k in RESSENTI]
    l4 = lg.dropna(subset=cols4)
    l4 = l4[l4.groupby('Participant')['Participant'].transform('size') == 2]
    a4 = ajuster(_formule(l4, base + ' + ' + ' + '.join(cols4)), l4, facteur)
    r4 = ajuster(_formule(l4, base), l4, facteur)
    out.update({'N_ADJ4': l4['Participant'].nunique(), 'REF_echADJ4_p': r4['p'],
                'ADJ4_coef': a4['coef'], 'ADJ4_p': a4['p']})
    li = l1.dropna(subset=['DeltaStress'])
    it = interaction(_formule(li, f"Valeur ~ Mode_transport * {facteur} + DeltaStress * {facteur}"),
                     li, facteur)
    out.update({'INTER_coef': it['coef'], 'INTER_p': it['p']})
    # correlation de rang Delta indicateur vs Delta stress (sur les trajets reels)
    w = l1.pivot(index='Participant', columns='Trajet_reel', values='Valeur')
    d = (w['Close'] - w['Away']).rename('DeltaValeur').reset_index()
    d = d.merge(l1[['Participant', 'DeltaStress']].drop_duplicates(), on='Participant')
    if len(d) >= 5:
        rho, prho = stats.spearmanr(d['DeltaValeur'], d['DeltaStress'])
    else:
        rho, prho = np.nan, np.nan
    out.update({'rho_DeltaStress': rho, 'p_rho': prho})
    return out


# ------------------------------------------------------------------ niveaux
def niveau_2(df, crit):
    ok = df[crit['col_A']].notna() & df[crit['col_C']].notna()
    sub = df[ok]
    return sub[sub[crit['col_C']] >= RATIO_MIN * sub[crit['col_A']]].copy()


def classes_niveau_3(df, crit):
    ok = df[crit['col_A']].notna() & df[crit['col_C']].notna() & (df[crit['col_A']] != df[crit['col_C']])
    sub = df[ok]
    inv = sub[crit['col_A']] > sub[crit['col_C']]
    return pd.DataFrame({'Participant': sub['Participant'].values,
                         'Away': np.where(inv, 'Fort', 'Faible'),
                         'Close': np.where(inv, 'Faible', 'Fort')}).set_index('Participant')


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df = filtrer_selection(charger_donnees(DATA_FILE), SELECTION_4)
    ress, delta = table_ressenti(df)
    print(f"{len(df)} participants ; stress disponible aux deux trajets : "
          f"{delta['DeltaStress'].notna().sum()}")
    catalogue = generer_catalogue()

    lignes = []
    plans = [('N1_intention', None, df, 'Trajet')]
    for crit in CRITERES:
        plans.append((f"N2_filtre_{crit['nom']}", crit, niveau_2(df, crit), 'Trajet'))
        plans.append((f"N3_reetiq_{crit['nom']}", crit, df, 'Exposition'))

    for label, crit, d_niv, facteur in plans:
        cls = classes_niveau_3(df, crit) if facteur == 'Exposition' else None
        if cls is not None:
            d_niv = d_niv[d_niv['Participant'].isin(cls.index)]
        for indic in catalogue:
            ca, cc = indic['col_away'], indic['col_close']
            if ca not in d_niv.columns or cc not in d_niv.columns:
                continue
            lg, _ = preparer_donnees_long(d_niv, ca, cc, indic.get('transformation'))
            if lg is None or lg['Participant'].nunique() < 10:
                continue
            if facteur == 'Exposition':
                lg['Exposition'] = [cls.loc[p, t] for p, t in zip(lg['Participant'], lg['Trajet_reel'])]
            lg = lg.merge(ress, on=['Participant', 'Trajet_reel'], how='left')
            lg = lg.merge(delta, on='Participant', how='left')
            row = {'Niveau': label, 'Facteur': facteur, 'Indicateur': indic['nom'],
                   'Tache': indic['tache'], 'Famille': indic['categorie'], 'Domaine': indic['domaine']}
            row.update(tester(lg, facteur))
            lignes.append(row)
        print(f"  {label} : {sum(1 for l in lignes if l['Niveau'] == label)} indicateurs")

    res = pd.DataFrame(lignes)
    for col in ['REF_p', 'ADJ1_p', 'ADJ4_p', 'INTER_p']:
        q = col.replace('_p', '_q')
        res[q] = np.nan
        for (niv, fam), g in res.groupby(['Niveau', 'Famille']):
            ok = g[col].notna()
            if ok.sum():
                res.loc[g.index[ok], q] = multipletests(g.loc[ok, col], method='fdr_bh')[1]
    res.to_excel(os.path.join(OUTPUT_DIR, 'Robustesse_stress.xlsx'), index=False)

    # ---------------------------------------------------------------- rapport
    L = []

    def w(s=''):
        print(s)
        L.append(s)

    w("ROBUSTESSE DES EFFETS DU TRAJET AU STRESS ET AU RESSENTI DU TRAJET (26/10/2026)")
    w("REF = modele du chapitre ; ADJ1 = + stress de conduite (centre intra) ; "
      "ADJ4 = + stress, difficulte, fatigue, anxiete ; INTER = Trajet x DeltaStress")
    w("FDR BH par famille N-Back (8) / ANT (23), par niveau et critere.")
    w()
    for niv, g in res.groupby('Niveau', sort=False):
        w(f"--- {niv} (N {g['N'].min()}-{g['N'].max()}, ADJ1 N {g['N_ADJ1'].min()}-{g['N_ADJ1'].max()})")
        w(f"    p<0,05 : REF {int((g.REF_p < .05).sum())} | ADJ1 {int((g.ADJ1_p < .05).sum())} | "
          f"ADJ4 {int((g.ADJ4_p < .05).sum())} | INTER {int((g.INTER_p < .05).sum())}   "
          f"q<0,05 : REF {int((g.REF_q < .05).sum())} | ADJ1 {int((g.ADJ1_q < .05).sum())} | "
          f"ADJ4 {int((g.ADJ4_q < .05).sum())} | INTER {int((g.INTER_q < .05).sum())}")
        w(f"    rho(Delta indicateur, Delta stress) p<0,05 : {int((g.p_rho < .05).sum())} / {len(g)}")
    w()
    w("Effets de reference a p<0,10 (REF), et leur devenir :")
    cols = ['Niveau', 'Indicateur', 'N', 'REF_coef', 'REF_p', 'REF_q', 'N_ADJ1', 'REF_echADJ1_p',
            'ADJ1_coef', 'ADJ1_p', 'ADJ1_q', 'ADJ4_coef', 'ADJ4_p', 'INTER_p', 'rho_DeltaStress', 'p_rho']
    sel = res[res.REF_p < .10].sort_values(['Niveau', 'REF_p'])
    w(sel[cols].round(4).to_string(index=False))
    w()
    w("Tests a p<0,05 apres ajustement (ADJ1 ou ADJ4) alors que REF >= 0,05 :")
    nv = res[((res.ADJ1_p < .05) | (res.ADJ4_p < .05)) & (res.REF_p >= .05)]
    w(nv[cols].round(4).to_string(index=False) if len(nv) else "  aucun")
    w()
    w("Interactions Trajet x DeltaStress a p<0,05 :")
    it = res[res.INTER_p < .05]
    w(it[['Niveau', 'Indicateur', 'INTER_coef', 'INTER_p', 'INTER_q']].round(4).to_string(index=False) if len(it) else "  aucune")
    w()
    w("Correlations de rang Delta indicateur / Delta stress a p<0,05 (niveau 1) :")
    rr = res[(res.Niveau == 'N1_intention') & (res.p_rho < .05)]
    w(rr[['Indicateur', 'N_ADJ1', 'rho_DeltaStress', 'p_rho']].round(4).to_string(index=False) if len(rr) else "  aucune")
    with open(os.path.join(OUTPUT_DIR, 'Rapport_robustesse_stress.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"\n-> {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
