"""
Pré-traitement cognitif unifié — toutes campagnes
=================================================

Code UNIQUE remplaçant les 4 scripts dupliqués :
  - Campagne_aut_24\\Cognition\\Tests-cog\\Pré-traitement_ANT.py / Pré-traitement_NB.py
  - Campagne_print_25\\Cognition\\Tests-cog\\Pré-traitement_ANT.py / Pré-traitement_NB.py

Il va chercher les données brutes directement dans le sous-dossier Brut\\ de CHAQUE
campagne, calcule les indicateurs (n-back 1B/2B + ANT-I), et écrit les résultats
à la fois :
  - dans Data\\Cognition\\Tests-cog\\Resultats\\  (centralisé, lu par mise_en_F.py
    et trait_cog_all*.py)
  - dans le Resultats\\ local de chaque campagne

La logique de calcul est reprise À L'IDENTIQUE des scripts validés ; seules les deux
fonctions `filtrer_essais` (ANT vs NB, logiques différentes) ont été renommées pour
cohabiter, et la découverte des participants se fait par scan du dossier Brut.

Indicateurs ANT-I :
1. Alerte standard : sans – avec (indice sonore) [tous essais]
2. Alerte auditive : sans – avec (indice sonore) [essais SANS indice spatial]
3. Alerte visuelle : sans_indice – ((valide + invalide)/2) [essais SANS indice sonore]
4. Orientation : invalide – valide [essais AVEC indice spatial]
5. Contrôle exécutif : incongruent – congruent [tous essais]
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402

import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import norm


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Racine des donnees : voir chemins.py
RACINE = D()
RESULTATS_CENTRAL = D(r"Data\Cognition\Tests-cog\Resultats")

CAMPAGNES = [
    {"dossier": "Campagne_aut_24",   "label": "camp1"},
    {"dossier": "Campagne_print_25", "label": "camp2"},
]

TACHES_NB = ["1B", "2B"]

# Participants à exclure explicitement, par campagne (vide par défaut).
# ex: {"camp1": {110}, "camp2": {221, 225}}
EXCLUSIONS = {}


# ============================================================
# Découverte des fichiers
# ============================================================

def participants_disponibles(brut_dir, tache, exclusions=None):
    """Liste triée des numéros de participant présents dans Brut\\ pour une tâche."""
    exclusions = exclusions or set()
    participants = set()
    motif = os.path.join(brut_dir, f"P*-{tache}-*.csv")
    for chemin in glob.glob(motif):
        token = os.path.basename(chemin).split("-")[0]  # "P101"
        try:
            num = int(token.lstrip("Pp"))
        except ValueError:
            continue
        if num not in exclusions:
            participants.add(num)
    return sorted(participants)


def fichier_sujet(brut_dir, tache, participant):
    """Fichiers CSV (passations 1 à 3) pour un participant / une tâche donnés."""
    fichiers = []
    for passation in range(1, 4):
        filename = f"P{participant}-{tache}-{passation}.csv"
        filepath = os.path.join(brut_dir, filename)
        if os.path.isfile(filepath):
            fichiers.append(filepath)
        else:
            print(f"Le fichier {filename} n'existe pas.")
    return fichiers


def charger_participant_data(participant_xlsx):
    """Charge Participant.xlsx avec index 'Numéro' uniformisé en chaîne ('P101')."""
    df = pd.read_excel(participant_xlsx)
    df["Numéro"] = df["Numéro"].astype(str).str.strip()
    return df.set_index("Numéro")


# ============================================================
# Tâche ANT-I  (logique reprise de Pré-traitement_ANT.py)
# ============================================================

def filtrer_essais_ant(data, limite_inf=200, limite_sup=1700):
    """
    Retire les essais sans réponse (None/'none'/NaN) et ceux avec TR hors bornes.
    Retourne les données filtrées et un résumé des essais supprimés.
    """
    sans_reponse = (
        data['response_keyboard_response'].isin(['none', None]) |
        data['response_keyboard_response'].isna()
    )
    data_sans_nonrep = data[~sans_reponse]

    filtred_data = data_sans_nonrep[
        (data_sans_nonrep['response_time_keyboard_response'] >= limite_inf) &
        (data_sans_nonrep['response_time_keyboard_response'] <= limite_sup)
    ]

    essais_supprimes = data[~data.index.isin(filtred_data.index)]

    recap = {
        "Nb_essais_supprimes": len(essais_supprimes),
        "Moyenne_TR_supprimes": essais_supprimes['response_time_keyboard_response'].mean() if not essais_supprimes.empty else None,
        "Écart-type_TR_supprimes": essais_supprimes['response_time_keyboard_response'].std() if not essais_supprimes.empty else None,
        "Indices_supprimes": essais_supprimes.index.tolist()
    }

    return filtred_data, recap


def calcul_indicateurs(data, col_rt="response_time_keyboard_response"):
    """Calcule ER (taux d'erreur) et RT (temps de réaction moyen)."""
    if len(data) == 0:
        return np.nan, np.nan

    er = 1 - data['correct'].mean()
    rt = data[col_rt].mean()
    return er, rt


def indicateurs_ANT(data, col_rt="response_time_keyboard_response"):
    """Calcule les 5 indicateurs ANT-I (ER, RT) pour chaque effet."""

    # === Masques de base ===
    avec_son = data['signal_alerte'] == 1
    sans_son = data['signal_alerte'] == 0

    avec_indice_spatial = data['indice_orientation'] != 0
    sans_indice_spatial = data['indice_orientation'] == 0

    indice_valide = (data['indice_orientation'] == data['position']) & avec_indice_spatial
    indice_invalide = (data['indice_orientation'] != data['position']) & avec_indice_spatial

    congruent = data['condition'] == 'congruente'
    incongruent = data['condition'] == 'incongruente'

    resultats = {}

    # 1. ALERTE STANDARD : sans – avec (indice sonore) [tous essais]
    er_sans_son, rt_sans_son = calcul_indicateurs(data[sans_son], col_rt)
    er_avec_son, rt_avec_son = calcul_indicateurs(data[avec_son], col_rt)
    resultats['Alerte_Standard'] = {
        'ER': er_sans_son - er_avec_son,
        'RT': rt_sans_son - rt_avec_son
    }

    # 2. ALERTE AUDITIVE : sans – avec (indice sonore) [essais SANS indice spatial]
    sous_ensemble_sans_spatial = data[sans_indice_spatial]
    er_aud_sans, rt_aud_sans = calcul_indicateurs(
        sous_ensemble_sans_spatial[sous_ensemble_sans_spatial['signal_alerte'] == 0], col_rt
    )
    er_aud_avec, rt_aud_avec = calcul_indicateurs(
        sous_ensemble_sans_spatial[sous_ensemble_sans_spatial['signal_alerte'] == 1], col_rt
    )
    resultats['Alerte_Auditive'] = {
        'ER': er_aud_sans - er_aud_avec,
        'RT': rt_aud_sans - rt_aud_avec
    }

    # 3. ALERTE VISUELLE : sans_indice – ((valide + invalide)/2) [essais SANS indice sonore]
    sous_ensemble_sans_son = data[sans_son]
    er_valide, rt_valide = calcul_indicateurs(
        sous_ensemble_sans_son[
            (sous_ensemble_sans_son['indice_orientation'] == sous_ensemble_sans_son['position']) &
            (sous_ensemble_sans_son['indice_orientation'] != 0)
        ], col_rt
    )
    er_invalide, rt_invalide = calcul_indicateurs(
        sous_ensemble_sans_son[
            (sous_ensemble_sans_son['indice_orientation'] != sous_ensemble_sans_son['position']) &
            (sous_ensemble_sans_son['indice_orientation'] != 0)
        ], col_rt
    )
    er_sans_indice, rt_sans_indice = calcul_indicateurs(
        sous_ensemble_sans_son[sous_ensemble_sans_son['indice_orientation'] == 0], col_rt
    )
    resultats['Alerte_Visuelle'] = {
        'ER': er_sans_indice - ((er_valide + er_invalide) / 2),
        'RT': rt_sans_indice - ((rt_valide + rt_invalide) / 2)
    }

    # 4. ORIENTATION : invalide – valide [essais AVEC indice spatial]
    er_ori_valide, rt_ori_valide = calcul_indicateurs(data[indice_valide], col_rt)
    er_ori_invalide, rt_ori_invalide = calcul_indicateurs(data[indice_invalide], col_rt)
    resultats['Orientation'] = {
        'ER': er_ori_invalide - er_ori_valide,
        'RT': rt_ori_invalide - rt_ori_valide
    }

    # 5. CONTRÔLE EXÉCUTIF : incongruent – congruent [tous essais]
    er_cong, rt_cong = calcul_indicateurs(data[congruent], col_rt)
    er_incong, rt_incong = calcul_indicateurs(data[incongruent], col_rt)
    resultats['Controle_Executif'] = {
        'ER': er_incong - er_cong,
        'RT': rt_incong - rt_cong
    }

    return resultats


def traitement_ANT(fichiers, participant_data):
    """Traite les fichiers ANT d'un participant et calcule les indicateurs."""
    resultats = []
    recap_essais_supprimes = []

    for fichier in fichiers:
        data = pd.read_csv(fichier)
        data_filtre, recap = filtrer_essais_ant(data)

        passation = int(os.path.basename(fichier).split('-')[-1].split('.')[0])
        participant_id = os.path.basename(fichier).split('-')[0]

        if passation == 1:
            etat_passation = "0"  # Baseline
        else:
            etat_passation = participant_data.loc[participant_id, f"passation{passation}"]

        indicateurs_avant = indicateurs_ANT(data)
        indicateurs_apres = indicateurs_ANT(data_filtre)

        for filtrage, indic in [("Avant", indicateurs_avant), ("Apres", indicateurs_apres)]:
            ligne = [participant_id, passation, etat_passation, filtrage]
            for nom_indic in ['Alerte_Standard', 'Alerte_Auditive', 'Alerte_Visuelle',
                              'Orientation', 'Controle_Executif']:
                ligne.append(indic[nom_indic]['ER'])
                ligne.append(indic[nom_indic]['RT'])
            resultats.append(ligne)

        recap_essais_supprimes.append({
            "Participant": participant_id,
            "Passation": passation,
            "Etat_Passation": etat_passation,
            "Nb_essais_supprimes": recap["Nb_essais_supprimes"],
            "Moyenne_TR_supprimes": recap["Moyenne_TR_supprimes"],
            "Écart-type_TR_supprimes": recap["Écart-type_TR_supprimes"],
            "Indices_essais_supprimes": ", ".join(map(str, recap["Indices_supprimes"]))
        })

    colonnes = [
        'Participant', 'Passation', 'Etat_Passation', 'Filtrage',
        'ER_Alerte_Standard', 'RT_Alerte_Standard',
        'ER_Alerte_Auditive', 'RT_Alerte_Auditive',
        'ER_Alerte_Visuelle', 'RT_Alerte_Visuelle',
        'ER_Orientation', 'RT_Orientation',
        'ER_Controle_Executif', 'RT_Controle_Executif'
    ]

    df_resultats = pd.DataFrame(resultats, columns=colonnes)
    df_recap = pd.DataFrame(recap_essais_supprimes)

    return df_resultats, df_recap


def run_ANT(campagne, participant_data, brut_dir, dossiers_sortie):
    """Pré-traitement ANT pour une campagne -> Resultats_ANT_<label>.xlsx + Recap."""
    label = campagne["label"]
    tache = "ANT"
    participants = participants_disponibles(brut_dir, tache, EXCLUSIONS.get(label, set()))

    tache_df_list = []
    recap_df_list = []

    for participant in participants:
        fichiers = fichier_sujet(brut_dir, tache, participant)
        if not fichiers:
            continue
        result, recap = traitement_ANT(fichiers, participant_data)
        tache_df_list.append(result)
        recap_df_list.append(recap)

    if not tache_df_list:
        print(f"[X] Aucun fichier ANT trouvé pour {label}.")
        return

    tache_df = pd.concat(tache_df_list, ignore_index=True)

    # Concat robuste aux recaps vides (version camp2)
    recap_df_list_clean = [df for df in recap_df_list if not df.empty]
    recap_df = (pd.concat(recap_df_list_clean, ignore_index=True)
                if recap_df_list_clean else pd.DataFrame())

    ecrire(tache_df, f"Resultats_ANT_{label}.xlsx", dossiers_sortie, excel=True)
    ecrire(recap_df, f"Recap_essais_supprimes_ANT_{label}.xlsx", dossiers_sortie, excel=True)
    print(f"[OK] ANT {label} : {tache_df['Participant'].nunique()} participants.")


# ============================================================
# Tâches n-back 1B / 2B  (logique reprise de Pré-traitement_NB.py)
# ============================================================

def filtrer_essais_nb(data, seuil=2):
    """Filtre les essais répondus avec TR hors bornes [200, 1500] ms."""
    repondus = data[data['response'] != 'none']
    if repondus.empty:
        return data, {
            "Nb_essais_supprimes": 0,
            "Moyenne_TR_supprimes": None,
            "Écart-type_TR_supprimes": None,
            "Index_essais_supprimes": []
        }

    borne_inf = 200
    borne_sup = 1500
    data_filtrée = repondus[(repondus['response_time_keyboard_response'] >= borne_inf) &
                            (repondus['response_time_keyboard_response'] <= borne_sup)]
    essais_supprimes = repondus[~repondus.index.isin(data_filtrée.index)]

    data_finale = pd.concat([data_filtrée, data[data['response'] == 'none']])

    recap = {
        "Nb_essais_supprimes": len(essais_supprimes),
        "Moyenne_TR_supprimes": essais_supprimes['response_time_keyboard_response'].mean() if not essais_supprimes.empty else None,
        "Écart-type_TR_supprimes": essais_supprimes['response_time_keyboard_response'].std() if not essais_supprimes.empty else None,
        "Index_essais_supprimes": essais_supprimes.index.tolist() if not essais_supprimes.empty else []
    }
    return data_finale, recap


def corriger_taux_extreme(taux, n):
    """Correction de Hautus (1995) : max(1/(2n), min(taux, 1 - 1/(2n)))."""
    borne_inf = 1 / (2 * n)
    borne_sup = 1 - 1 / (2 * n)
    return max(borne_inf, min(taux, borne_sup))


def calculer_dprime(hits, false_alarms, n_cibles, n_non_cibles):
    """d-prime avec correction des cas extrêmes (Hautus, 1995)."""
    hit_rate = hits / n_cibles if n_cibles > 0 else 0
    fa_rate = false_alarms / n_non_cibles if n_non_cibles > 0 else 0

    hit_rate_corr = corriger_taux_extreme(hit_rate, n_cibles)
    fa_rate_corr = corriger_taux_extreme(fa_rate, n_non_cibles)

    z_hit = norm.ppf(hit_rate_corr)
    z_fa = norm.ppf(fa_rate_corr)
    d_prime = z_hit - z_fa

    c_bias = -0.5 * (z_hit + z_fa)

    return d_prime, hit_rate_corr, fa_rate_corr, c_bias


def traitement_nback(fichier, participant_data, seuil=2):
    """Traite un fichier n-back et calcule les métriques de performance."""
    essais_cibles = 42       # réponse attendue = num_0
    essais_non_cibles = 98   # réponse attendue ≠ num_0

    try:
        data = pd.read_csv(fichier, sep=',').fillna('none')
    except Exception as e:
        print(f"Erreur lecture {fichier}: {e}")
        return None, None

    # ===== Données brutes =====
    hits_brut = ((data['correct_response'] == 'num_0') & (data['response'] == 'num_0')).sum()
    false_ala_brut = ((data['correct_response'] != 'num_0') & (data['response'] == 'num_0')).sum()
    misses_brut = essais_cibles - hits_brut
    correct_rej_brut = essais_non_cibles - false_ala_brut

    Rtime_brut = data.loc[
        (data['correct_response'] == 'num_0') & (data['response'] == 'num_0'),
        'response_time_keyboard_response'
    ].tolist()
    TR_moyen_brut = np.mean(Rtime_brut) if Rtime_brut else None

    d_prime_brut, hit_rate_corr_brut, fa_rate_corr_brut, c_bias_brut = calculer_dprime(
        hits_brut, false_ala_brut, essais_cibles, essais_non_cibles
    )

    # ===== Filtrage =====
    data_filtrée, recap = filtrer_essais_nb(data, seuil)

    # ===== Données filtrées =====
    hits_filtre = ((data_filtrée['correct_response'] == 'num_0') & (data_filtrée['response'] == 'num_0')).sum()
    false_ala_filtre = ((data_filtrée['correct_response'] != 'num_0') & (data_filtrée['response'] == 'num_0')).sum()
    misses_filtre = essais_cibles - hits_filtre
    correct_rej_filtre = essais_non_cibles - false_ala_filtre

    Rtime_filtre = data_filtrée.loc[
        (data_filtrée['correct_response'] == 'num_0') & (data_filtrée['response'] == 'num_0'),
        'response_time_keyboard_response'
    ].tolist()
    TR_moyen_filtre = np.mean(Rtime_filtre) if Rtime_filtre else None

    d_prime_filtre, hit_rate_corr_filtre, fa_rate_corr_filtre, c_bias_filtre = calculer_dprime(
        hits_filtre, false_ala_filtre, essais_cibles, essais_non_cibles
    )

    # ===== Métadonnées =====
    passation = int(os.path.basename(fichier).split('-')[-1].split('.')[0])
    participant_id = os.path.basename(fichier).split('-')[0]
    passation_etat = "0" if passation == 1 else participant_data.loc[participant_id, f"passation{passation}"]

    # ===== Efficacité (d'/TR * 100) =====
    efficacite_brut = (d_prime_brut / TR_moyen_brut * 100) if (TR_moyen_brut and TR_moyen_brut != 0) else None
    efficacite_filtre = (d_prime_filtre / TR_moyen_filtre * 100) if (TR_moyen_filtre and TR_moyen_filtre != 0) else None

    resultats = [
        passation,
        passation_etat,
        # Métriques brutes
        hits_brut,
        false_ala_brut,
        misses_brut,
        correct_rej_brut,
        TR_moyen_brut,
        d_prime_brut,
        c_bias_brut,
        efficacite_brut,
        # Métriques filtrées
        hits_filtre,
        false_ala_filtre,
        misses_filtre,
        correct_rej_filtre,
        TR_moyen_filtre,
        d_prime_filtre,
        c_bias_filtre,
        efficacite_filtre
    ]

    return resultats, recap


def run_nback(campagne, participant_data, brut_dir, dossiers_sortie):
    """Pré-traitement 1B/2B pour une campagne -> Resultats_<tache>_<label>.csv + Recap."""
    label = campagne["label"]

    for tache in TACHES_NB:
        participants = participants_disponibles(brut_dir, tache, EXCLUSIONS.get(label, set()))

        tache_df = []
        recap_essais_supprimes_df = []

        for participant in participants:
            fichiers = fichier_sujet(brut_dir, tache, participant)
            if not fichiers:
                continue

            for fichier in fichiers:
                result, recap_essais_supprimes = traitement_nback(fichier, participant_data)
                if result:
                    tache_df.append([participant] + result)

                if recap_essais_supprimes:
                    passation = os.path.basename(fichier).split('-')[-1].split('.')[0]
                    recap_essais_supprimes_df.append([
                        participant, passation,
                        result[1] if result else None,  # Etat_Passation
                        recap_essais_supprimes["Nb_essais_supprimes"],
                        recap_essais_supprimes["Moyenne_TR_supprimes"],
                        recap_essais_supprimes["Écart-type_TR_supprimes"],
                        ';'.join(map(str, recap_essais_supprimes["Index_essais_supprimes"]))
                    ])

        colonnes = [
            'Participant', 'Passation', 'Etat_Passation',
            # Métriques brutes
            'Hits_brut', 'False_Alarms_brut', 'Misses_brut', 'Correct_Rejections_brut',
            'RT_moyen_brut', 'd_prime_brut', 'c_bias_brut', 'Efficacite_brut',
            # Métriques filtrées
            'Hits_filtre', 'False_Alarms_filtre', 'Misses_filtre', 'Correct_Rejections_filtre',
            'RT_moyen_filtre', 'd_prime_filtre', 'c_bias_filtre', 'Efficacite_filtre'
        ]

        df_resultats = pd.DataFrame(tache_df, columns=colonnes)
        ecrire(df_resultats, f"Resultats_{tache}_{label}.csv", dossiers_sortie, excel=False)

        if recap_essais_supprimes_df:
            recap_cols = [
                'Participant', 'Passation', 'Etat_Passation',
                'Nb_essais_supprimes', 'Moyenne_TR_supprimes',
                'Écart-type_TR_supprimes', 'Index_essais_supprimes'
            ]
            df_recap = pd.DataFrame(recap_essais_supprimes_df, columns=recap_cols)
            ecrire(df_recap, f"Recap_essais_supprimes_{tache}_{label}.xlsx", dossiers_sortie, excel=True)

        print(f"[OK] {tache} {label} : {df_resultats['Participant'].nunique()} participants.")


# ============================================================
# Écriture (double : centralisé + local campagne)
# ============================================================

def ecrire(df, nom, dossiers, excel=False):
    for d in dossiers:
        os.makedirs(d, exist_ok=True)
        chemin = os.path.join(d, nom)
        if excel:
            df.to_excel(chemin, index=False)
        else:
            df.to_csv(chemin, index=False)
        print(f"  -> {chemin}")


# ============================================================
# Programme principal
# ============================================================

def main():
    print(f"Racine détectée : {RACINE}")
    print(f"Résultats centralisés : {RESULTATS_CENTRAL}\n")

    for campagne in CAMPAGNES:
        tests_cog = os.path.join(RACINE, campagne["dossier"], "Cognition", "Tests-cog")
        brut_dir = os.path.join(tests_cog, "Brut")
        participant_xlsx = os.path.join(tests_cog, "Participant.xlsx")
        resultats_local = os.path.join(tests_cog, "Resultats")
        dossiers_sortie = [RESULTATS_CENTRAL, resultats_local]

        print(f"===== {campagne['dossier']} ({campagne['label']}) =====")

        if not os.path.isdir(brut_dir):
            print(f"[X] Dossier Brut introuvable : {brut_dir}\n")
            continue
        if not os.path.isfile(participant_xlsx):
            print(f"[X] Participant.xlsx introuvable : {participant_xlsx}\n")
            continue

        participant_data = charger_participant_data(participant_xlsx)

        run_nback(campagne, participant_data, brut_dir, dossiers_sortie)
        run_ANT(campagne, participant_data, brut_dir, dossiers_sortie)
        print()

    print("Pré-traitement terminé.")


if __name__ == "__main__":
    main()
