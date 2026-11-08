# -*- coding: utf-8 -*-
"""
Traitement UNIFIÉ des questionnaires "passation" (par session) des deux campagnes.

Ce script remplace les deux Q_trait_passation_v4.py identiques (hors nom de
fichier / liste de participants / nom de sortie) :
  - Campagne_aut_24\\Cognition\\Questionnaires\\Q_trait_passation_v4.py
  - Campagne_print_25\\Cognition\\Questionnaires\\Q_trait_passation_v4.py

Il lit les CSV bruts LimeSurvey de chaque campagne (Cahier de passation : roues
émotionnelles, trajets, post-questionnaires PostP, informations préalables
InfoP) et écrit, pour chaque campagne, le tableau "complet" (3 sessions) + le
tableau "sessions 2-3" directement dans
<racine>\\Data\\Cognition\\Questionnaire (lu ensuite par format_all3.py).

La logique de calcul est strictement identique à l'ancien Q_trait_passation_v4.py ;
seuls le CSV source, la liste de participants et les noms de sortie changent par
campagne (voir CAMPAGNES ci-dessous).
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402

import os
import sys
import re
import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

# Console UTF-8 (les prints contiennent des caractères accentués / symboles)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

#####################################
#### Configuration des campagnes ####
#####################################

DOSSIER_SORTIE = D(r"Data\Cognition\Questionnaire")

CAMPAGNES = [
    {
        "nom": "Q1 - automne 2024",
        "csv": D(r"Campagne_aut_24\Cognition\Questionnaires\questionnaire_passation1.csv"),
        "participants": list(range(101, 117)) + list(range(201, 217)),
        "sortie_complet": "resultats_Q1-passation_complet.csv",
        "sortie_s23": "resultats_Q1-passation_sessions2-3.csv",
    },
    {
        "nom": "Q2 - printemps 2025",
        "csv": D(r"Campagne_print_25\Cognition\Questionnaires\questionnaire_passation2.csv"),
        "participants": list(range(121, 140)) + list(range(221, 240)),
        "sortie_complet": "resultats_Q2-passation_complet.csv",
        "sortie_s23": "resultats_Q2-passation_sessions2-3.csv",
    },
]


##############################
#### Fonctions de calcul  ####
##############################

def calcul_infoP(df_infoP):
    df = df_infoP.copy()

    # Ajout ID et Session
    idsess = id_session(df)
    df["ID"] = idsess["ID"].values
    df["Session"] = idsess["Session"].values

    df_result = []

    substances_map = {
        "InfoP2[SQ001]": "café",
        "InfoP2[SQ002]": "tabac",
        "InfoP2[SQ003]": "alcool",
        "InfoP2[SQ004]": "thé",
    }

    for _, row in df.iterrows():
        sess = int(row["Session"])
        row_result = {
            "ID": row["ID"]
        }

        # Médicament
        med_col = f"{sess}_medication_journee"
        row_result[med_col] = row["InfoP1B"] if row.get("InfoP1") == "Y" else ""

        # Sommeil (avec gestion point/virgule)
        sommeil_col = f"{sess}_sommeil_journee"
        sommeil_val = row.get("InfoP1C", "")
        if pd.notna(sommeil_val) and sommeil_val != "":
            # Remplacer le point par une virgule si nécessaire
            sommeil_val = str(sommeil_val).replace(".", ",")
        row_result[sommeil_col] = sommeil_val

        # Stimulants
        stim_col = f"{sess}_stimulant_journee"
        if row.get("InfoP2[SQ005]") == "Y":
            row_result[stim_col] = "Non"
        else:
            stimulants = [nom for k, nom in substances_map.items() if row.get(k) == "Y"]
            row_result[stim_col] = ", ".join(stimulants) if stimulants else ""

        # Délai stimulants
        row_result[f"{sess}_delai_stimulant"] = row.get("InfoP2B", "")

        # Sensation et stress
        try:
            row_result[f"{sess}_sensation"] = int(row.get("InfoP3")[1:]) - 1
        except:
            row_result[f"{sess}_sensation"] = None

        try:
            row_result[f"{sess}_stress"] = int(row.get("InfoP4")[1:]) - 1
        except:
            row_result[f"{sess}_stress"] = None

        # Heure dernier repas
        if row.get("InfoP5") == "Y":
            row_result[f"{sess}_heure_dernier_repas"] = row.get("InfoP5B", "")
            row_result[f"{sess}_description_repas"] = row.get("InfoP5C", "")
        else:
            row_result[f"{sess}_heure_dernier_repas"] = ""
            row_result[f"{sess}_description_repas"] = ""

        df_result.append(row_result)

    df_result = pd.DataFrame(df_result)
    df_result = df_result.groupby("ID", as_index=False).first()

    return df_result


def calcul_postP(df_postP):
    # copie et ajout de ID/Session
    df = df_postP.copy()
    idsess = id_session(df)  # doit retourner colonnes "ID" et "Session"
    df["ID"] = idsess["ID"].values
    df["Session"] = idsess["Session"].values

    # helper pour convertir "A1"→0 … "A11"→10
    def code_to_score(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s.startswith("A") and s[1:].isdigit():
            return int(s[1:]) - 1
        else:
            return np.nan

    # on part du long format
    df_result = pd.DataFrame({
        "ID": df["ID"],
        "Session": df["Session"],
        # difficulté
        "difficulte_conduite": df["PostP1[SQ001]"].map(code_to_score),
        "difficulte_1Back":     df["PostP1[SQ002]"].map(code_to_score),
        "difficulte_2Back":     df["PostP1[SQ003]"].map(code_to_score),
        "difficulte_ANT":       df["PostP1[SQ004]"].map(code_to_score),
        # stress
        "stress_conduite":      df["PostP2[SQ001]"].map(code_to_score),
        "stress_1Back":         df["PostP2[SQ002]"].map(code_to_score),
        "stress_2Back":         df["PostP2[SQ003]"].map(code_to_score),
        "stress_ANT":           df["PostP2[SQ004]"].map(code_to_score),
        # fatigue
        "fatigue":              df["PostP3"].map(code_to_score),
        # remarques
        "remarques":            df["PostP4"].astype(str).replace("nan", "")
    })

    return df_result


def id_session(df):
    # 2) détecter dynamiquement ID et session (AVANT toute conversion)
    id_col, sess_col = df.columns[0], df.columns[1]

    # 3) extraire et renommer ID et session
    df2 = df[[id_col, sess_col]].rename(columns={id_col: "ID", sess_col: "Session_label"})

    # 4) nettoyer la session (strip, NBSP)
    df2["Session_label"] = (
        df2["Session_label"].astype(str)
                         .str.replace("\u00A0", "")
                         .str.strip()
    )

    # 5) coder en chiffres la session
    df2["Session"] = df2["Session_label"].map({"A1": 1, "A2": 2, "A3": 3})
    return df2


def calcul_trajet(df_trajet):
    df = df_trajet.copy()

    # 1) On récupère ID et Session de façon alignée, sans merge cartésien
    idsess = id_session(df)             # contient les colonnes ID, Session_label, Session
    # idsess a exactement la même index que df
    df["Session"] = idsess["Session"].values
    df["ID"]      = idsess["ID"].values

    # 2) On peut maintenant construire df_result ligne à ligne
    modes = ["Pied", "Voiture", "Vélo", "VAE", "Trottinette", "Bus", "e-Trott", "Métro", "Tram", "Train"]

    # initialise le résultat avec ID+Session
    df_result = df[["ID", "Session"]].copy()

    # 3) calcul des modes utilisés (trajet1[SQ001..010])
    mode_cols = [f"Trajet1[SQ{j:03d}]" for j in range(1, 11)]
    df_result["Modes_transport_utilisés"] = df[mode_cols].apply(
        lambda row: ", ".join([m for m, v in zip(modes, row) if str(v).strip().upper() == "Y"]),
        axis=1
    )

    # 4) durée
    df_result["Duree_trajet_minutes"] = df["Trajet2"].apply(
        lambda x: round(float(x) * 60) if pd.notna(x) and float(x) < 10 else x
    )

    # 5) trafic
    df_result["Trafic_percu"] = df["Trajet3"].apply(
        lambda x: int(str(x).lstrip("A")) - 1 if pd.notna(x) and str(x).startswith("A") else None
    )

    return df_result


def calcul_roue_differences(df_questionnaire):
    """
    Calcule les différences entre les roues émotionnelles.

    Pour Session 1 : seulement PreTest - PostTest (impact des tests)
    Pour Sessions 2 et 3 :
        - PreTrajet - PreTest (impact du trajet)
        - PreTrajet - PostTest (impact total)
        - PreTest - PostTest (impact des tests)
    """
    # 1) copie
    df = df_questionnaire.copy()
    df2 = id_session(df)

    # 5) coder en chiffres la session
    df2["Session"] = df2["Session_label"].map({"A1": 1, "A2": 2, "A3": 3})

    # récupérer uniquement les colonnes de roue
    roue_cols = [c for c in df.columns
                 if c.startswith("RouePreTest")
                 or c.startswith("RouePostTest")
                 or c.startswith("RouePre[")]

    # 7) créer un DataFrame séparé pour les valeurs de roue, en appliquant la conversion A1→5… uniquement là
    conv = {"A1": 5, "A2": 4, "A3": 3, "A4": 2, "A5": 1, "A6": 0}
    df_roues_vals = df[roue_cols].replace(conv)

    # 8) concaténer ID/session et valeurs de roue
    df2 = pd.concat([df2, df_roues_vals.reset_index(drop=True)], axis=1)

    # 9) construire la sortie avec les différences
    sq2emo = {
      "SQ001": "Interet", "SQ002": "Joie", "SQ003": "Energie", "SQ004": "Serenite",
      "SQ005": "Compassion", "SQ006": "Colere", "SQ007": "Ennui", "SQ008": "Peur",
      "SQ009": "Anxiete", "SQ010": "Tristesse"
    }

    out = []
    for _, row in df2.iterrows():
        sn = row["Session"]
        if pd.isna(sn):
            continue
        sn = int(sn)
        base = {"ID": row["ID"], "Session": sn}

        # Pour chaque émotion, calculer les différences
        for code, emo in sq2emo.items():
            col_pretest = f"RouePreTest[{code}]"
            col_posttest = f"RouePostTest[{code}]"
            col_pretrajet = f"RouePre[{code}]"

            # Valeurs des roues
            val_pretest = row.get(col_pretest)
            val_posttest = row.get(col_posttest)
            val_pretrajet = row.get(col_pretrajet)

            # Session 1 : seulement PreTest - PostTest
            if sn == 1:
                if pd.notna(val_pretest) and pd.notna(val_posttest):
                    base[f"{sn}_roue_diff_impact-tests_{emo}"] = val_pretest - val_posttest
                else:
                    base[f"{sn}_roue_diff_impact-tests_{emo}"] = np.nan

            # Sessions 2 et 3 : les trois différences
            else:
                # Impact du trajet : PreTrajet - PreTest
                if pd.notna(val_pretrajet) and pd.notna(val_pretest):
                    base[f"{sn}_roue_diff_impact-trajet_{emo}"] = val_pretrajet - val_pretest
                else:
                    base[f"{sn}_roue_diff_impact-trajet_{emo}"] = np.nan

                # Impact total : PreTrajet - PostTest
                if pd.notna(val_pretrajet) and pd.notna(val_posttest):
                    base[f"{sn}_roue_diff_impact-total_{emo}"] = val_pretrajet - val_posttest
                else:
                    base[f"{sn}_roue_diff_impact-total_{emo}"] = np.nan

                # Impact des tests : PreTest - PostTest
                if pd.notna(val_pretest) and pd.notna(val_posttest):
                    base[f"{sn}_roue_diff_impact-tests_{emo}"] = val_pretest - val_posttest
                else:
                    base[f"{sn}_roue_diff_impact-tests_{emo}"] = np.nan

        out.append(base)

    return pd.DataFrame(out)


############################################
#### Traitement d'une campagne complète ####
############################################

def traiter_campagne(config):
    """Reproduit le traitement de Q_trait_passation_v4.py pour une campagne."""
    print(f"\n=== {config['nom']} ===")
    print(f"  Lecture : {config['csv']}")

    tache_map = {1: "1-Back", 2: "2-Back", 3: "ANT"}

    # --- Chargement et préparation du questionnaire
    df_all = pd.read_csv(config["csv"], sep=';', encoding='utf-8', quotechar='"', engine='python')
    df_questionnaire = df_all.iloc[:, 5:]   # on garde Cahier1 en position 0
    col_ID = df_questionnaire.columns[0]
    participants = config["participants"]

    # --- Construction du df_final pour ordre tâches + conditions
    df_final = pd.DataFrame()
    for pid in participants:
        df_p = df_questionnaire[df_questionnaire[col_ID] == pid]
        if df_p.empty:
            continue

        # Ordre des tâches
        tache_cols = [c for c in df_questionnaire.columns if re.match(r"Cahier3\[SQ\d{3}\]", c)]
        ex = df_p.iloc[0]
        taches = [tache_map.get(ex[c], str(ex[c])) for c in tache_cols]
        base = {"ID": pid, "Ordre_tache": " / ".join(taches)}

        # Ordre des conditions
        val = df_p["Cahier4"].dropna().astype(str).values
        if len(val) > 0 and val[0] in ("A1", "A2"):
            base["Ordre_conditions"] = "Fort/Faible" if val[0] == "A1" else "Faible/Fort"
        else:
            base["Ordre_conditions"] = "Inconnu"

        df_final = pd.concat([df_final, pd.DataFrame([base])], ignore_index=True)

    # --- Calcul et fusion des roues émotionnelles (avec différences)
    cols_utiles_roues = ["Cahier1", "Cahier2"] + [col for col in df_all.columns if col.startswith("Roue")]
    df_roue_input = df_all[cols_utiles_roues].copy()
    df_roues = calcul_roue_differences(df_roue_input)
    df_roues = df_roues.groupby("ID", as_index=False).first()

    df_final = df_final.merge(df_roues, on="ID", how="left")

    # --- Calcul et fusion des trajets (3 passations)
    cols_trajet = ["Cahier1", "Cahier2"] + [col for col in df_all.columns if col.startswith("Trajet")]
    df_trajet_input = df_all[cols_trajet].copy()
    df_trajet_long = calcul_trajet(df_trajet_input)
    df_trajet_long = df_trajet_long.groupby(["ID", "Session"], as_index=False).first()

    for i in (1, 2, 3):
        df_t_i = df_trajet_long[df_trajet_long["Session"] == i][["ID",
            "Modes_transport_utilisés", "Duree_trajet_minutes", "Trafic_percu"]].copy()

        df_t_i = df_t_i.rename(columns={
            "Modes_transport_utilisés": f"{i}_Modes_transport_utilisés",
            "Duree_trajet_minutes":     f"{i}_Duree_trajet_minutes",
            "Trafic_percu":             f"{i}_Trafic_percu"
        })

        df_final = df_final.merge(df_t_i, on="ID", how="left")

    # --- Calcul et fusion des post-questionnaires
    cols_postP = ["Cahier1", "Cahier2"] + [col for col in df_all.columns if col.startswith("PostP")]
    df_postP_input = df_all[cols_postP].copy()
    df_postP_long = calcul_postP(df_postP_input)
    df_postP_long = df_postP_long.groupby(["ID", "Session"], as_index=False).first()

    for i in (1, 2, 3):
        df_p_i = df_postP_long[df_postP_long["Session"] == i].copy()
        rename_cols = {col: f"{i}_{col}" for col in df_p_i.columns if col not in ["ID", "Session"]}
        df_p_i = df_p_i.rename(columns=rename_cols)
        df_p_i = df_p_i.drop(columns=["Session"])
        df_final = df_final.merge(df_p_i, on="ID", how="left")

    # --- Calcul et fusion des informations préalables (InfoP)
    cols_infoP = ["Cahier1", "Cahier2"] + [col for col in df_all.columns if col.startswith("InfoP")]
    df_infoP_input = df_all[cols_infoP].copy()
    df_infoP_wide = calcul_infoP(df_infoP_input)
    df_final = df_final.merge(df_infoP_wide, on="ID", how="left")

    # --- Nettoyage final
    df_final = df_final.groupby("ID", as_index=False).first()
    if "Session" in df_final.columns:
        df_final = df_final.drop(columns="Session")

    return df_final


def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    for config in CAMPAGNES:
        df_final = traiter_campagne(config)

        # --- Export du fichier COMPLET (avec les 3 sessions)
        chemin_complet = os.path.join(DOSSIER_SORTIE, config["sortie_complet"])
        df_final.to_csv(chemin_complet, index=False, encoding="utf-8-sig")
        print(f"  ✓ Fichier complet créé : {chemin_complet}")

        # --- Création du fichier avec UNIQUEMENT les sessions 2 et 3
        # On filtre toutes les colonnes qui commencent par "1_"
        colonnes_session1 = [col for col in df_final.columns if col.startswith("1_")]
        df_sessions_2_3 = df_final.drop(columns=colonnes_session1)

        chemin_s23 = os.path.join(DOSSIER_SORTIE, config["sortie_s23"])
        df_sessions_2_3.to_csv(chemin_s23, index=False, encoding="utf-8-sig")
        print(f"  ✓ Fichier sessions 2-3 créé : {chemin_s23}")

        print(f"  📊 Participants traités : {len(df_final)} | "
              f"colonnes complet : {len(df_final.columns)} | "
              f"colonnes sessions 2-3 : {len(df_sessions_2_3.columns)} "
              f"(session 1 retirée : {len(colonnes_session1)})")


if __name__ == "__main__":
    main()
