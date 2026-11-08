# -*- coding: utf-8 -*-
"""
Traitement UNIFIÉ des questionnaires "vie" (mode de vie) des deux campagnes.

Ce script remplace les deux Q_traitement.py quasi identiques :
  - Campagne_aut_24\\Cognition\\Questionnaires\\Q_traitement.py
  - Campagne_print_25\\Cognition\\Questionnaires\\Q_traitement.py

Il lit les CSV bruts LimeSurvey dans les dossiers de chaque campagne, calcule les
indicateurs (identité, adresse, exposition, activité physique, sommeil,
médication, consommations, personnalité, ressenti, stress, alimentation), puis
exporte un tableau "1 ligne / sujet" par campagne dans
<racine>\\Data\\Cognition\\Questionnaire (lu ensuite par format_all3.py).

Adresses : domicile PEL1 -> Ville_domicile/Adresse_domicile ; lieu de travail ou d'étude
PEL8 -> Ville_travail/Adresse_travail. CORRIGÉ 2026-09-07 : PEL8 était exporté sous
« Ville »/« Adresse » et documenté à tort comme le domicile (PEL8 = adresse professionnelle)
est désormais extraite et ajoutée aux tableaux finaux.

Différences entre campagnes (gérées via CAMPAGNES ci-dessous, comportement
historique de chaque script reproduit à l'identique) :
  - Année de référence pour l'âge : 2024 (automne) / 2025 (printemps).
  - Activité physique de loisir (APYL) : l'automne stocke des heures/semaine
    (APYL3A utilisé directement), le printemps stocke jours/semaine (APYL3A)
    × durée (APYL4A).
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402

import os
import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

#####################################
#### Configuration des campagnes ####
#####################################

DOSSIER_SORTIE = D(r"Data\Cognition\Questionnaire")

CAMPAGNES = [
    {
        "nom": "Q1 - automne 2024",
        "csv": D(r"Campagne_aut_24\Cognition\Questionnaires\questionnaireTRAPCo1_2.csv"),
        "sortie": "resultats_Q1-vie.xlsx",
        "annee_ref": 2024,
        # Automne : APYL3A = heures/semaine de loisir intense (utilisé directement)
        "vpa_loisir_x_duree": False,
    },
    {
        "nom": "Q2 - printemps 2025",
        "csv": D(r"Campagne_print_25\Cognition\Questionnaires\questionnaire_vie2.csv"),
        "sortie": "resultats_Q2-vie.xlsx",
        "annee_ref": 2025,
        # Printemps : APYL3A = jours/semaine, APYL4A = durée/séance -> produit
        "vpa_loisir_x_duree": True,
    },
]

# Valeurs de texte à considérer comme adresse manquante (réponse vide / refus)
ADRESSE_MANQUANTE = {"", "nan", "none", "x", "xx", "xxx", "-", "/", "aucune"}


##############################
#### Fonctions de calcul  ####
##############################

def calcul_ID(df_ID, annee_ref):
    # Copie du DataFrame pour éviter de modifier l'original
    df_ID = df_ID.copy()

    # Création du DataFrame résultat
    df_result = pd.DataFrame(index=df_ID.index)
    df_result["ID"] = df_ID["ID"]

    # Normalisation de la taille (ID3)
    def clean_height(value):
        if isinstance(value, float) and 0 < value < 2:
            return int(value * 100)  # Convertir m en cm
        elif isinstance(value, str):
            return int(value.replace(",", "").replace(".", ""))  # Nettoyage des séparateurs
        return int(value)  # Déjà un entier

    Taille_cm = df_ID["ID3"].apply(clean_height)
    df_result["Taille_m"] = Taille_cm / 100  # Conversion en mètres

    # Transformation du sexe (ID2)
    df_result["Sexe"] = df_ID["ID2"].map({"A1": "Femme", "A2": "Homme"})

    # Calcul de l'âge (ID1)
    def clean_birth_year(value):
        try:
            year = int(str(value)[-4:])  # Extraire les 4 derniers chiffres
            return year if 1980 <= year <= 2010 else None
        except ValueError:
            return None

    Année_naissance = df_ID["ID1"].apply(clean_birth_year)
    df_result["Âge"] = annee_ref - Année_naissance

    # Calcul de l'IMC
    df_result["Poids"] = pd.to_numeric(df_ID["ID4"], errors="coerce")  # Conversion en numérique
    df_result["IMC"] = df_result["Poids"] / (df_result["Taille_m"] ** 2)

    # Calcul de l'IMG
    sexe_num = df_result["Sexe"].map({"Femme": 0, "Homme": 1})
    df_result["IMG"] = (1.20 * df_result["IMC"]) + (0.23 * df_result["Âge"]) - (10.8 * sexe_num) - 5.4

    # Détermination du diplôme le plus élevé (ID5[SQ001] à ID5[SQ008])
    diplomes = [
        "Brevet", "CAP, BEP", "Baccalauréat", "DEUG, BTS, DUT, DEUST",
        "Licence", "Maîtrise", "Master", "Doctorat"
    ]
    cols_diplomes = [f"ID5[SQ00{i}]" for i in range(1, 9)]

    def get_highest_degree(row):
        for i, col in reversed(list(enumerate(cols_diplomes))):
            if row[col] == "Y":
                return diplomes[i]
        return "Aucun"

    df_result["Diplôme_max"] = df_ID[cols_diplomes].apply(get_highest_degree, axis=1)

    return df_result


def calcul_adresse(df_adresse):
    """Extrait les adresses déclarées par le participant.

    PEL1[SQ001] / PEL1[SQ002] -> Ville_domicile / Adresse_domicile
        (« 1. Adresse du lieu de vie » : ville / voie)
    PEL8[SQ001] / PEL8[SQ002] -> Ville_travail / Adresse_travail
        (« 8. Adresse professionnelle principale (ou lieu d'étude ou d'activité) » ;
        question posée seulement si la profession PEL7 n'est pas « x » = sans activité,
        et « si principalement en télétravail, renseigner à nouveau votre adresse »)

    Jusqu'au 2026-09-07, seul PEL8 était exporté (colonnes « Ville »/« Adresse »)
    et documenté à tort comme le domicile.
    Les réponses vides ou non renseignées (ex. "X", "aucune") sont mises à NaN.
    """
    df_adresse = df_adresse.copy()
    df_result = pd.DataFrame(index=df_adresse.index)
    df_result["ID"] = df_adresse["ID"]

    def _nettoyer(col):
        serie = df_adresse[col].astype(str).str.strip()
        masque_vide = serie.str.lower().isin(ADRESSE_MANQUANTE)
        return serie.mask(masque_vide, np.nan)

    df_result["Ville_domicile"] = _nettoyer("PEL1[SQ001]")
    df_result["Adresse_domicile"] = _nettoyer("PEL1[SQ002]")
    df_result["Ville_travail"] = _nettoyer("PEL8[SQ001]")
    df_result["Adresse_travail"] = _nettoyer("PEL8[SQ002]")

    return df_result


# Ajouter profession / Lieu de travail / salaire
def calcul_expo(df_expo):
    # Calcule de l'indice d'exposition environnementale :
        # Colonnes PEL1B[SQ001] et PEL1B[SQ002] si il y a une valeur dans la case = 3 (Soit 6 si une valeurs dans chaque)
        # Colonne PEL1B[SQ003] = -1
        # Colonne PEL1B[SQ004] = 0
        # Colonne PEL1B[SQ005] = 2

    # Calcule de l'indice d'exposition domestique :
        # Colonnes PEL2[PEL801], PEL2[PEL802], PEL2[PEL803], PEL2[PEL805], PEL2[PEL807], PEL2[PEL808] et PEL2[PEL809] si il y a une valeur dans la case = 2 (chaque)
        # Colonnes PEL2[PEL804] et PEL2[PEL806] si il y a une valeur dans la case = 1 (chaque)
        # Colonne PEL2[PEL8010] si il y a une valeur dans la case = 0 (si il y a une valeurs dans cette case, aucune autre des PEL2[PEL8...] ne devrais en avoir une)

        # Colonne PEL3 correspondance des valeurs : [A1 = 3 ; A2 = 1 ; A3, A4 et A5 = 0]

        # Colonnes PEL4[SQ001], PEL4[SQ002] et PEL4[SQ003] si il y a une valeur dans la case = 3
        # Colonnes PEL4[SQ004], PEL4[SQ005] et PEL4[SQ006] si il y a une valeur dans la case = 2
        # Colonne PEL4[SQ007] = 1
        # Colonne PEL4[other] = 0

        # Colonne PEL5 correspondance des valeurs : [A1 = 0 ; A2 = 2 ; A3 = 5]

        # Colonne PEL6 correspondance des valeurs : ["N" = 0 ; "Y" = 3]

    # Calcule de l'indice d'exposition professionnelle :
        # Colonne PEL9 correspondance des valeurs : ["N" = 0 ; "Y" = 5]
        # Colonne PEL9B correspondance des valeurs : [A1 = 5 ; A2 = 3 ; A3 = 2 ; A4 = 1]
        # Colonnes PEL9C[SQ001], PEL9C[SQ002] et PEL9C[SQ003] si il y a une valeur dans les cases respectives : SQ001 = 3 / SQ002 et SQ003 = 2

    df_expo = df_expo.copy()
    df_result = pd.DataFrame(index=df_expo.index)
    df_result["ID"] = df_expo["ID"]

    # --- EXPOSITION ENVIRONNEMENTALE ---
    score_env = (
        df_expo["PEL1B[SQ001]"].fillna("").eq("Y").astype(int) * 3 +
        df_expo["PEL1B[SQ002]"].fillna("").eq("Y").astype(int) * 3 +
        df_expo["PEL1B[SQ003]"].fillna("").eq("Y").astype(int) * (-1) +
        df_expo["PEL1B[SQ005]"].fillna("").eq("Y").astype(int) * 2
        # PEL1B[SQ004] = 0 → ignorée
    )

    # --- EXPOSITION DOMESTIQUE ---
    pel8_2pts = ["PEL2[PEL801]", "PEL2[PEL802]", "PEL2[PEL803]",
                 "PEL2[PEL805]", "PEL2[PEL807]", "PEL2[PEL808]", "PEL2[PEL809]"]
    pel8_1pt = ["PEL2[PEL804]", "PEL2[PEL806]"]

    score_dom = df_expo[pel8_2pts].fillna("").eq("Y").sum(axis=1) * 2
    score_dom += df_expo[pel8_1pt].fillna("").eq("Y").sum(axis=1) * 1

    # Si PEL2[PEL8010] contient "Y", on pourrait ignorer les autres — ici, on l'ajoute comme présence (score = 0)
    score_dom += df_expo["PEL2[PEL8010]"].fillna("").eq("Y").astype(int) * 0  # à adapter si tu veux qu'elle annule les autres

    # PEL3 : A1 = 3 ; A2 = 1 ; A3-A5 = 0
    score_dom += df_expo["PEL3"].map({"A1": 3, "A2": 1}).fillna(0)

    # PEL4
    score_dom += (
        df_expo["PEL4[SQ001]"].fillna("").eq("Y").astype(int) * 3 +
        df_expo["PEL4[SQ002]"].fillna("").eq("Y").astype(int) * 3 +
        df_expo["PEL4[SQ003]"].fillna("").eq("Y").astype(int) * 3 +
        df_expo["PEL4[SQ004]"].fillna("").eq("Y").astype(int) * 2 +
        df_expo["PEL4[SQ005]"].fillna("").eq("Y").astype(int) * 2 +
        df_expo["PEL4[SQ006]"].fillna("").eq("Y").astype(int) * 2 +
        df_expo["PEL4[SQ007]"].fillna("").eq("Y").astype(int) * 1
        # PEL4[other] = 0
    )

    # PEL5 : A1 = 0 ; A2 = 2 ; A3 = 5
    score_dom += df_expo["PEL5"].map({"A2": 2, "A3": 5}).fillna(0)

    # PEL6 : "Y" = 3 ; "N" = 0
    score_dom += df_expo["PEL6"].map({"Y": 3, "N": 0}).fillna(0)

    # --- EXPOSITION PROFESSIONNELLE ---
    score_pro = df_expo["PEL9"].map({"Y": 5, "N": 0}).fillna(0)
    score_pro += df_expo["PEL9B"].map({"A1": 5, "A2": 3, "A3": 2, "A4": 1}).fillna(0)

    score_pro += (
        df_expo["PEL9C[SQ001]"].fillna("").eq("Y").astype(int) * 3 +
        df_expo["PEL9C[SQ002]"].fillna("").eq("Y").astype(int) * 2 +
        df_expo["PEL9C[SQ003]"].fillna("").eq("Y").astype(int) * 2
    )

    # Assemblage
    df_result["expo_env"] = score_env
    df_result["expo_dom"] = score_dom
    df_result["expo_pro"] = score_pro
    df_result["expo_total"] = score_env + score_dom + score_pro

    return df_result


def calcul_actiphy(df_actiphy, vpa_loisir_x_duree):
    # Attention :
        # colonnes en heures à convertir en minutes : APYT3 / APYL3B / APYL5B
        # Certaines des valeurs renseigné dans les équations seront vide pour certains participant, faire en sorte que le calule s'execute sans ces valeurs

    # Mesure des temps hebdomadaires d’activités physiques d’intensité modérée et de forte en min/semaine
    # Durée hebdomadaire actiphy forte intensité (VPA) = (APYT1A * APYT1Bb) + (APYL3A)
    # Durée hebdomadaire actiphy intensité modérée (MPA) = (APYT2A * APYT2B) + (APYD3A * APYD4A) + (APYD3B * APYD4B) + (APYL3B)
    # Durée hebdo d'actiphy d'intensité modéré et forte (MVPA) = VPA + MPA
    # Si MVPA < 150 min/semaine => Inactif
    # Si MVPA >= 150 min/semaine => Actif

    # Calcul en METs.min/semaine (volume d’activité physique hebdomadaire total) :
    # VPA_met = 8*((APYT1A * APYT1Bb) + (APYL3A))
    # MPA_met = 4*((APYT2A * APYT2B) + (APYL3B)) + 3.3*(APYD3A * APYD4A) + 6*(APYD3B * APYD4B)
    # Activitéphy totale APtot = MPA_met + VPA_met
    # TRES ACTIF :
        # Actiphy intense au moins 3j/semaine avec un minimum de 1500 METs.min/semaine
        # ou actiphy quotidienne modéré ou intense avec un minimum de 3000 METs.min/semaine
    # ACTIF :
        # 30 à 60 min d'actiphy de forte intensité au moins 3j/semaine
        # 30 min minimum par jour d'activité d'intensité modéré ou marche, 5j ou +/semaine
        # Ayant un APtot >= 600 METs.min/semaine
    # INACTIFS : Sujet ayant moins de 600 METs.min/semaine

    # Comportement sédentaire :
    # Temps sédentaire (SB) = APYT3 + ((APYD5A * APYD5B)+ APYL5A + APYL5B) /7 => donne des minutes/j
    # Sédentarité faible : SB < 3h/j
    # Sédentarité modéré : 3h/j <= SB <= 7h/j
    # Sédentarité élevé : SB > 7h/j

    df_actiphy = df_actiphy.copy()
    df_result = pd.DataFrame(index=df_actiphy.index)
    df_result["ID"] = df_actiphy["ID"]

    df_actiphy = df_actiphy.fillna(0)

    # Conversion des heures en minutes pour les colonnes concernées
    colonnes_heures = ["APYT3", "APYL3A", "APYL3B", "APYL5B"]
    df_actiphy[colonnes_heures] = df_actiphy[colonnes_heures] * 60

    # Activité physique de loisir intense :
    #   - automne  : APYL3A = heures/semaine (utilisé directement, déjà *60)
    #   - printemps: APYL3A = jours/semaine, APYL4A = durée -> produit
    if vpa_loisir_x_duree and "APYL4A" in df_actiphy.columns:
        loisir_vpa = df_actiphy["APYL3A"] * df_actiphy["APYL4A"]
    else:
        loisir_vpa = df_actiphy["APYL3A"]

    # Calcul des durées hebdomadaires d'activités physiques
    df_result["VPA"] = (df_actiphy["APYT1A"] * df_actiphy["APYT1Bb"]) + loisir_vpa
    df_result["MPA"] = ((df_actiphy["APYT2A"] * df_actiphy["APYT2B"]) + (df_actiphy["APYD3A"] * df_actiphy["APYD4A"]) +
                        (df_actiphy["APYD3B"] * df_actiphy["APYD4B"]) + df_actiphy["APYL3B"])
    df_result["MVPA"] = df_result["VPA"] + df_result["MPA"]

    # Catégorisation de l'activité physique
    df_result["statut_actiphy"] = np.where(df_result["MVPA"] < 150, "Inactif", "Actif")

    # Calcul en METs.min/semaine
    df_result["VPA_met"] = 8 * df_result["VPA"]
    df_result["MPA_met"] = (4 * (df_actiphy["APYT2A"] * df_actiphy["APYT2B"]) + 3.3 * (df_actiphy["APYD3A"] * df_actiphy["APYD4A"]) +
                            6 * (df_actiphy["APYD3B"] * df_actiphy["APYD4B"]) + 4 * df_actiphy["APYL3B"])
    df_result["APtot"] = df_result["VPA_met"] + df_result["MPA_met"]

    # Catégorisation du niveau d'activité physique
    df_result["niveau_actiphy"] = np.select(
        [(df_result["VPA"] >= 3) & (df_result["APtot"] >= 1500),
        (df_result["APtot"] >= 3000),
        (df_result["APtot"] >= 600)],
        ["Très Actif", "Très Actif", "Actif"], default="Inactif"
    )

    # Calcul du comportement sédentaire
    df_result["SB"] = (df_actiphy["APYT3"] + ((df_actiphy["APYD5A"] * df_actiphy["APYD5B"]) + df_actiphy["APYL5A"] + df_actiphy["APYL5B"])) / 7

    # Catégorisation du temps sédentaire
    df_result["statut_sedentaire"] = np.select(
        [df_result["SB"] < 180,
        (df_result["SB"] >= 180) & (df_result["SB"] <= 420),
        df_result["SB"] > 420],
        ["Faible", "Modéré", "Élevé"]
    )

    return df_result


# pas encore implémenté prise en compte de dormir seul ou non et temps d'écran avant sommeil
def calcul_sommeil(df_sommeil):
    df_result = pd.DataFrame(index=df_sommeil.index)
    df_result["ID"] = df_sommeil["ID"]
    df_sommeil = df_sommeil.copy()

    # Convertir les valeurs "A1", "A2", ... en numériques
    conversion_scores = {"A1": 0, "A2": 1, "A3": 2, "A4": 3, "A5": 4, "N": 0, "Y": 1}
    df_sommeil = df_sommeil.replace(conversion_scores).infer_objects(copy=False)

    # Durée de sommeil
    df_result["Dure_sommeil"] = np.select(
        [df_sommeil["PSQI4"] > 7, df_sommeil["PSQI4"] == 7, df_sommeil["PSQI4"] == 6, df_sommeil["PSQI4"] < 6],
        [0, 1, 2, 3]
    )

    # Nombre d'heures passées au lit
    # Traiter le cas où PSQI1 est égal à 24 et le convertir en 0 (minuit)
    df_sommeil["PSQI1"] = df_sommeil["PSQI1"].replace(24, 0)

    # Convertir les heures de coucher (PSQI1) et de lever (PSQI3) en heures décimales
    df_sommeil["PSQI1"] = pd.to_datetime(df_sommeil["PSQI1"], format='%H', errors='coerce').dt.hour
    df_sommeil["PSQI3"] = pd.to_datetime(df_sommeil["PSQI3"], format='%H', errors='coerce').dt.hour

    # Calculer la différence
    df_sommeil["nb_h_lit"] = df_sommeil["PSQI3"] - df_sommeil["PSQI1"]

    # Si l'heure de coucher est après minuit, ajuster la différence
    df_sommeil["nb_h_lit"] = df_sommeil["nb_h_lit"].apply(lambda x: x + 24 if x < 0 else x)

    # Efficacité du sommeil
    df_result["efficacite_sommeil"] = (df_sommeil["PSQI4"] / df_sommeil["nb_h_lit"]) * 100
    df_result["efficacite_sommeil"] = df_result["efficacite_sommeil"].round(0)

    # Score d'efficacité
    df_result["score_efficacite"] = np.select(
        [df_result["efficacite_sommeil"] >= 85,
         (df_result["efficacite_sommeil"] >= 75) & (df_result["efficacite_sommeil"] < 85),
         (df_result["efficacite_sommeil"] >= 65) & (df_result["efficacite_sommeil"] < 75),
         df_result["efficacite_sommeil"] < 65],
        [0, 1, 2, 3]
    )

    # Calcul de la latence du sommeil en fonction de PSQI5
    df_result["Latence_sommeil"] = df_sommeil.apply(
        lambda row: row["PSQI2"] if row["PSQI5"] == 0 else row["PSQI2"] + row["PSQI5B[SQ001]"],
        axis=1
    )

    df_result["Latence_sommeil"] = np.select(
        [df_result["Latence_sommeil"] == 0, df_result["Latence_sommeil"] <= 2, df_result["Latence_sommeil"] <= 4, df_result["Latence_sommeil"] >= 5],
        [0, 1, 2, 3]
    )

    # Calcul des troubles du sommeil
    colonnes_trouble = [f"PSQI5B[SQ00{i}]" for i in range(2, 10) if f"PSQI5B[SQ00{i}]" in df_sommeil.columns]

    df_result["Trouble_sommeil"] = df_sommeil.apply(
        lambda row: 0 if row["PSQI5"] == 0 else row[colonnes_trouble].sum(),
        axis=1
    )

    df_result["Trouble_sommeil"] = np.select(
        [df_result["Trouble_sommeil"] == 0, df_result["Trouble_sommeil"] <= 9, df_result["Trouble_sommeil"] <= 18, df_result["Trouble_sommeil"] >= 19],
        [0, 1, 2, 3]
    )

    # Qualité subjective du sommeil
    df_result["qualité_subjective"] = df_sommeil["PSQI6"]

    # Médication pour dormir
    df_result["médication"] = df_sommeil["PSQI7"]

    # Dysfonctionnement diurne
    df_result["Dysfonction_diurne"] = df_sommeil["PSQI8"] + df_sommeil["PSQI9"]
    df_result["Dysfonction_diurne"] = np.select(
        [df_result["Dysfonction_diurne"] == 0, df_result["Dysfonction_diurne"] <= 2,
         df_result["Dysfonction_diurne"] <= 4, df_result["Dysfonction_diurne"] >= 5],
        [0, 1, 2, 3]
    )

    # Score global PSQI
    df_result["Score_PSQI"] = (
        df_result["Dure_sommeil"] + df_result["score_efficacite"] + df_result["Latence_sommeil"] +
        df_result["Trouble_sommeil"] + df_result["qualité_subjective"] + df_result["médication"] +
        df_result["Dysfonction_diurne"]
    )

    return df_result


def calcul_med(df_med):
    # Copie du DataFrame pour éviter de modifier l'original
    df_med = df_med.copy()

    # Création du DataFrame résultat
    df_result = pd.DataFrame(index=df_med.index)
    df_result["ID"] = df_med["ID"]

    # Condition sur MED1 (ligne par ligne)
    df_result["Intervention_Chir"] = df_med["MED1"].apply(lambda x: "Oui" if x == "Y" else "Non")

    # Condition sur MED2 (ligne par ligne)
    def medication_choice(med, comment):
        if med == "A1":
            return comment
        elif med == "A2":
            return "Non"
        else:
            return "Inconnue"

    df_result["medication"] = df_med.apply(lambda row: medication_choice(row["MED2"], row["MED2[comment]"]), axis=1)

    return df_result


def calcul_conso(df_conso):
    df_conso = df_conso.copy()

    df_result = pd.DataFrame(index=df_conso.index)
    df_result["ID"] = df_conso["ID"]

    substances = {
        "ASSIST0[SQ001]": "nicotine",
        "ASSIST0[SQ002]": "alcool",
        "ASSIST0[SQ003]": "cannabis",
        "ASSIST0[SQ004]": "cocaine",
        "ASSIST0[SQ005]": "amphétamine",
        "ASSIST0[SQ006]": "solvants",
        "ASSIST0[SQ007]": "calmants_somnifères",
        "ASSIST0[SQ008]": "hallucinogènes",
        "ASSIST0[SQ009]": "opiacés",
        "ASSIST0[SQ010]": "gaz",
        "ASSIST0[SQ011]": "autres"
    }

    questions = {
        "ASSIST0[SQ001]": {"q1": ["ASSIST1A[SQ001]", "ASSIST1A[SQ002]", "ASSIST1A[SQ003]", "ASSIST1A[SQ004]"], "q2": ["ASSIST2A[SQ001]", "ASSIST2A[SQ002]"]},
        "ASSIST0[SQ002]": {"q1": ["ASSIST1B[SQ001]", "ASSIST1B[SQ002]", "ASSIST1B[SQ003]", "ASSIST1B[SQ004]"], "q2": ["ASSIST2B[SQ001]", "ASSIST2B[SQ002]"]},
        "ASSIST0[SQ003]": {"q1": ["ASSIST1C[SQ001]", "ASSIST1C[SQ002]", "ASSIST1C[SQ003]", "ASSIST1C[SQ004]"], "q2": ["ASSIST2C[SQ001]", "ASSIST2C[SQ002]"]},
        "ASSIST0[SQ004]": {"q1": ["ASSIST1D[SQ001]", "ASSIST1D[SQ002]", "ASSIST1D[SQ003]", "ASSIST1D[SQ004]"], "q2": ["ASSIST2D[SQ001]", "ASSIST2D[SQ002]"]},
        "ASSIST0[SQ005]": {"q1": ["ASSIST1E[SQ001]", "ASSIST1E[SQ002]", "ASSIST1E[SQ003]", "ASSIST1E[SQ004]"], "q2": ["ASSIST2E[SQ001]", "ASSIST2E[SQ002]"]},
        "ASSIST0[SQ006]": {"q1": ["ASSIST1F[SQ001]", "ASSIST1F[SQ002]", "ASSIST1F[SQ003]", "ASSIST1F[SQ004]"], "q2": ["ASSIST2F[SQ001]", "ASSIST2F[SQ002]"]},
        "ASSIST0[SQ007]": {"q1": ["ASSIST1G[SQ001]", "ASSIST1G[SQ002]", "ASSIST1G[SQ003]", "ASSIST1G[SQ004]"], "q2": ["ASSIST2G[SQ001]", "ASSIST2G[SQ002]"]},
        "ASSIST0[SQ008]": {"q1": ["ASSIST1H[SQ001]", "ASSIST1H[SQ002]", "ASSIST1H[SQ003]", "ASSIST1H[SQ004]"], "q2": ["ASSIST2H[SQ001]", "ASSIST2H[SQ002]"]},
        "ASSIST0[SQ009]": {"q1": ["ASSIST1I[SQ001]", "ASSIST1I[SQ002]", "ASSIST1I[SQ003]", "ASSIST1I[SQ004]"], "q2": ["ASSIST2I[SQ001]", "ASSIST2I[SQ002]"]},
        "ASSIST0[SQ010]": {"q1": ["ASSIST1J[SQ001]", "ASSIST1J[SQ002]", "ASSIST1J[SQ003]", "ASSIST1J[SQ004]"], "q2": ["ASSIST2J[SQ001]", "ASSIST2J[SQ002]"]},
        "ASSIST0[SQ011]": {"q1": ["ASSIST1K[SQ001]", "ASSIST1K[SQ002]", "ASSIST1K[SQ003]", "ASSIST1K[SQ004]"], "q2": ["ASSIST2K[SQ001]", "ASSIST2K[SQ002]"]}
    }

    conversion_q1 = {"A1": 0, "A2": 5, "A3": 6, "A4": 7, "A5": 8}
    conversion_q2 = {"A1": 0, "A2": 6, "A3": 3}

    for key, sub in substances.items():
        if key in df_conso.columns:
            df_result[f"score_{sub}"] = 0
            mask = df_conso[key] == "A1"
            if mask.any():
                for q in questions[key]["q1"]:
                    if q in df_conso.columns:
                        df_result.loc[mask, f"score_{sub}"] += df_conso[q].map(conversion_q1).fillna(0)
                for q in questions[key]["q2"]:
                    if q in df_conso.columns:
                        df_result.loc[mask, f"score_{sub}"] += df_conso[q].map(conversion_q2).fillna(0)
            if sub == "alcool":
                df_result[f"indicateur_{sub}"] = np.select(
                    [df_result[f"score_{sub}"] <= 10, df_result[f"score_{sub}"] <= 26, df_result[f"score_{sub}"] >= 27],
                    ["faible", "modérée", "élevée"]
                )
            else:
                df_result[f"indicateur_{sub}"] = pd.cut(df_result[f"score_{sub}"],
                                                          bins=[-1, 3, 26, float("inf")],
                                                          labels=["faible", "modéré", "élevé"])
    return df_result


# Catégorisé les score pour chaque élément en haut / moyen / bas ?
def calcul_personnalite(df_personnalite):
    # Définition des conversions
    conversion_directe = {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5, "A6": 6, "A7": 7}
    conversion_inverse = {"A1": 7, "A2": 6, "A3": 5, "A4": 4, "A5": 3, "A6": 2, "A7": 1}

    # Colonnes concernées
    cols_directe = ["TIPI0[TIPI1]", "TIPI0[TIPI3]", "TIPI0[TIPI5]", "TIPI0[TIPI7]", "TIPI0[TIPI9]"]
    cols_inverse = ["TIPI0[TIPI2]", "TIPI0[TIPI4]", "TIPI0[TIPI6]", "TIPI0[TIPI8]", "TIPI0[TIPI10]"]

    # Appliquer les conversions
    df_personnalite = df_personnalite.copy()
    df_personnalite.loc[:, cols_directe] = df_personnalite[cols_directe].replace(conversion_directe)
    df_personnalite.loc[:, cols_inverse] = df_personnalite[cols_inverse].replace(conversion_inverse)

    # Création du DataFrame résultat
    df_result = pd.DataFrame(index=df_personnalite.index)
    df_result["ID"] = df_personnalite["ID"]

    # Calcul des scores de personnalité
    df_result["Extraversion"] = df_personnalite[["TIPI0[TIPI1]", "TIPI0[TIPI6]"]].mean(axis=1)
    df_result["Agréabilité"] = df_personnalite[["TIPI0[TIPI2]", "TIPI0[TIPI7]"]].mean(axis=1)
    df_result["Conscienciosité"] = df_personnalite[["TIPI0[TIPI3]", "TIPI0[TIPI8]"]].mean(axis=1)
    df_result["Stabilité_émotionnelle"] = df_personnalite[["TIPI0[TIPI4]", "TIPI0[TIPI9]"]].mean(axis=1)
    df_result["Ouverture"] = df_personnalite[["TIPI0[TIPI5]", "TIPI0[TIPI10]"]].mean(axis=1)

    return df_result


def calcul_ressenti(df_ressenti):
    conversion = {"A1": 0, "A2": 1, "A3": 2, "A4": 3}

    # Appliquer les conversions
    df_ressenti = df_ressenti.copy()
    colonnes = ["PHQ0[SQ001]", "PHQ0[SQ002]", "PHQ0[SQ004]", "PHQ0[SQ005]", "PHQ0[SQ009]", "PHQ0[SQ007]", "PHQ0[SQ008]", "PHQ0[SQ010]", "PHQ0[SQ003]"]
    df_ressenti.loc[:, colonnes] = df_ressenti[colonnes].replace(conversion)

    # Création du DataFrame résultat
    df_result = pd.DataFrame(index=df_ressenti.index)
    df_result["ID"] = df_ressenti["ID"]

    # Calcul du score total PHQ
    df_result["Score_PHQ"] = df_ressenti[colonnes].sum(axis=1)

    # Définition des catégories de dépression
    df_result["indicateur_depression"] = np.select(
        [df_result["Score_PHQ"] <= 4,
         (df_result["Score_PHQ"] >= 5) & (df_result["Score_PHQ"] <= 9),
         (df_result["Score_PHQ"] >= 10) & (df_result["Score_PHQ"] <= 14),
         (df_result["Score_PHQ"] >= 15) & (df_result["Score_PHQ"] <= 19),
         df_result["Score_PHQ"] >= 20],
        ["Non", "Bénin", "modéré", "élevé", "sévère"]
    )

    return df_result


def calcul_stress(df_stress):
    # Définition des conversions
    conversion_directe = {"A1": 4, "A2": 3, "A3": 2, "A4": 1, "A5": 0}
    conversion_inverse = {"A1": 0, "A2": 1, "A3": 2, "A4": 3, "A5": 4}

    # Colonnes concernées
    cols_directe = ["PSS0[PSS4]", "PSS0[PSS5]", "PSS0[PSS7]", "PSS0[PSS8]"]
    cols_inverse = ["PSS0[PSS1]", "PSS0[PSS2]", "PSS0[PSS3]", "PSS0[PSS6]", "PSS0[PSS9]", "PSS0[PSS10]"]

    # Appliquer les conversions
    df_stress = df_stress.copy()
    df_stress.loc[:, cols_directe] = df_stress[cols_directe].replace(conversion_directe)
    df_stress.loc[:, cols_inverse] = df_stress[cols_inverse].replace(conversion_inverse)

    # Création du DataFrame résultat
    df_result = pd.DataFrame(index=df_stress.index)
    df_result["ID"] = df_stress["ID"]

    # Calcul du score total PSS
    df_result["Score_PSS"] = df_stress[cols_directe + cols_inverse].sum(axis=1)

    # Définition des catégories de stress
    df_result["indicateur_stress"] = np.select(
        [df_result["Score_PSS"] <= 13,
         (df_result["Score_PSS"] >= 14) & (df_result["Score_PSS"] <= 26),
         df_result["Score_PSS"] >= 27],
        ["faible", "modéré", "élevé"]
    )

    return df_result


def calcul_alim(df_alim):
    # Copie du DataFrame pour éviter de modifier l'original
    df_alim = df_alim.copy()

    # Création du DataFrame résultat
    df_result = pd.DataFrame(index=df_alim.index)
    df_result["ID"] = df_alim["ID"]

    # Définition de l'échelle de conversion
    conversion = {"A1": 5, "A2": 3, "A3": 1, "A4": 0}

    # Colonnes des antioxydants et pro-oxydants
    antiox_columns = ["FFQ1[SQ001]", "FFQ1[SQ002]", "FFQ1[SQ003]", "FFQ1[SQ004]", "FFQ1[SQ005]", "FFQ1[SQ006]"]
    proox_columns = ["FFQ1[SQ008]", "FFQ1[SQ009]", "FFQ1[SQ010]", "FFQ1[SQ011]", "FFQ1[SQ012]"]

    # Conversion des réponses en scores
    df_alim.replace(conversion, inplace=True)

    # Création des ratios vides
    ratio_antiox = []
    ratio_proox = []

    # Traitement ligne par ligne
    for i, row in df_alim.iterrows():
        antiox_list = row[antiox_columns].dropna().tolist()
        proox_list = row[proox_columns].dropna().tolist()

        # Gestion du FFQ1[SQ007]
        if row["FFQ1[SQ007]"] == 5:  # Pro-oxydant
            proox_list.append(3)
        elif row["FFQ1[SQ007]"] == 1:  # Antioxydant
            antiox_list.append(3)

        # Calcul correct du ratio en ajustant bien le dénominateur
        ratio_antiox.append(sum(antiox_list) / len(antiox_list) if antiox_list else np.nan)
        ratio_proox.append(sum(proox_list) / len(proox_list) if proox_list else np.nan)

    # Stockage des résultats
    df_result["Ratio_Antioxydants"] = ratio_antiox
    df_result["Ratio_Pro-oxydants"] = ratio_proox
    df_result["Score_Nutrition"] = df_result["Ratio_Antioxydants"] - df_result["Ratio_Pro-oxydants"]

    return df_result


############################################
#### Traitement d'une campagne complète ####
############################################

def traiter_campagne(config):
    """Lit un CSV brut de campagne et renvoie le tableau "1 ligne / sujet"."""
    print(f"\n=== {config['nom']} ===")
    print(f"  Lecture : {config['csv']}")

    df_questionnaire = pd.read_csv(config["csv"], sep=';', encoding="utf-8-sig")

    # Suppression des 5 premières colonnes (métadonnées LimeSurvey : id, submitdate, ...)
    df_questionnaire = df_questionnaire.iloc[:, 5:]

    # Identification de la colonne contenant l'ID des participants (= "ID")
    col_ID = df_questionnaire.columns[0]
    # Nettoyage des ID : suppression des lettres pour ne garder que les chiffres
    df_questionnaire[col_ID] = df_questionnaire[col_ID].astype(str).str.extract(r'(\d+)').astype(int)

    # Création des sous-tableaux en filtrant les colonnes par préfixe
    df_ID = df_questionnaire.filter(like="ID")
    df_expo = df_questionnaire[[col_ID] + list(df_questionnaire.filter(like="PEL").columns)]
    df_actiphy = df_questionnaire[[col_ID] + list(df_questionnaire.filter(like="APY").columns)]
    df_sommeil = df_questionnaire[[col_ID] + list(df_questionnaire.filter(like="PSQI").columns)]
    df_med = df_questionnaire[[col_ID] + list(df_questionnaire.filter(like="MED").columns)]
    df_conso = df_questionnaire[[col_ID] + list(df_questionnaire.filter(like="ASSIST").columns)]
    df_personnalite = df_questionnaire[[col_ID] + list(df_questionnaire.filter(like="TIPI").columns)]
    df_ressenti = df_questionnaire[[col_ID] + list(df_questionnaire.filter(like="PHQ0").columns)]
    df_stress = df_questionnaire[[col_ID] + list(df_questionnaire.filter(like="PSS0").columns)]
    df_alim = df_questionnaire[[col_ID] + list(df_questionnaire.filter(like="FFQ").columns)]
    # Adresses : colonnes explicites (filter(like="PEL8") capturerait aussi PEL2[PEL80x])
    df_adresse = df_questionnaire[[col_ID, "PEL1[SQ001]", "PEL1[SQ002]", "PEL8[SQ001]", "PEL8[SQ002]"]]

    # Calculs
    df_result_ID = calcul_ID(df_ID, config["annee_ref"])
    df_result_adresse = calcul_adresse(df_adresse)
    df_result_expo = calcul_expo(df_expo)
    df_result_actiphy = calcul_actiphy(df_actiphy, config["vpa_loisir_x_duree"])
    df_result_sommeil = calcul_sommeil(df_sommeil)
    df_result_med = calcul_med(df_med)
    df_result_conso = calcul_conso(df_conso)
    df_result_personnalite = calcul_personnalite(df_personnalite)
    df_result_ressenti = calcul_ressenti(df_ressenti)
    df_result_stress = calcul_stress(df_stress)
    df_result_alim = calcul_alim(df_alim)

    df_final = pd.concat(
        [df_result_ID, df_result_adresse, df_result_expo, df_result_actiphy, df_result_sommeil,
         df_result_med, df_result_conso, df_result_personnalite, df_result_ressenti,
         df_result_stress, df_result_alim],
        axis=1
    )

    # S'assurer que la colonne ID n'est pas dupliquée
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    return df_final


def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    for config in CAMPAGNES:
        df_final = traiter_campagne(config)
        chemin_sortie = os.path.join(DOSSIER_SORTIE, config["sortie"])
        df_final.to_excel(chemin_sortie, index=False)
        n_dom = df_final[["Ville_domicile", "Adresse_domicile"]].notna().any(axis=1).sum()
        n_trav = df_final[["Ville_travail", "Adresse_travail"]].notna().any(axis=1).sum()
        print(f"  [OK] {len(df_final)} sujets, {len(df_final.columns)} colonnes "
              f"({n_dom} avec domicile, {n_trav} avec lieu de travail)")
        print(f"  -> {chemin_sortie}")


if __name__ == "__main__":
    main()
