
import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402
import os
import glob
import pandas as pd
import re
from datetime import datetime

# --- Chemins de base ---
chemin_EEG = D(r"Data\Cognition\EEG\Study_all\Résultats_all")
chemin_cognition = D(r"Data\Cognition\Tests-cog\Resultats")
chemin_questionnaire = D(r"Data\Cognition\Questionnaire")
chemin_exposition = D(r"Data\Exposition")
dossier_sortie = D(r"Data\Table_combine")
chemin_base_ref = D(r"Data")

# Dossiers de données brutes capteurs (Température / Humidité) par campagne
chemin_aggr_aut = D(r"Campagne_aut_24\Trajets\Aggregated")
chemin_aggr_print = D(r"Campagne_print_25\Trajets\Aggregated")

# --- Fichiers par catégorie ---
fichiers_cognition = [
    "Resultats_1B_camp1.csv",
    "Resultats_1B_camp2.csv",
    "Resultats_2B_camp1.csv",
    "Resultats_2B_camp2.csv",
    "Resultats_ANT_camp1.xlsx",
    "Resultats_ANT_camp2.xlsx"
]

fichiers_eeg = [
    "1B_N1_P3_extraction.csv",
    "2B_N1_P3_extraction.csv",
    "ANT_indicateurs_ERP_multi_triggers.xlsx"  # Fichier avec indicateurs déjà calculés
]

fichiers_questionnaire = [
    "resultats_Q1-passation_complet.csv",
    "resultats_Q1-vie.xlsx",
    "resultats_Q2-passation_complet.csv",
    "resultats_Q2-vie.xlsx"
]

fichiers_exposition = [
    "Concentration_Estimators_Only.xlsx",
    "Concentration_Integration_Only.xlsx",
    "InhaledDoseComparisons_Fall&Winter.xlsx"
]

fichier_acoustique = "Leq_parcours.csv"


MAPPING_ETAT = {
    "0": "0",
    "training": "0",
    "B" : "0",
    "faible": "A",
    "away": "A",
    "l": "A",
    "a": "A",
    "fort": "C",
    "close": "C",
    "h": "C",
    "c": "C",
}

# Colonnes exclues communes pour les pivots
COLS_EXCLU_PASSATION = [
    "Participant",
    "Mode_transport",
    "Etat_Passation",
    "Passation",
    "Tache",
    "Sujet_num",
    "Ordre_tache",
    "Ordre_conditions",
    "Source",
]

COLS_EXCLU_SUJET = [
    "Participant",
    "Mode_transport",
    "Etat_Passation",
    "Passation",
    "Tache",
    "Ordre_tache",
    "Ordre_conditions",
    "Source",
]


# --- Utils ---
def nettoyer_colonnes(df):
    """Standardise les noms de colonnes (espaces et tirets)."""
    df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")
    return df


def lire_table(chemin: str) -> pd.DataFrame:
    """Lit un CSV (sniff sep) ou un Excel selon l'extension."""
    if chemin.lower().endswith(".csv"):
        return pd.read_csv(chemin, sep=None, engine="python", encoding="utf-8-sig")
    return pd.read_excel(chemin)


def harmoniser_identifiant(df: pd.DataFrame) -> pd.DataFrame:
    if "Participant" not in df.columns:
        if "ID" in df.columns:
            df = df.rename(columns={"ID": "Participant"})
        elif "Sujet" in df.columns:
            df = df.rename(columns={"Sujet": "Participant"})
    return df


def normaliser_passation_str(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)", expand=False)


def normaliser_etat(series: pd.Series) -> pd.Series:
    """Normalise les états de passation selon MAPPING_ETAT."""
    return series.astype(str).str.lower().map(MAPPING_ETAT).fillna(series)


def drop_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df.drop(columns=cols, errors="ignore")


def regrouper_colonnes_par_type(df: pd.DataFrame) -> pd.DataFrame:
    """Regroupe les colonnes par type d'information (EEG, Cognition, Questionnaire, Exposition)."""
    colonnes_base = []
    for col_base in ["Participant", "Mode_transport", "Etat_Passation", "Passation"]:
        if col_base in df.columns:
            colonnes_base.append(col_base)
    
    types_colonnes = {
        "EEG": [],
        "Cognition": [],
        "Questionnaire": [],
        "Exposition": [],
        "Acoustique": [],
        "Autres": []
    }
    
    for col in df.columns:
        if col in colonnes_base:
            continue
            
        col_lower = col.lower()
        
        # Classification basée sur les suffixes et préfixes
        # MODIFIÉ : Ajout des patterns pour les indicateurs ANT EEG
        if (any(suffix in col_lower for suffix in ["_1b", "_2b", "_ant"]) and 
            any(term in col_lower for term in ["amplitude", "puissance", "n1", "p3"])) or \
           any(term in col_lower for term in ["roi_ext_", "roi_moy_", "elec_ext_", "elec_moy_", 
                                               "_alerte_", "_orient", "_ctrl_exec", "moyenne_roi_",
                                               "n1_max_", "n1_moy_", "p3_max_", "p3_moy_",
                                               "_alerte_standard", "_alerte_auditive", "_alerte_visuelle",
                                               "_orientation", "_controle"]):
            types_colonnes["EEG"].append(col)
        elif any(suffix in col_lower for suffix in ["_1b", "_2b", "_ant"]) and not any(term in col_lower for term in ["amplitude", "puissance", "n1", "p3"]):
            types_colonnes["Cognition"].append(col)
        elif col_lower.startswith("leq_"):
            types_colonnes["Acoustique"].append(col)
        elif "_exp" in col_lower or any(term in col_lower for term in ["concentration", "dose", "inhal"]):
            types_colonnes["Exposition"].append(col)
        elif any(term in col_lower for term in ["q1", "q2", "questionnaire", "passation", "vie"]):
            types_colonnes["Questionnaire"].append(col)
        else:
            types_colonnes["Autres"].append(col)
    
    colonnes_ordonnees = colonnes_base.copy()
    for type_info, colonnes in types_colonnes.items():
        if colonnes:
            colonnes_ordonnees.extend(sorted(colonnes))
    
    return df[colonnes_ordonnees]


def pivot_passation(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot par passation avec vérification des colonnes requises."""
    colonnes_requises = ["Participant", "Mode_transport", "Etat_Passation", "Passation", "Tache"]
    manquantes = [c for c in colonnes_requises if c not in df.columns]
    if manquantes:
        print(f"⚠️ Colonnes manquantes pour pivot_passation : {manquantes}")
        return pd.DataFrame()
    
    valeurs = [col for col in df.columns if col not in COLS_EXCLU_PASSATION]
    dfp = df.pivot_table(
        index=["Participant", "Mode_transport", "Etat_Passation", "Passation"],
        columns="Tache",
        values=valeurs,
        aggfunc="first",
    )
    dfp.columns = [f"{val}_{tache}" if tache else f"{val}" for val, tache in dfp.columns]
    
    return dfp.reset_index()


def pivot_sujet(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot par sujet avec vérification des colonnes requises."""
    colonnes_requises = ["Participant", "Mode_transport", "Etat_Passation", "Tache"]
    manquantes = [c for c in colonnes_requises if c not in df.columns]
    if manquantes:
        print(f"⚠️ Colonnes manquantes pour pivot_sujet : {manquantes}")
        return pd.DataFrame()
    
    valeurs = [col for col in df.columns if col not in COLS_EXCLU_SUJET]
    dfs = df.pivot_table(
        index=["Participant", "Mode_transport"],
        columns=["Etat_Passation", "Tache"],
        values=valeurs,
        aggfunc="first",
    )
    dfs.columns = [
        f"{val}_{etat}{'_' + tache if tache else ''}" if tache else f"{val}_{etat}"
        for val, etat, tache in dfs.columns
    ]
    return dfs.reset_index()


def ajouter_infos(df):
    """Ajoute les informations de base et normalise Etat_Passation UNE SEULE FOIS."""
    df["Participant"] = df["Participant"].astype(str).str.extract(r"(\d+)", expand=False)
    
    try:
        df["Participant"] = df["Participant"].astype(int)
    except (ValueError, TypeError) as e:
        print(f"⚠️ Erreur conversion Participant en int : {e}")
        n_invalides = df["Participant"].isna().sum()
        if n_invalides > 0:
            print(f"   {n_invalides} participants avec IDs invalides détectés")
    
    df["Sujet_num"] = df["Participant"]
    df["Mode_transport"] = df["Participant"].astype(str).str[0].map({"1": "Bike", "2": "e-Bike"})
    
    if "Etat_Passation" in df.columns:
        df["Etat_Passation"] = normaliser_etat(df["Etat_Passation"])

    return df


# --- Traitements par catégorie ---
def traiter_cognition():
    dfs = []
    for fichier in fichiers_cognition:
        chemin = os.path.join(chemin_cognition, fichier)
        if not os.path.exists(chemin):
            print(f"[!] Fichier manquant : {chemin}")
            continue

        df = lire_table(chemin)
        if "Filtrage" in df.columns:
            df = df[df["Filtrage"] == "Apres"]
            df = df.drop(columns=["Filtrage"])

        df = nettoyer_colonnes(df)

        if "Participant" not in df.columns and "Sujet" in df.columns:
            df = df.rename(columns={"Sujet": "Participant"})
        if "Participant" in df.columns:
            df["Participant"] = df["Participant"].astype(str).str.replace("^P", "", regex=True)

        if "1B" in fichier:
            df["Tache"] = "1B"
        elif "2B" in fichier:
            df["Tache"] = "2B"
        elif "ANT" in fichier:
            df["Tache"] = "ANT"
        else:
            df["Tache"] = "Cognition"

        df = ajouter_infos(df)
        df["Source"] = "Cognition"
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else None


def traiter_eeg():
    dfs = []
    for fichier in fichiers_eeg:
        chemin = os.path.join(chemin_EEG, fichier)
        if not os.path.exists(chemin):
            print(f"[!] Fichier manquant : {chemin}")
            continue

        df = lire_table(chemin)
        df = nettoyer_colonnes(df)
        df = df.rename(columns={"Sujet": "Participant"})
        
        # Pour EEG: Condition (H/L) -> Etat_Passation (C/A) - AVANT le traitement ANT
        if "Condition" in df.columns:
            df["Etat_Passation"] = df["Condition"].astype(str).map({"H": "C", "L": "A"}).fillna(df["Condition"])
            df = df.drop(columns=["Condition"])
        elif "Etat_Passation" not in df.columns:
            df["Etat_Passation"] = "0"
            df["Passation"] = "1"
        
        if "1B" in fichier:
            df["Tache"] = "1B"
        elif "2B" in fichier:
            df["Tache"] = "2B"
        elif "ANT" in fichier:
            df["Tache"] = "ANT"
            # Les indicateurs sont déjà calculés dans ANT_indicateurs_ERP.xlsx
            # On garde toutes les colonnes N1/P3
            colonnes_indicateurs = [col for col in df.columns if col.startswith(('N1_', 'P3_'))]
            print(f"    [INFO] ANT EEG Indicateurs: {len(colonnes_indicateurs)} colonnes d'indicateurs chargées")
            if colonnes_indicateurs:
                print(f"    [INFO] Indicateurs disponibles : {', '.join(colonnes_indicateurs[:5])}...")
        else:
            df["Tache"] = "EEG"

        df = ajouter_infos(df)
        df["Source"] = "EEG"
        
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else None


def traiter_questionnaire_passation(df, fichier):
    df = harmoniser_identifiant(df)

    prefixed = [col for col in df.columns if re.match(r"^\d+_", col)]
    if not prefixed:
        print(f"[!] Aucune colonne préfixée trouvée dans {fichier}")
        return None

    df_long = df.melt(id_vars=["Participant", "Ordre_tache", "Ordre_conditions"],
                      value_vars=prefixed,
                      var_name="Var",
                      value_name="Valeur")

    df_long["Passation"] = df_long["Var"].str.extract(r"^(\d+)_")
    df_long["Variable"] = df_long["Var"].str.replace(r"^\d+_", "", regex=True)
    df_long = df_long.drop(columns=["Var"])

    def map_etat(row):
        if row["Passation"] == "1":
            return "0"
        elif row["Passation"] == "2":
            return "A" if "Faible" in str(row["Ordre_conditions"]) else "C"
        elif row["Passation"] == "3":
            return "C" if "Faible" in str(row["Ordre_conditions"]) else "A"
        else:
            return None

    df_long["Etat_Passation"] = df_long.apply(map_etat, axis=1)

    df_wide = df_long.pivot_table(
        index=["Participant", "Passation", "Etat_Passation", "Ordre_tache", "Ordre_conditions"],
        columns="Variable", values="Valeur", aggfunc="first"
    ).reset_index()

    df_wide = ajouter_infos(df_wide)
    df_wide["Tache"] = ""
    df_wide["Source"] = "Questionnaire"

    return df_wide


def traiter_questionnaire_vie(df, fichier):
    """Transforme un fichier questionnaire type *-vie (1 ligne / sujet)."""
    df = ajouter_infos(df)
    df["Tache"] = ""
    df["Source"] = "Questionnaire"
    return df


def traiter_questionnaire():
    dfs = []
    dfs_vie = []

    for fichier in fichiers_questionnaire:
        chemin = os.path.join(chemin_questionnaire, fichier)
        if not os.path.exists(chemin):
            print(f"[!] Fichier manquant : {chemin}")
            continue

        try:
            df = lire_table(chemin)
        except Exception as e:
            print(f"[X] Erreur lecture {chemin}: {e}")
            continue

        df = nettoyer_colonnes(df)
        df = harmoniser_identifiant(df)

        if "-passation" in fichier:
            df_transforme = traiter_questionnaire_passation(df, fichier)
            if df_transforme is not None:
                dfs.append(df_transforme)
        elif "-vie" in fichier:
            df_transforme = traiter_questionnaire_vie(df, fichier)
            if df_transforme is not None:
                dfs_vie.append(df_transforme)

    return (pd.concat(dfs, ignore_index=True) if dfs else None,
            pd.concat(dfs_vie, ignore_index=True) if dfs_vie else None)


def traiter_exposition():
    dfs = []
    for fichier in fichiers_exposition:
        chemin = os.path.join(chemin_exposition, fichier)
        if not os.path.exists(chemin):
            print(f"[!] Fichier manquant : {chemin}")
            continue

        try:
            df = lire_table(chemin)
        except Exception as e:
            print(f"[X] Erreur lecture {chemin}: {e}")
            continue

        df = nettoyer_colonnes(df)
        df = harmoniser_identifiant(df)
        df = df.drop(columns=[col for col in ["Season", "season"] if col in df.columns], errors="ignore")

        if "Route" in df.columns:
            df = df.rename(columns={"Route": "Etat_Passation"})
        elif "route" in df.columns:
            df = df.rename(columns={"route": "Etat_Passation"})
        else:
            print(f"[!] Colonne Route manquante dans {fichier}")
            continue

        df = ajouter_infos(df)
        df["Source"] = "Exposition"
        df["Tache"] = "EXP"

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else None


def traiter_acoustique():
    """Extrait Leq_90 / Leq_10 par (Participant, Etat_Passation) depuis Leq_parcours.csv."""
    chemin = os.path.join(chemin_exposition, fichier_acoustique)
    if not os.path.exists(chemin):
        print(f"[!] Fichier manquant : {chemin}")
        return None

    try:
        df = lire_table(chemin)
    except Exception as e:
        print(f"[X] Erreur lecture {chemin}: {e}")
        return None

    df = nettoyer_colonnes(df)

    df["Participant"] = df["parcours_name"].astype(str).str.extract(r"P(\d+)_", expand=False)
    df["Etat_Passation"] = df["parcours_type"].astype(str).str.strip().str.upper()

    df = df[["Participant", "Etat_Passation", "Leq_90", "Leq_10"]]

    df = ajouter_infos(df)
    return df[["Participant", "Mode_transport", "Etat_Passation", "Leq_90", "Leq_10"]]


# --- Ambiance (Température / Humidité) ---
# Format des dates Base.xlsx : "30/09 : 17h15", "3/10 : 17h", "2/12: 8h", " 14/05 :8h"
_RE_DATE_PASSATION = re.compile(r"^\s*(\d{1,2})\s*/\s*(\d{1,2})\s*:?\s*(\d{1,2})\s*h")
# Nom de fichier brut : Data_AA-MM-JJ_8h.xlsx / Data_AA-MM-JJ_17h.xlsx
_RE_FICHIER_AGGR = re.compile(r"Data_(\d\d)-(\d\d)-(\d\d)_(\d+)h")


def parser_moment(date_str):
    """('30/09 : 17h15') -> (mois:int, jour:int, moment:str) ; None si NaN/illisible."""
    if pd.isna(date_str):
        return None
    m = _RE_DATE_PASSATION.match(str(date_str))
    if not m:
        return None
    jour, mois, heure = int(m.group(1)), int(m.group(2)), int(m.group(3))
    moment = "matin" if heure < 12 else "aprem"
    return (mois, jour, moment)


def indexer_fichiers_aggreges():
    """Index {(mois, jour, moment) -> chemin} sur les 2 dossiers Aggregated.

    Mois disjoints entre campagnes (sept-déc 2024 vs mars-juin 2025) -> pas de
    collision (vérifié sur les 115 sessions datées).
    """
    index = {}
    for dossier in (chemin_aggr_aut, chemin_aggr_print):
        for chemin in glob.glob(os.path.join(dossier, "*.xlsx")):
            m = _RE_FICHIER_AGGR.search(os.path.basename(chemin))
            if not m:
                continue
            mois, jour, heure = int(m.group(2)), int(m.group(3)), int(m.group(4))
            moment = "matin" if heure < 12 else "aprem"
            index[(mois, jour, moment)] = chemin
    return index


def moyenne_t_rh(chemin):
    """Moyenne de T (C) et RH (%) sur tout le fichier (1 fichier = 1 trajet).

    Gère la virgule décimale française (campagne automne 2024).
    """
    df = pd.read_excel(chemin)

    def _moy(col):
        if col not in df.columns:
            return float("nan")
        serie = df[col].astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(serie, errors="coerce").mean()

    return _moy("T (C)"), _moy("RH (%)")


def traiter_ambiance():
    """Extrait T_moy / RH_moy par passation depuis les fichiers bruts capteurs.

    Apparie Base.xlsx (Date_P2/Date_P3 -> jour/mois/moment) aux fichiers
    Aggregated. Calqué sur charger_mapping_base() (lecture Base + ajouter_infos).
    """
    chemin = os.path.join(chemin_base_ref, "Base.xlsx")
    if not os.path.exists(chemin):
        print(f"[!] Fichier de référence Base.xlsx manquant : {chemin}")
        return None
    try:
        df = pd.read_excel(chemin)
    except Exception as e:
        print(f"[X] Erreur lecture Base.xlsx (ambiance) : {e}")
        return None

    df = nettoyer_colonnes(df)
    col_p2 = "Passation_2" if "Passation_2" in df.columns else ("Passation-2" if "Passation-2" in df.columns else None)
    col_p3 = "Passation_3" if "Passation_3" in df.columns else ("Passation-3" if "Passation-3" in df.columns else None)
    if col_p2 is None or col_p3 is None or "Participant" not in df.columns:
        print("[!] Colonnes attendues absentes dans Base.xlsx (ambiance)")
        return None
    if "Date_P2" not in df.columns or "Date_P3" not in df.columns:
        print("[!] Colonnes Date_P2 / Date_P3 absentes dans Base.xlsx")
        return None

    index_fichiers = indexer_fichiers_aggreges()

    lignes = []
    manquants = []
    for _, r in df.iterrows():
        for passation, col_date, col_etat in (
            ("2", "Date_P2", col_p2),
            ("3", "Date_P3", col_p3),
        ):
            cle = parser_moment(r[col_date])
            if cle is None:
                continue
            chemin_fichier = index_fichiers.get(cle)
            if chemin_fichier is None:
                manquants.append((r["Participant"], passation, r[col_date]))
                continue
            t_moy, rh_moy = moyenne_t_rh(chemin_fichier)
            lignes.append({
                "Participant": r["Participant"],
                "Passation": passation,
                "Etat_Passation": r[col_etat],
                "T_moy": t_moy,
                "RH_moy": rh_moy,
            })

    if manquants:
        print(f"⚠️ Ambiance : {len(manquants)} session(s) sans fichier brut apparié :")
        for part, passation, date_brute in manquants:
            print(f"   - {part} P{passation} (date Base : {date_brute})")

    if not lignes:
        print("[!] Aucune donnée ambiance (T/RH) extraite")
        return None

    df_amb = pd.DataFrame(lignes)
    df_amb = ajouter_infos(df_amb)
    return df_amb[["Participant", "Mode_transport", "Etat_Passation", "Passation", "T_moy", "RH_moy"]]


def charger_mapping_base():
    chemin = os.path.join(chemin_base_ref, "Base.xlsx")
    if not os.path.exists(chemin):
        print(f"[!] Fichier de référence Base.xlsx manquant : {chemin}")
        return None
    try:
        df = pd.read_excel(chemin)
    except Exception as e:
        print(f"[X] Erreur lecture Base.xlsx : {e}")
        return None

    df = nettoyer_colonnes(df)
    col_p2 = "Passation_2" if "Passation_2" in df.columns else ("Passation-2" if "Passation-2" in df.columns else None)
    col_p3 = "Passation_3" if "Passation_3" in df.columns else ("Passation-3" if "Passation-3" in df.columns else None)
    if col_p2 is None or col_p3 is None or "Participant" not in df.columns:
        print("[!] Colonnes attendues absentes dans Base.xlsx (Participant, Passation-2, Passation-3)")
        return None

    df = df[["Participant", col_p2, col_p3]].copy()
    df = ajouter_infos(df)

    lignes = []
    for _, r in df.iterrows():
        lignes.append({"Participant": r["Participant"], "Passation": "1", "Etat_Passation": "0"})
        if pd.notna(r[col_p2]):
            lignes.append({"Participant": r["Participant"], "Passation": "2", "Etat_Passation": r[col_p2]})
        if pd.notna(r[col_p3]):
            lignes.append({"Participant": r["Participant"], "Passation": "3", "Etat_Passation": r[col_p3]})
    df_map = pd.DataFrame(lignes)
    
    return df_map


# --- Assemblage global ---
def assembler_global():
    dfs_passation = []
    dfs_vie = []

    # --- Cognition ---
    df_cog = traiter_cognition()
    if df_cog is not None:
        dfs_passation.append(df_cog)
        print(f"[✓] Cognition chargée : {len(df_cog)} lignes")
    else:
        print("[!] Aucune donnée cognition")

    # --- EEG ---
    df_eeg = traiter_eeg()
    if df_eeg is not None:
        dfs_passation.append(df_eeg)
        print(f"[✓] EEG chargée : {len(df_eeg)} lignes")
    else:
        print("[!] Aucune donnée EEG")

    # --- Questionnaire ---
    df_q_pass, df_q_vie = traiter_questionnaire()
    if df_q_pass is not None:
        dfs_passation.append(df_q_pass)
        print(f"[✓] Questionnaire (passation) chargé : {len(df_q_pass)} lignes")
    else:
        print("[!] Aucune donnée questionnaire (passation)")
    
    if df_q_vie is not None:
        dfs_vie.append(df_q_vie)
        print(f"[✓] Questionnaire (vie) chargé : {len(df_q_vie)} lignes")
    else:
        print("[!] Aucune donnée questionnaire (vie)")

    # --- Exposition ---
    df_exp_source = traiter_exposition()
    if df_exp_source is not None:
        print(f"[✓] Exposition chargée : {len(df_exp_source)} lignes")
    else:
        print("[!] Aucune donnée exposition")

    # --- Ambiance (Température / Humidité) ---
    df_amb = traiter_ambiance()
    if df_amb is not None:
        print(f"[✓] Ambiance (T/RH) chargée : {len(df_amb)} lignes")
    else:
        print("[!] Aucune donnée ambiance (T/RH)")

    # --- Acoustique (Leq) ---
    df_acous = traiter_acoustique()
    if df_acous is not None:
        print(f"[✓] Acoustique (Leq) chargée : {len(df_acous)} lignes")
    else:
        print("[!] Aucune donnée acoustique (Leq)")

    if not dfs_passation:
        raise ValueError("Aucune donnée passation chargée !")

    df_all = pd.concat(dfs_passation, ignore_index=True)
    
    if "Passation" in df_all.columns:
        df_all["Passation"] = normaliser_passation_str(df_all["Passation"]) 

    mask_eeg = df_all["Source"] == "EEG"
    df_eeg = df_all[mask_eeg].copy()
    df_non_eeg = df_all[~mask_eeg].copy()
    
    df_base_map = charger_mapping_base()
    if df_base_map is not None and not df_base_map.empty and len(df_non_eeg) > 0:
        df_non_eeg = pd.merge(df_non_eeg, df_base_map.rename(columns={"Etat_Passation": "Etat_from_base"}), on=["Participant", "Passation"], how="left")
        if "Etat_from_base" in df_non_eeg.columns:
            df_non_eeg["Etat_Passation"] = df_non_eeg["Etat_from_base"].fillna(df_non_eeg["Etat_Passation"])
            df_non_eeg = df_non_eeg.drop(columns=["Etat_from_base"], errors="ignore")

        df_base_map_pe = df_base_map.drop_duplicates(subset=["Participant", "Etat_Passation"]).rename(columns={"Passation": "Passation_from_base"})
        df_non_eeg = pd.merge(
            df_non_eeg,
            df_base_map_pe[["Participant", "Etat_Passation", "Passation_from_base"]],
            on=["Participant", "Etat_Passation"],
            how="left",
        )
        if "Passation_from_base" in df_non_eeg.columns:
            df_non_eeg["Passation"] = normaliser_passation_str(df_non_eeg["Passation_from_base"].fillna(df_non_eeg["Passation"]))
            df_non_eeg = df_non_eeg.drop(columns=["Passation_from_base"], errors="ignore")
    
    df_all = pd.concat([df_eeg, df_non_eeg], ignore_index=True)

    mask_eeg_final = df_all["Source"] == "EEG"
    df_eeg_final = df_all[mask_eeg_final].copy()
    df_non_eeg_final = df_all[~mask_eeg_final].copy()
    
    df_non_eeg_final = df_non_eeg_final.drop_duplicates(subset=[col for col in ["Participant", "Mode_transport", "Etat_Passation", "Passation", "Tache"] if col in df_non_eeg_final.columns])
    
    df_all = pd.concat([df_eeg_final, df_non_eeg_final], ignore_index=True)
    
    print(f"\n[INFO] Données assemblées :")
    print(f"  Sources uniques : {df_all['Source'].unique()}")
    print(f"  États uniques : {df_all['Etat_Passation'].unique()}")
    print(f"  Tâches uniques : {df_all['Tache'].unique()}")
    print(f"  Total lignes : {len(df_all)}")

    # --- Export version passation ---
    df_passation = pivot_passation(df_all)

    if df_exp_source is not None:  
        cols_exp_exclure = {"Participant", "Mode_transport", "Etat_Passation", "Passation", "Tache", "Sujet_num", "Source"}
        valeurs_exp = [c for c in df_exp_source.columns if c not in cols_exp_exclure]

        df_exp_pivot = df_exp_source[["Participant", "Mode_transport", "Etat_Passation"] + valeurs_exp].copy()
        df_exp_pivot = df_exp_pivot.groupby(["Participant", "Mode_transport", "Etat_Passation"], as_index=False).agg("first")
        df_exp_pivot = df_exp_pivot.rename(columns={c: f"{c}_EXP" for c in valeurs_exp})

        df_passation = pd.merge(
            df_passation,
            df_exp_pivot,
            on=["Participant", "Mode_transport", "Etat_Passation"],
            how="left",
        )

    if df_amb is not None:
        df_passation = pd.merge(
            df_passation,
            df_amb[["Participant", "Mode_transport", "Etat_Passation", "T_moy", "RH_moy"]],
            on=["Participant", "Mode_transport", "Etat_Passation"],
            how="left",
        )

    if df_acous is not None:
        df_passation = pd.merge(
            df_passation,
            df_acous[["Participant", "Mode_transport", "Etat_Passation", "Leq_90", "Leq_10"]],
            on=["Participant", "Mode_transport", "Etat_Passation"],
            how="left",
        )

    df_passation = regrouper_colonnes_par_type(df_passation)

    date_actuelle = datetime.now().strftime("%Y%m%d")
    sortie_passation = os.path.join(dossier_sortie, f"Tab_passation_{date_actuelle}.xlsx")
    df_passation.to_excel(sortie_passation, index=False)

    # --- Version sujet ---
    df_all = df_all.drop(columns=["Sujet_num", "Ordre_tache", "Ordre_conditions"], errors="ignore")

    if "Tache" not in df_all.columns:
        df_all["Tache"] = ""

    df_sujet = pivot_sujet(df_all)

    if df_exp_source is not None:
        df_exp = df_exp_source.copy()
        
        df_exp = df_exp.drop(columns=[col for col in ["Passation", "Sujet_num", "Source"] if col in df_exp.columns], errors="ignore")
        colonnes_exclues_exp = [
            "Participant", "Mode_transport", "Etat_Passation", "Passation", "Tache"
        ]
        df_exp_wide = df_exp.pivot_table(
            index=["Participant", "Mode_transport"],
            columns=["Etat_Passation"],
            values=[col for col in df_exp.columns if col not in colonnes_exclues_exp],
            aggfunc="first",
        )
        df_exp_wide.columns = [f"{val}_{etat}_EXP" for val, etat in df_exp_wide.columns]
        df_exp_wide = df_exp_wide.reset_index()

        df_sujet = pd.merge(df_sujet, df_exp_wide, on=["Participant", "Mode_transport"], how="left")

    if dfs_vie:
        df_vie = pd.concat(dfs_vie, ignore_index=True)
        df_vie = df_vie.drop(columns=[col for col in ["Etat_Passation", "Categorie", "Sujet_num", "Tache", "Source"] if col in df_vie.columns])
        df_sujet = pd.merge(df_sujet, df_vie, on=["Participant", "Mode_transport"], how="left")

    # ── Sexe connu hors questionnaire ──
    # Le participant 228 (VAE, printemps 2025) n'a rempli aucun questionnaire de mode
    # de vie : toutes ses colonnes « vie » sont vides, Sexe compris. Son sexe est
    # pourtant connu — c'est un homme, information corroborée par
    # Data\Exposition\ancien\Spring2025_CombinedTable_Pollutants.xlsx (Sex = M).
    # Sans ce report, l'information se reperd à chaque régénération de la table et le
    # §3.2 du Ch. 3 (28 femmes, 23 hommes sur les 51) cesse d'être reproductible.
    # Les autres colonnes « vie » de P228 restent volontairement vides : lui seul est
    # documenté ailleurs. Reporté aussi à la main dans Tab_sujet_20260205.xlsx, la
    # table que lisent les scripts d'analyse du manuscrit.
    SEXE_HORS_QUESTIONNAIRE = {228: "Homme"}
    if "Sexe" in df_sujet.columns:
        for participant, sexe in SEXE_HORS_QUESTIONNAIRE.items():
            masque = (df_sujet["Participant"] == participant) & df_sujet["Sexe"].isna()
            if masque.any():
                df_sujet.loc[masque, "Sexe"] = sexe
                print(f"[INFO] Sexe renseigné hors questionnaire : P{participant} = {sexe}")

    if df_amb is not None:
        df_amb_wide = df_amb.pivot_table(
            index=["Participant", "Mode_transport"],
            columns=["Etat_Passation"],
            values=["T_moy", "RH_moy"],
            aggfunc="first",
        )
        df_amb_wide.columns = [f"{val}_{etat}" for val, etat in df_amb_wide.columns]
        df_amb_wide = df_amb_wide.reset_index()
        df_sujet = pd.merge(df_sujet, df_amb_wide, on=["Participant", "Mode_transport"], how="left")

    if df_acous is not None:
        df_acous_wide = df_acous.pivot_table(
            index=["Participant", "Mode_transport"],
            columns=["Etat_Passation"],
            values=["Leq_90", "Leq_10"],
            aggfunc="first",
        )
        df_acous_wide.columns = [f"{val}_{etat}" for val, etat in df_acous_wide.columns]
        df_acous_wide = df_acous_wide.reset_index()
        df_sujet = pd.merge(df_sujet, df_acous_wide, on=["Participant", "Mode_transport"], how="left")

    df_sujet = regrouper_colonnes_par_type(df_sujet)

    sortie_sujet = os.path.join(dossier_sortie, f"Tab_sujet_{date_actuelle}.xlsx")
    df_sujet.to_excel(sortie_sujet, index=False)

    print(f"\n[✓] Export effectué :")
    print(f"   - {sortie_passation}")
    print(f"   - {sortie_sujet}")
    
    print(f"\n[INFO] Vérifications finales :")
    print(f"  Tab_passation : {len(df_passation)} lignes, {len(df_passation.columns)} colonnes")
    print(f"  Tab_sujet : {len(df_sujet)} lignes, {len(df_sujet.columns)} colonnes")
    print(f"  Participants dans Tab_passation : {df_passation['Participant'].nunique()}")
    print(f"  Participants dans Tab_sujet : {len(df_sujet)}")


if __name__ == "__main__":
    assembler_global()