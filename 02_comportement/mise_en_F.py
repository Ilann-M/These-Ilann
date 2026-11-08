
import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from chemins import D, S, DEPOT  # noqa: E402
import sys
import pandas as pd
import os

def fusionner_fichiers_par_tache(dossier, noms_fichiers):
    """
    Fusionne les fichiers de résultats (CSV ou XLSX) par tâche.

    - Passation 1 est CONSERVÉE (baseline).
    - Etat_Passation utilisé comme préfixe des colonnes.
    - Pour la tâche ANT (xlsx) : conserve uniquement Filtrage == "Apres".
    - Renommage des états : 0 → B (Baseline), Faible → A, Fort → C.
    - Sortie : un fichier CSV par tâche, avec un participant par ligne.
    """
    resultats = {}

    # Regrouper fichiers par tâche (1B, 2B, ANT, ...)
    fichiers_par_tache = {}
    for fichier in noms_fichiers:
        nom_tache = fichier.split("_")[1]  # ex: "1B" dans "Resultats_1B_camp1.csv"
        fichiers_par_tache.setdefault(nom_tache, []).append(fichier)

    # Dictionnaire de renommage des états
    mapping_etats = {
        "0": "B",        # Baseline (Passation 1)
        "Faible": "A",   # Condition Faible
        "Fort": "C"      # Condition Fort
    }

    for tache, fichiers in fichiers_par_tache.items():
        all_data = []

        for fichier in fichiers:
            chemin = os.path.join(dossier, fichier)
            if not os.path.isfile(chemin):
                print(f"⛔ Fichier non trouvé : {fichier}")
                continue

            print(f"🔄 Lecture : {fichier}")

            # Lire selon l'extension
            if fichier.lower().endswith(".xlsx"):
                df = pd.read_excel(chemin)
            else:
                df = pd.read_csv(chemin, sep=",")

            # Cas ANT : garder uniquement Filtrage == "Apres"
            if "Filtrage" in df.columns:
                df = df[df["Filtrage"] == "Apres"]

            # MODIFICATION : On garde toutes les passations (y compris Passation 1)
            # La ligne suivante est SUPPRIMÉE :
            # if "Passation" in df.columns:
            #     df = df[df["Passation"] != 1]

            # S'assurer que Etat_Passation est en string pour le mapping
            if "Etat_Passation" in df.columns:
                df["Etat_Passation"] = df["Etat_Passation"].astype(str)

            if not df.empty:
                all_data.append(df)

        if not all_data:
            continue

        # Concaténer tous les fichiers de la tâche
        df_all = pd.concat(all_data, ignore_index=True)

        # Colonnes fixes possibles
        colonnes_fixes = ["Participant", "Passation", "Etat_Passation", "Filtrage"]
        colonnes_mesures = [c for c in df_all.columns if c not in colonnes_fixes]

        # Pivot large
        df_pivot = df_all.pivot(
            index="Participant",
            columns="Etat_Passation",
            values=colonnes_mesures
        )

        # Aplatir multi-index colonnes + appliquer mapping
        df_pivot.columns = [
            f"{mapping_etats.get(str(etat), str(etat))}_{col}"  # remplacement si trouvé
            for col, etat in df_pivot.columns
        ]
        df_final = df_pivot.reset_index()

        # Réorganiser les colonnes : Participant, puis B_, A_, C_ dans l'ordre
        colonnes_ordonnees = ["Participant"]
        prefixes_ordre = ["B", "A", "C"]
        
        for prefixe in prefixes_ordre:
            cols_prefixe = sorted([c for c in df_final.columns if c.startswith(f"{prefixe}_")])
            colonnes_ordonnees.extend(cols_prefixe)
        
        # Ajouter les colonnes qui ne correspondent à aucun préfixe (si existantes)
        colonnes_restantes = [c for c in df_final.columns if c not in colonnes_ordonnees]
        colonnes_ordonnees.extend(colonnes_restantes)
        
        df_final = df_final[colonnes_ordonnees]

        # Sauvegarde
        nom_sortie = f"{tache}_fusion.csv"
        chemin_sortie = os.path.join(dossier, nom_sortie)
        df_final.to_csv(chemin_sortie, index=False, encoding="utf-8")

        print(f"✅ Fichier exporté : {chemin_sortie}")
        print(f"👥 Nombre de participants pour {tache} : {len(df_final)}")
        print(f"📊 Colonnes créées : {', '.join([c for c in df_final.columns if c != 'Participant'])}\n")

        resultats[tache] = df_final

    return resultats


# Exemple d'utilisation
if __name__ == "__main__":
    # Préparation des données
    # Dossier de travail : Data\Cognition\Tests-cog\Resultats sous la racine
    # des donnees (chemins.py), ou chemin passe en argument.
    chemin_dossier = (sys.argv[1] if len(sys.argv) > 1
                      else D(r'Data\Cognition\Tests-cog\Resultats'))
    print(f"Dossier de travail : {chemin_dossier}")
    fichiers = [
        "Resultats_1B_camp1.csv",
        "Resultats_1B_camp2.csv",
        "Resultats_2B_camp1.csv",
        "Resultats_2B_camp2.csv",
        "Resultats_ANT_camp1.xlsx",
        "Resultats_ANT_camp2.xlsx"
    ]
    
    dfs_par_tache = fusionner_fichiers_par_tache(chemin_dossier, fichiers)
    
    # Affichage des premières lignes de chaque tâche
    for tache, df in dfs_par_tache.items():
        print(f"\n{'='*80}")
        print(f"Aperçu de {tache}_fusion.csv ({len(df)} participants)")
        print(f"{'='*80}")
        print(df.head(3))