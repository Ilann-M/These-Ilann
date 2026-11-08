# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:26:00 2024

@author: ilann
"""
import os
import pandas as pd
import glob
from datetime import datetime

# Fonction pour vérifier si le nom du dossier correspond à un mois
def est_un_mois(nom_dossier):
    return nom_dossier.isdigit() and 1 <= int(nom_dossier) <= 12

# Fonction pour lire un fichier CSV avec deux séparateurs potentiels
def lire_csv_avec_separateurs(fichier_csv):
    # Liste des séparateurs possibles
    separateurs = [';', ',', '\t']

    # Liste des formats de dates possibles dans les fichiers
    formats_date = ['%d/%m/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%d/%m/%y %H:%M:%S']

    for sep in separateurs:
        try:
            df = pd.read_csv(fichier_csv, sep=sep, on_bad_lines='skip')

            if 'Time' in df.columns:
                df.columns = df.columns.str.strip()  # Nettoie les noms de colonnes

                # Essaye chaque format de date explicitement
                for fmt in formats_date:
                    try:
                        df['Time'] = pd.to_datetime(df['Time'], format=fmt)
                        break
                    except Exception:
                        continue
                else:
                    # Si aucun format ne marche, utilise parsing auto (avec dayfirst=True en dernier recours)
                    df['Time'] = pd.to_datetime(df['Time'], dayfirst=True, errors='coerce')

                return df
        except Exception as e:
            print(f"[!] Erreur avec le séparateur '{sep}' : {e}")

    print(f"[!] Échec de lecture du fichier : {fichier_csv}")
    return None

# Fonction pour diviser le DataFrame en sous-ensembles basés sur les intervalles de temps
def diviser_dataframe_par_intervalles(donnees, T='Time'):
    donnees[T] = pd.to_datetime(donnees[T], errors='coerce')
    donnees = donnees.sort_values(by=T)
    
    # Calculer la différence de temps
    donnees['diff'] = donnees[T].diff().dt.total_seconds().fillna(0) / 3600  # En heures
    
    # Identifier les segments où la différence de temps est supérieure à l'intervalle spécifié
    segments = (donnees['diff'] > 2).cumsum()
    
    # Diviser le DataFrame en groupes basés sur les segments
    donnees_divisees = donnees.groupby(segments)
    
    dfs_divises = []
    
    for _, df_segment in donnees_divisees:
        dfs_divises.append(df_segment.drop(columns=['diff']))
    return dfs_divises

def format_aerocet():
    # Dossier principal contenant les sous-dossiers par mois
    dossier_aero = 'Capteurs/Aerocet'

    # Parcourir les dossiers dans 'Capteurs/Aerocet'
    for mois in os.listdir(dossier_aero):
        print('Traitement du mois:', mois)
        mois_path = os.path.join(dossier_aero, mois)
        
        if est_un_mois(mois):
            # Regarder les sous-dossiers dont les 4 premiers caractères correspondent
            sous_dossiers = [d for d in os.listdir(mois_path) if os.path.isdir(os.path.join(mois_path, d))]
            # Dictionnaire pour stocker les DataFrames fusionnés par les 4 premiers caractères du dossier
            donnees_fusionnees = {}

            for sous_dossier in sous_dossiers:
                # Extraire les 5 premiers caractères (ex: '02 11', '03 17')
                prefixe = sous_dossier[:5]
                # Chemin du sous-dossier
                sous_dossier_path = os.path.join(mois_path, sous_dossier)
                fichiers_csv = glob.glob(os.path.join(sous_dossier_path, 'Data.csv'))
                
                for fichier_csv in fichiers_csv:
                    df = lire_csv_avec_separateurs(fichier_csv)

                    if df is not None:
                        # Supprimer les colonnes sans nom
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

                        # Afficher les noms de colonnes après suppression des espaces
                        df.columns = df.columns.str.strip()  # Supprime les espaces autour des noms de colonnes

                        # Vérifier si 'Time' est dans les colonnes
                        if 'Time' in df.columns:
                            # Convertir la colonne 'Time' en datetime
                            df['Time'] = pd.to_datetime(df['Time'], dayfirst=True, errors='coerce')

                            # Fusionner les données par préfixe en se basant sur la colonne 'Time'
                            if prefixe in donnees_fusionnees:
                                # Effectuer un merge avec le DataFrame existant sur la colonne 'Time'
                                donnees_fusionnees[prefixe] = pd.merge(donnees_fusionnees[prefixe], df, on='Time', how='outer')
                            else:
                                df = df.drop(columns = ['AT (C)', 'RH (%)'])
                                donnees_fusionnees[prefixe] = df
                        else:
                            print(f"Erreur : La colonne 'Time' est absente dans {fichier_csv}")
                    
            # Afficher ou sauvegarder les données fusionnées
            for prefixe, donnees in donnees_fusionnees.items():
                # Diviser les données en fonction des intervalles de temps
                dfs_divises = diviser_dataframe_par_intervalles(donnees)

                for df_segment in dfs_divises:
                    # Créer le nom de fichier de sortie basé sur la date et l'heure de la colonne 'Time'
                    if not df_segment.empty:
                        heure = int(df_segment['Time'].min().strftime('%H'))
                        date_heure = df_segment['Time'].min().strftime('%y-%m-%d') 
                        output_filename = f'Aerocet_{date_heure}_{heure}h.csv'
                        output_path = os.path.join(dossier_aero, 'Traité', output_filename)

                        # Sauvegarde des données segmentées dans un nouveau fichier CSV
                        df_segment.to_csv(output_path, index=False)
                        print(f'Données segmentées sauvegardées dans {output_path}')
