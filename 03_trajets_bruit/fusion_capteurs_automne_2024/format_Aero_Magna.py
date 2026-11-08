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
    # Essayer de lire avec le séparateur ';'
    try:
        df = pd.read_csv(fichier_csv, sep=';', on_bad_lines='skip')
        if 'Time' in df.columns:
            return df
    except Exception as e:
        print(f"Erreur lors de la lecture avec ';': {e}")

    # Essayer de lire avec le séparateur ','
    try:
        df = pd.read_csv(fichier_csv, sep=',', on_bad_lines='skip')
        if 'Time' in df.columns:
            return df
    except Exception as e:
        print(f"Erreur lors de la lecture avec ',': {e}")

    # Essayer de lire avec le séparateur '\t' (tabulation)
    try:
        df = pd.read_csv(fichier_csv, sep='\t', on_bad_lines='skip')
        if 'Time' in df.columns:
            return df
    except Exception as e:
        print(f"Erreur lors de la lecture avec '\\t': {e}")

    # Si aucun des trois n'a fonctionné
    print(f"Impossible de lire {fichier_csv} avec les trois séparateurs.")
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

                            # date problématique
                            date_pb = datetime(2024, 10, 28).date()
                            # Vérifier si la date est égale à date_pb et l'heure est inférieure à 12h
                            condition = (df['Time'].dt.date == date_pb) & (df['Time'].dt.hour < 12)
                            # Appliquer l'ajustement d'une heure pour les lignes qui satisfont la condition
                            df.loc[condition, 'Time'] = df['Time'] - pd.DateOffset(hours=1)

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


def format_magna():
    dossier_magna = r'Capteurs/Magnasci'
    dossier_magna_b = os.path.join(dossier_magna, 'Brut')

    for magna_brut in os.listdir(dossier_magna_b):
        # Créer le chemin complet du fichier
        magna_brut_path = os.path.join(dossier_magna_b, magna_brut)  # Utiliser le chemin complet

        try:
            # Lire le fichier CSV avec le chemin complet
            magnasci_df = pd.read_csv(magna_brut_path)  
            magnasci_df['time'] = pd.to_datetime(magnasci_df['time'], unit='s')

            dfs_divises = diviser_dataframe_par_intervalles(magnasci_df, T='time')

            for df_segment in dfs_divises:
                if not df_segment.empty:
                    # date de changement d'heure
                    date_pb = datetime(2024, 10, 28).date()
                    mask = df_segment['time'].dt.date >= date_pb

                    df_segment.loc[mask, 'time'] += pd.DateOffset(hours=1)
                    df_segment.loc[~mask, 'time'] += pd.DateOffset(hours=2)

                    heure = int(df_segment['time'].min().strftime('%H'))
                    date_heure = df_segment['time'].min().strftime('%y-%m-%d')
                    output_filename = f'Magnasci_{date_heure}_{heure}h.csv'
                    output_path = os.path.join(dossier_magna, 'Traité', output_filename)

                    # Sauvegarder le segment dans un fichier CSV
                    df_segment.to_csv(output_path, index=False)
                    print(f'Données segmentées sauvegardées dans {output_path}')

        except FileNotFoundError:
            print(f"Le fichier {magna_brut_path} est introuvable.")
        except pd.errors.EmptyDataError:
            print(f"Le fichier {magna_brut_path} est vide ou illisible.")
