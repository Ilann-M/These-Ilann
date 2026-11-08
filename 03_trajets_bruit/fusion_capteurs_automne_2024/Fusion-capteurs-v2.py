# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:53:14 2024

@author: mahou
"""

import os as _os, sys as _sys  # chemins du depot (voir chemins.py a la racine)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from chemins import D  # noqa: E402
import pandas as pd
from imports import import_files_df
from generate_map import gen_map
from files_management import count_errors, get_aerocet_filenames, get_discmini_filenames, get_garmin_filenames, get_ma200_filenames, get_magnasci_filenames, someFilesAreValid
from format_Aero_Magna import format_aerocet, format_magna
from datetime import datetime, timedelta
import os
os.chdir(D(r'Campagne_aut_24\Trajets'))  # dossier des donnees capteurs

def list_dates(start, end):
    # Generate all dates between start and end
    current_date = start
    date_list = []
    
    while current_date <= end:
        year = current_date.year % 100  # Get last two digits of the year
        month = current_date.strftime('%m')  # Get month as two digits
        day = current_date.strftime('%d')  # Get day as two digits
        date_list.append(f"{year:02}-{month}-{day}")
        current_date += timedelta(days=1)
    return date_list

def process_serie(filenames, date, time):
    #check existance of each file
    if not someFilesAreValid(filenames):
        return False
    
    print(f'--------------- {date}_{time} ---------------')
    all_files_df = import_files_df(filenames)
    final_df = pd.DataFrame(merge_the_shit_together(all_files_df))
    printable_df = pd.DataFrame(final_df)
    # Convert "time" from DateTime to string
    printable_df['time'] = printable_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    printable_df.to_excel(f'Aggregated/Data_{date}_{time}.xlsx', index=False)
    try:
        gen_map(final_df, date, time)
    except Exception as e:
        print(f"Impossible de générer la carte pour le {date} à {time}, {e}")
    print('-------------------- OK --------------------\n\n')


def merge_the_shit_together(all_files_df):
    tolerance = pd.Timedelta(hours=3)
    result_merge = pd.DataFrame(all_files_df[0]).sort_values('time')
    for df in all_files_df[1:]:
        if df is None:
            continue
        result_merge = pd.merge_asof(result_merge, pd.DataFrame(df).sort_values('time'), on='time', direction='nearest',
                                     tolerance=tolerance)

    merged_df = result_merge.fillna(0)

    # Définir la liste complète des colonnes à récupérer
    all_columns = ['time', 'latitude', 'longitude', 'AT (C)', 'RH (%)', 'heart', 'PM1 (ug/m3)', 'PM2.5 (ug/m3)',
                   'PM4 (ug/m3)', 'PM7 (ug/m3)', 'PM10 (ug/m3)', 'TSP (ug/m3)', '0.3 (/L)', '0.5 (/L)', '1.0 (/L)',
                   '2.5 (/L)', '4.0 (/L)', '5.0 (/L)', '7.0 (/L)', '10 (/L)', 'IR BCc Post', 'UV BCc Post',
                   'Blue BCc Post', 'Red BCc Post', 'Green BCc Post', 'Fossil Fuel BCc', 'Biomass BCc', 'noise', 'voc',
                   'Number', 'Size', 'LDSA']

    # Ne garder que les colonnes présentes dans le DataFrame
    columns_to_get = [col for col in all_columns if col in merged_df.columns]

    final_df = merged_df[columns_to_get]

    # Renommer uniquement les colonnes présentes pour éviter les erreurs
    rename_columns = {
        'time': 'time', 'latitude': 'Latitude', 'longitude': 'Longitude', 'AT (C)': 'T (C)', 'RH (%)': 'RH (%)',
        'heart': 'Heart Rate', 'PM1 (ug/m3)': 'PM1 (ug/m3)', 'PM2.5 (ug/m3)': 'PM2.5 (ug/m3)','PM4 (ug/m3)': 'PM4 (ug/m3)',
        'PM7 (ug/m3)': 'PM7 (ug/m3)', 'PM10 (ug/m3)': 'PM10 (ug/m3)', 'TSP (ug/m3)': 'TSP (ug/m3)','0.3 (/L)': '0.3 (/L)',
        '0.5 (/L)': '0.5 (/L)', '1.0 (/L)': '1.0 (/L)', '2.5 (/L)': '2.5 (/L)', '4.0 (/L)': '4.0 (/L)',
        '5.0 (/L)': '5.0 (/L)', '7.0 (/L)': '7.0 (/L)', '10 (/L)': '10 (/L)', 'IR BCc Post': 'BC880 (ug/m3)',
        'UV BCc Post': 'BC375 (ug/m3)', 'Blue BCc Post': 'Blue BCc (ug/m3)', 'Red BCc Post': 'Red BCc (ug/m3)',
        'Green BCc Post': 'Green BCc (ug/m3)', 'Fossil Fuel BCc': 'Fossil Fuel', 'Biomass BCc': 'Biomass',
        'noise': 'noise', 'voc': 'voc', 'Number': 'Number', 'Size': 'Size', 'LDSA': 'LDSA'
    }

    # Appliquer le renommage en fonction des colonnes disponibles
    final_df = final_df.rename(columns={key: rename_columns[key] for key in rename_columns if key in final_df.columns})

    return final_df


#############################
##      DEBUT DU CODE      ##
#############################

# Lire la dernière date depuis le fichier Excel
'''
fichier_date = 'Date_traitement.xlsx'
date_df = pd.read_excel(fichier_date, usecols = ["Date_traitement"])
date_df['Date_traitement'] = pd.to_datetime(date_df['Date_traitement'])
last_date = date_df["Date_traitement"].iloc[-1].date()
'''
last_date = datetime.strptime('02-12-2024','%d-%m-%Y').date() # 29-09-2024
aujourdhui = datetime.today().date()
print(aujourdhui, last_date)

# Get the list of dates
dates = list_dates(last_date, aujourdhui)

## Initialiser df_fichier_date avec le DataFrame existant
#df_fichier_date = date_df.copy()

# on format les nouveaux fichiers
format_aerocet() 
format_magna()

for date in dates:
    # Pour chaque date on génère les noms de fichiers
    ma8h, ma17h = get_ma200_filenames(date)
    garmin8h, garmin17h = get_garmin_filenames(date)
    magnasci8h, magnasci17h = get_magnasci_filenames(date)
    discmini8h, discmini17h = get_discmini_filenames(date)
    aerocet8h, aerocet17h = get_aerocet_filenames(date)
    process_serie([ma8h, garmin8h, magnasci8h, discmini8h, aerocet8h], date, '8h')
    process_serie([ma17h, garmin17h, magnasci17h, discmini17h, aerocet17h], date, '17h')

    ## Ajouter la nouvelle date au DataFrame
    #nouvelle_ligne = pd.DataFrame({'Date_traitement': [date]})
    #df_fichier_date = pd.concat([df_fichier_date, nouvelle_ligne], ignore_index=True)

# Sauvegarder le DataFrame mis à jour dans le fichier Excel
#df_fichier_date.to_excel(fichier_date, index=False)

# Print colored text
COLOR_RED = '\033[31m'
COLOR_GREEN = '\033[32m'
COLOR_RESET = '\033[0m'  # Reset to default color

count = count_errors()
if count > 0:
    print(f"{COLOR_RED}============== {count} missing files =============={COLOR_RESET}")
else:
    print(f"{COLOR_GREEN}============== SUCCESS ! =============={COLOR_RESET}")

