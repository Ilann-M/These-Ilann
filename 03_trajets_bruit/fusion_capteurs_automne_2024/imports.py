import re
import pandas as pd
from datetime import datetime

# Initialisation des DataFrames
ma200_df = pd.DataFrame()
garmin_df = pd.DataFrame()
magnasci_df = pd.DataFrame()
discmini_df = pd.DataFrame()
aerocet_df = pd.DataFrame()

def import_files_df(filenames):
    ma200_df = try_import(import_ma200_df, filenames[0])
    garmin_df = try_import(import_garmin_df, filenames[1])
    magnasci_df = try_import(import_magnasci_df, filenames[2])
    discmini_df = try_import(import_discmini_df, filenames[3])
    aerocet_df = try_import(import_aerocet_df, filenames[4])
    return [ma200_df, garmin_df, magnasci_df, discmini_df, aerocet_df]

def try_import(method, filename):
    try:
        return method(filename)
    except Exception as e:
        print("error while importing", filename, e)
        return None
    
def import_aerocet_df(filename):
    aerocet_df = pd.read_csv(filename)
    aerocet_df['time'] = pd.to_datetime(aerocet_df['Time'], format='%Y-%m-%d %H:%M:%S')
    # aerocet_df = aerocet_df.sort_values('time')
    Aero_colones = ['time', 'PM1 (ug/m3)', 'PM2.5 (ug/m3)', 'PM4 (ug/m3)', 'PM7 (ug/m3)', 'PM10 (ug/m3)', 'TSP (ug/m3)', 'AT (C)', 'RH (%)',
                    '0.3 (/L)','0.5 (/L)','1.0 (/L)','2.5 (/L)','4.0 (/L)','5.0 (/L)','7.0 (/L)','10 (/L)']

    # Delete les lignes qui ne possedent pas de valeurs d'Aerocet afin de ne prendre que les lignes utiles
    aerocet_df = aerocet_df.dropna(subset=['PM1 (ug/m3)'])
    return aerocet_df[Aero_colones]

def import_magnasci_df(filename):
    magnasci_df = pd.read_csv(filename)
    magnasci_df['time'] = pd.to_datetime(magnasci_df['time'], format='%Y-%m-%d %H:%M:%S')
    return magnasci_df[['time', 'noise', 'voc']]

def import_ma200_df(filename):
    ma200_df = pd.read_csv(filename)
    ma200_df['time'] = ma200_df['Date Time']
    ma200_df["time"] = pd.to_datetime(ma200_df['time'], format='%Y-%m-%dT%H:%M:%S')
    ma_echelle = ['IR BCc Post', 'UV BCc Post', 'Blue BCc Post', 'Red BCc Post', 'Green BCc Post', 'Fossil Fuel BCc', 'Biomass BCc']

    date_1 = datetime(2024, 10, 28).date()
    date_2 = datetime(2024, 10, 29).date()
    mask = (ma200_df['time'].dt.date >= date_1) & (ma200_df['time'].dt.date <= date_2)
    ma200_df.loc[mask, 'time'] += pd.DateOffset(hours=1)

    for j in ma_echelle:
        ma200_df[j] = ma200_df[j].astype(float)
        ma200_df[j] = ma200_df[j].fillna(0)
        ma200_df[j] = ma200_df[j].apply(lambda x: max(0, x))
        ma200_df[j] = ma200_df[j] / 1000
    return ma200_df[['time', 'IR BCc Post', 'UV BCc Post', 'Blue BCc Post', 'Red BCc Post', 'Green BCc Post', 'Fossil Fuel BCc', 'Biomass BCc']]

pattern_date = r'\d{4}\.\d{2}\.\d{2}'
pattern_time = r'\d{2}:\d{2}:\d{2}'

def import_discmini_df(filename):
    disc_date = pd.read_csv(filename, sep='\t', nrows=2, skiprows=2, header=None)
    line1 = disc_date.iloc[0].to_string()
    line2 = disc_date.iloc[1].to_string()
    match_date = re.search(pattern_date, line1)
    match_time = re.search(pattern_time, line2)
    if match_date and match_time:
        full_date_s = match_date.group() + "T" + match_time.group()
        full_date = datetime.strptime(full_date_s, '%Y.%m.%dT%H:%M:%S')
    else:
        raise ValueError("Could not find date or time from file header", filename)
    Disc_df = pd.read_csv(filename, skiprows=6, sep='\t', names=['time', 'Number', 'Size', 'LDSA', 'Filter', 'Diff'])
    Disc_df["time"] = pd.Timestamp(full_date) + pd.to_timedelta(Disc_df["time"].str.replace(',', '.').astype(float), unit='s')

    date_pb = datetime(2024, 10, 28).date()
    mask = Disc_df['time'].dt.date >= date_pb
    Disc_df.loc[mask, 'time'] -= pd.DateOffset(hours=1)

    Disc_echelle = ['Number', 'Size', 'LDSA']
    for col in Disc_echelle:
        Disc_df[col] = Disc_df[col].astype(str).str.replace(',', '.').astype(float).fillna(0)

    return Disc_df[['time', 'Number', 'Size', 'LDSA']]

def import_garmin_df(filename):
    garmin_df = pd.read_csv(filename, sep=',', usecols=['time', 'latitude', 'longitude', 'heart'],index_col=False)
    
    # Conversion de la colonne 'time' en format datetime puis ajout des deux heures de décalages et trie
    try:
        garmin_df['time'] = pd.to_datetime(garmin_df['time'], format='%Y-%m-%dT%H:%M:%S')
    except :
        garmin_df['time'] = pd.to_datetime(garmin_df['time'], format='%d/%m/%YT%H:%M:%S')
    
    # changement d'heure
    date_pb = datetime(2024, 10, 28).date()
    mask = garmin_df['time'].dt.date >= date_pb
    garmin_df.loc[mask, 'time'] += pd.DateOffset(hours=1)
    garmin_df.loc[~mask, 'time'] += pd.DateOffset(hours=2)

    garmin_df = garmin_df.sort_values('time')
    
    # Filtrer les valeurs de latitude et longitude invalides
    garmin_df = garmin_df[garmin_df['latitude'] != -1]
    return garmin_df
