import os
Dossier = 'Capteurs'

def get_ma200_filenames(date):
    return os.path.join(Dossier, 'MA200', f"MA200_{date}_8h_post-processed.csv"), os.path.join(Dossier, 'MA200',f"MA200_{date}_17h_post-processed.csv")

def get_garmin_filenames(date):
    return os.path.join(Dossier, r'Garmin\Garmin-traité', f"Garmin_{date}_8h.csv"), os.path.join(Dossier, r'Garmin\Garmin-traité',f"Garmin_{date}_17h.csv")

def get_magnasci_filenames(date):
    return os.path.join(Dossier, r'Magnasci\Traité',f"Magnasci_{date}_8h.csv"), os.path.join(Dossier, r'Magnasci\Traité',f"Magnasci_{date}_17h.csv")

def get_discmini_filenames(date):
    return os.path.join(Dossier, r'DiSCmini\Traité-10s',f"{date}_8h_output.txt"), os.path.join(Dossier, r'DiSCmini\Traité-10s',f"{date}_17h_output.txt")

def get_aerocet_filenames(date):
    return os.path.join(Dossier, r'Aerocet\Traité',f"Aerocet_{date}_8h.csv"), os.path.join(Dossier, r'Aerocet\Traité',f"Aerocet_{date}_17h.csv")

def tryFile(filename):
    return os.path.isfile(filename)

def someFilesAreValid(list_filenames):
    list_valid = list(map(lambda filename: tryFile(filename) , list_filenames))
    if (False not in list_valid): # All files exist
        return True
    if (True not in list_valid): # No file(s)
        return False
    # Some files are missing but some exist
    for index, valid in enumerate(list_valid):
        if not valid:
            # log missing files
            write_error(f"Filename [{list_filenames[index]}] is missing")
    return True

count=0
with open('error-logs.txt', 'w') as file:
    file.write("\n")
def write_error(message):
    global count
    count += 1
    with open('error-logs.txt', 'a') as file:
        file.write(f"{message}\n")

def count_errors():
    global count
    return count
