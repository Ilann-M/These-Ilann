"""
Chemins locaux du dépôt.

Les scripts n'utilisent aucun chemin absolu : ils passent par les fonctions ci-dessous.

    D(r'Data\\Table_combine\\Tab_sujet_20260205.xlsx')  -> fichier sous la racine des données
    S(r'Resultats_LMM_intention\\...')                  -> dossier de sortie

Réglage, au choix :
  - variables d'environnement THESE_RACINE (racine des données) et, facultativement,
    THESE_SORTIES (racine des sorties, par défaut <THESE_RACINE>/Data/Traitement) ;
  - ou modification des valeurs par défaut ci-dessous.

Les données ne sont pas publiques (voir README.md). L'arborescence attendue sous la racine
est celle décrite dans le README.

L'import de ce module ajoute aussi 07_modeles/ au chemin de recherche Python, car plusieurs
scripts de robustesse et de description réutilisent les fonctions des scripts de modèles.
"""
import os
import re
import sys

DEPOT = os.path.dirname(os.path.abspath(__file__))
RACINE = os.environ.get('THESE_RACINE', os.path.join('CHEMIN', 'VERS', 'DONNEES'))
SORTIES = os.environ.get('THESE_SORTIES', os.path.join(RACINE, 'Data', 'Traitement'))


def _joindre(base, relatif):
    morceaux = [m for m in re.split(r'[\\/]+', relatif) if m]
    return os.path.join(base, *morceaux)


def D(relatif=''):
    """Chemin sous la racine des données."""
    return _joindre(RACINE, relatif)


def S(relatif=''):
    """Chemin sous la racine des sorties."""
    return _joindre(SORTIES, relatif)


_modeles = os.path.join(DEPOT, '07_modeles')
if _modeles not in sys.path:
    sys.path.insert(1, _modeles)
