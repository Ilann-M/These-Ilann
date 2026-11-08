function C = chemins()
% CHEMINS  Chemins locaux utilisés par les scripts MATLAB du dépôt.
%
%   C.racine : racine des données (variable d'environnement THESE_RACINE)
%   C.eeglab : dossier d'installation d'EEGLAB (variable d'environnement EEGLAB_DIR)
%
%   Renseigner les variables d'environnement, par exemple :
%       setenv('THESE_RACINE', 'D:\mes_donnees');
%       setenv('EEGLAB_DIR', 'C:\outils\eeglab2026.0.0');
%   ou modifier les valeurs par défaut ci-dessous.
%   Si le dossier EEGLAB existe, il est ajouté au chemin MATLAB.

C.racine = getenv('THESE_RACINE');
if isempty(C.racine)
    C.racine = fullfile('CHEMIN', 'VERS', 'DONNEES');
end

C.eeglab = getenv('EEGLAB_DIR');
if isempty(C.eeglab)
    C.eeglab = fullfile('CHEMIN', 'VERS', 'eeglab');
end
if exist(C.eeglab, 'dir')
    addpath(C.eeglab);
end
end
