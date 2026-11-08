%% EXTRACTION DES INDICATEURS ERP ANT - VERSION 4 (variante locale)
%  Copie de extract_erp_ANT_roi_v4.m. Seuls changent : le chemin EEGLAB, le
%  demarrage sans interface, et les chemins de donnees, qui pointent le
%  disque local au lieu du partage reseau. Le calcul est inchange.
% Ce script calcule les indicateurs ANT à partir de 3 types de triggers :
%   - Triggers flèches : N1, P3 (5 indicateurs)
%   - Triggers sonores : N1, P2 (2 indicateurs)
%   - Triggers indice : N1 (1 indicateur)
%
% Indicateurs calculés :
%   - Alerte standard : sans - avec (indice sonore)
%   - Alerte auditive : sans - avec (indice sonore, uniquement essais sans indice spatial)
%   - Alerte visuelle : sans_indice - moyenne(indicé)
%   - Orientation : invalide - valide
%   - Contrôle exécutif : incongruent - congruent
%
% Version 4 : Intégration des 3 types de triggers avec ROIs spécifiques

clear; clc; close all;

%% === INITIALISATION EEGLAB ===
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));  % chemins.m, a la racine du depot
C = chemins();
addpath(C.eeglab);
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab('nogui');

%% === PARAMÈTRES ===
base_folder = fullfile(C.racine, 'Data', 'Cognition', 'EEG', 'Study_all');
% lecture en local, jeu synchronise depuis le partage le 03/09
output_folder = fullfile(C.racine, 'Data', 'Cognition', 'EEG', 'Study_all', 'Résultats_all');
figures_folder = fullfile(output_folder, 'Figures_ANT_Indicateurs');

% Dossiers des 3 types de triggers
data_folders = struct();
data_folders.fleche = fullfile(base_folder, 'ANT_all', 'processed', 'trigger_fleche');
data_folders.sonore = fullfile(base_folder, 'ANT_all', 'processed', 'trigger_sonore');
data_folders.indice = fullfile(base_folder, 'ANT_all', 'processed', 'trigger_indice');

% Créer les dossiers de sortie si nécessaire
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
if ~exist(figures_folder, 'dir')
    mkdir(figures_folder);
end

%% === CHARGEMENT DU FICHIER DE MAPPING DES CONDITIONS ===
base_csv_path = fullfile(C.racine, 'Data', 'Base.csv');
fprintf('Chargement du fichier de mapping : %s\n', base_csv_path);

if ~exist(base_csv_path, 'file')
    error('Le fichier Base.csv est introuvable : %s', base_csv_path);
end

% Lire le fichier Base.csv avec différents délimiteurs possibles
try
    base_data = readtable(base_csv_path, 'Delimiter', ';', 'VariableNamingRule', 'preserve');
    if width(base_data) < 3
        base_data = readtable(base_csv_path, 'Delimiter', ',', 'VariableNamingRule', 'preserve');
    end
    if width(base_data) < 3
        base_data = readtable(base_csv_path, 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
    end
catch
    error('Impossible de lire le fichier Base.csv correctement');
end

fprintf('Nombre de colonnes : %d, Nombre de lignes : %d\n', width(base_data), height(base_data));

if width(base_data) < 3
    error('Le fichier Base.csv doit contenir au moins 3 colonnes');
end

% Créer une structure de mapping
mapping_conditions = struct();

for i = 1:height(base_data)
    participant_id = base_data{i, 1};
    
    if isnumeric(participant_id)
        participant = sprintf('P%d', participant_id);
    elseif iscell(participant_id)
        participant = char(participant_id{1});
        if ~startsWith(participant, 'P')
            participant = sprintf('P%s', participant);
        end
    else
        participant = char(participant_id);
        if ~startsWith(participant, 'P')
            participant = sprintf('P%s', participant);
        end
    end
    
    cond_pass2 = base_data{i, 2};
    if iscell(cond_pass2)
        cond_pass2 = char(cond_pass2{1});
    else
        cond_pass2 = char(cond_pass2);
    end
    
    cond_pass3 = base_data{i, 3};
    if iscell(cond_pass3)
        cond_pass3 = char(cond_pass3{1});
    else
        cond_pass3 = char(cond_pass3);
    end
    
    mapping_conditions.(participant).passation2 = cond_pass2;
    mapping_conditions.(participant).passation3 = cond_pass3;
end

fprintf('Mapping chargé pour %d participants\n', height(base_data));

%% === DÉFINITION DES COMPOSANTES PAR TYPE DE TRIGGER ===

% --- TRIGGERS FLÈCHES ---
composantes_fleche = struct();
composantes_fleche.N1.fenetre = [190, 250];
composantes_fleche.N1.func = @min;  % N1 est négative
composantes_fleche.N1.roi = {'Oz', 'P7', 'P8', 'O1', 'O2'};

composantes_fleche.P3.fenetre = [280, 500];
composantes_fleche.P3.func = @max;  % P3 est positive
composantes_fleche.P3.roi = {'Pz', 'P3', 'P4', 'PO3', 'PO4'};

% --- TRIGGERS SONORES ---
composantes_sonore = struct();
composantes_sonore.N1.fenetre = [140, 200];
composantes_sonore.N1.func = @min;  % N1 est négative
composantes_sonore.N1.roi = {'Cz', 'Fz', 'FC1', 'FC2'};

composantes_sonore.P2.fenetre = [220, 280];
composantes_sonore.P2.func = @max;  % P2 est positive
composantes_sonore.P2.roi = {'Cz', 'FC1', 'FC2', 'CP1', 'CP2'};

% --- TRIGGERS INDICE ---
composantes_indice = struct();
composantes_indice.N1.fenetre = [170, 230];
composantes_indice.N1.func = @min;  % N1 est négative
composantes_indice.N1.roi = {'Oz', 'P7', 'P8', 'O1', 'O2'};

%% === DÉFINITION DES TRIGGERS ===

% --- TRIGGERS FLÈCHES (tous les indicateurs) ---
triggers_fleche = struct();
triggers_fleche.congruent = {'fleche_1', 'fleche_3', 'fleche_5', 'fleche_7', 'fleche_9', 'fleche_11'};
triggers_fleche.incongruent = {'fleche_2', 'fleche_4', 'fleche_6', 'fleche_8', 'fleche_10', 'fleche_12'};
triggers_fleche.avec_son = {'fleche_1', 'fleche_2', 'fleche_5', 'fleche_6', 'fleche_9', 'fleche_10'};
triggers_fleche.sans_son = {'fleche_3', 'fleche_4', 'fleche_7', 'fleche_8', 'fleche_11', 'fleche_12'};
triggers_fleche.avec_son_sans_spatial = {'fleche_1', 'fleche_2'};
triggers_fleche.sans_son_sans_spatial = {'fleche_3', 'fleche_4'};
triggers_fleche.indice_spatial_sans_son = {'fleche_7', 'fleche_8', 'fleche_11', 'fleche_12'};
triggers_fleche.sans_indice_sans_son = {'fleche_3', 'fleche_4'};
triggers_fleche.valide = {'fleche_5', 'fleche_6', 'fleche_7', 'fleche_8'};
triggers_fleche.invalide = {'fleche_9', 'fleche_10', 'fleche_11', 'fleche_12'};

% --- TRIGGERS SONORES (Alerte Standard uniquement) ---
triggers_sonore = struct();
triggers_sonore.avec_son = {'Sons'};
triggers_sonore.sans_son = {'NoS'};

% --- TRIGGERS INDICE (Alerte Visuelle) ---
triggers_indice = struct();
triggers_indice.avec_indice = {'indice_correct', 'indice_incorrect'};
triggers_indice.sans_indice = {'indice_absent'};

%% === CRÉATION DU HEADER ===
header = {'Sujet', 'Passation', 'Condition'};

% Indicateurs par type de trigger
indicateurs_fleche = {'Alerte_Standard', 'Alerte_Auditive', 'Alerte_Visuelle', 'Orientation', 'Controle'};
indicateurs_sonore = {'Alerte_Standard'};  % Alerte Auditive retirée (triggers fleche_X non disponibles dans fichiers sonores)
indicateurs_indice = {'Alerte_Visuelle'};

% Colonnes pour triggers flèches (N1 et P3)
for comp = {'N1', 'P3'}
    for ind = 1:length(indicateurs_fleche)
        header{end+1} = sprintf('%s_%s_Max_fleche', comp{1}, indicateurs_fleche{ind});
        header{end+1} = sprintf('%s_%s_Moy_fleche', comp{1}, indicateurs_fleche{ind});
    end
end

% Colonnes pour triggers sonores (N1 et P2)
for comp = {'N1', 'P2'}
    for ind = 1:length(indicateurs_sonore)
        header{end+1} = sprintf('%s_%s_Max_sonore', comp{1}, indicateurs_sonore{ind});
        header{end+1} = sprintf('%s_%s_Moy_sonore', comp{1}, indicateurs_sonore{ind});
    end
end

% Colonnes pour triggers indice (N1)
for ind = 1:length(indicateurs_indice)
    header{end+1} = sprintf('N1_%s_Max_indice', indicateurs_indice{ind});
    header{end+1} = sprintf('N1_%s_Moy_indice', indicateurs_indice{ind});
end

fprintf('\n=== Configuration ===\n');
fprintf('Nombre total de colonnes : %d\n', length(header));
fprintf('\n--- Triggers flèches ---\n');
fprintf('  N1 : [%d-%d] ms, ROI : %s\n', composantes_fleche.N1.fenetre(1), composantes_fleche.N1.fenetre(2), strjoin(composantes_fleche.N1.roi, ', '));
fprintf('  P3 : [%d-%d] ms, ROI : %s\n', composantes_fleche.P3.fenetre(1), composantes_fleche.P3.fenetre(2), strjoin(composantes_fleche.P3.roi, ', '));
fprintf('  Indicateurs : %s\n', strjoin(indicateurs_fleche, ', '));
fprintf('\n--- Triggers sonores (1 indicateur) ---\n');
fprintf('  N1 : [%d-%d] ms, ROI : %s\n', composantes_sonore.N1.fenetre(1), composantes_sonore.N1.fenetre(2), strjoin(composantes_sonore.N1.roi, ', '));
fprintf('  P2 : [%d-%d] ms, ROI : %s\n', composantes_sonore.P2.fenetre(1), composantes_sonore.P2.fenetre(2), strjoin(composantes_sonore.P2.roi, ', '));
fprintf('  Indicateurs : %s\n', strjoin(indicateurs_sonore, ', '));
fprintf('\n--- Triggers indice ---\n');
fprintf('  N1 : [%d-%d] ms, ROI : %s\n', composantes_indice.N1.fenetre(1), composantes_indice.N1.fenetre(2), strjoin(composantes_indice.N1.roi, ', '));
fprintf('  Indicateurs : %s\n', strjoin(indicateurs_indice, ', '));

%% === FONCTIONS D'EXTRACTION DES ERP ===
function [val_max, val_moy, n_essais] = extraire_erp(EEG, electrodes, fenetre, func_extreme, triggers_list)
    % Extraire les indices des électrodes
    chan_indices = [];
    for e = 1:length(electrodes)
        idx = find(strcmp({EEG.chanlocs.labels}, electrodes{e}));
        if ~isempty(idx)
            chan_indices = [chan_indices, idx];
        end
    end
    
    if isempty(chan_indices)
        val_max = NaN;
        val_moy = NaN;
        n_essais = 0;
        return;
    end
    
    % Trouver les indices temporels
    [~, tmin_idx] = min(abs(EEG.times - fenetre(1)));
    [~, tmax_idx] = min(abs(EEG.times - fenetre(2)));
    
    % Identifier les epochs correspondant aux triggers
    n_epochs = length(EEG.epoch);
    idx_epochs = false(1, n_epochs);
    
    for i = 1:n_epochs
        evt = EEG.epoch(i).eventtype;
        if iscell(evt)
            evt_str = evt;
        else
            evt_str = {char(evt)};
        end
        
        for trig_idx = 1:length(triggers_list)
            if any(strcmp(evt_str, triggers_list{trig_idx}))
                idx_epochs(i) = true;
                break;
            end
        end
    end
    
    n_essais = sum(idx_epochs);
    
    if n_essais == 0
        val_max = NaN;
        val_moy = NaN;
        return;
    end
    
    % Calculer l'ERP moyen sur les électrodes et les epochs sélectionnés
    erp_data = mean(mean(EEG.data(chan_indices, :, idx_epochs), 3), 1);
    signal = erp_data(tmin_idx:tmax_idx);
    
    % Calculer les métriques
    val_max = func_extreme(signal);
    val_moy = mean(signal);
end

%% === FONCTION PRINCIPALE D'ANALYSE ===
function results = analyser_ANT_multi_triggers(data_folders, header, composantes_fleche, composantes_sonore, composantes_indice, triggers_fleche, triggers_sonore, triggers_indice, mapping_conditions)
    fprintf('\n=== Analyse ANT Multi-Triggers ===\n');
    
    % Vérifier les dossiers
    for folder_type = {'fleche', 'sonore', 'indice'}
        if ~exist(data_folders.(folder_type{1}), 'dir')
            error('Le dossier %s n''existe pas : %s', folder_type{1}, data_folders.(folder_type{1}));
        end
    end
    
    % Lister les fichiers flèche comme référence
    fichiers_fleche = dir(fullfile(data_folders.fleche, '*_fleche.set'));
    fprintf('Nombre de fichiers flèche trouvés : %d\n', length(fichiers_fleche));
    
    if isempty(fichiers_fleche)
        warning('Aucun fichier *_fleche.set trouvé');
        results = [];
        return;
    end
    
    results = [];
    n_cols = length(header);

    for f = 1:length(fichiers_fleche)
        fichier_fleche = fichiers_fleche(f).name;
        
        % Extraire le nom de base (sans _fleche.set)
        base_name = strrep(fichier_fleche, '_fleche.set', '');
        
        % Construire les noms des fichiers correspondants
        fichier_sonore = [base_name '_sonore.set'];
        fichier_indice = [base_name '_indice.set'];
        
        fprintf('\n--- Traitement : %s ---\n', base_name);
        
        % Vérifier l'existence des 3 fichiers
        path_fleche = fullfile(data_folders.fleche, fichier_fleche);
        path_sonore = fullfile(data_folders.sonore, fichier_sonore);
        path_indice = fullfile(data_folders.indice, fichier_indice);
        
        has_fleche = exist(path_fleche, 'file');
        has_sonore = exist(path_sonore, 'file');
        has_indice = exist(path_indice, 'file');
        
        fprintf('  Fichiers disponibles : fleche=%d, sonore=%d, indice=%d\n', has_fleche, has_sonore, has_indice);
        
        if ~has_fleche
            warning('Fichier flèche manquant : %s', fichier_fleche);
            continue;
        end
        
        try
            % Extraire infos du nom de fichier
            tokens = regexp(base_name, '^P?(\d+)-ANT-(\d+)', 'tokens');
            if isempty(tokens)
                warning('Nom de fichier non conforme : %s', base_name);
                continue;
            end
            parts = tokens{1};
            
            sujet_num = parts{1};
            sujet = sprintf('P%s', sujet_num);
            passation = parts{2};
            passation_num = str2double(passation);
            
            % Utiliser le mapping pour la condition
            if isfield(mapping_conditions, sujet)
                if passation_num == 2
                    condition = mapping_conditions.(sujet).passation2;
                elseif passation_num == 3
                    condition = mapping_conditions.(sujet).passation3;
                else
                    condition = sprintf('P%d', passation_num);
                end
            else
                warning('Participant %s non trouvé dans Base.csv', sujet);
                condition = sprintf('P%d', passation_num);
            end
            
            fprintf('  Sujet: %s, Passation: %s, Condition: %s\n', sujet_num, passation, condition);

            % Initialiser la ligne de résultats
            row = cell(1, n_cols);
            row{1} = sujet_num;
            row{2} = passation;
            row{3} = condition;
            col_idx = 4;
            
            %% === EXTRACTION TRIGGERS FLÈCHES ===
            EEG_fleche = pop_loadset('filename', fichier_fleche, 'filepath', data_folders.fleche);
            fprintf('  Flèche : %d epochs chargées\n', EEG_fleche.trials);
            
            for comp = {'N1', 'P3'}
                comp_name = comp{1};
                fenetre = composantes_fleche.(comp_name).fenetre;
                func_extreme = composantes_fleche.(comp_name).func;
                roi = composantes_fleche.(comp_name).roi;
                
                % ALERTE STANDARD
                [max_avec, moy_avec, ~] = extraire_erp(EEG_fleche, roi, fenetre, func_extreme, triggers_fleche.avec_son);
                [max_sans, moy_sans, ~] = extraire_erp(EEG_fleche, roi, fenetre, func_extreme, triggers_fleche.sans_son);
                row{col_idx} = max_sans - max_avec; col_idx = col_idx + 1;
                row{col_idx} = moy_sans - moy_avec; col_idx = col_idx + 1;
                
                % ALERTE AUDITIVE
                [max_avec_aud, moy_avec_aud, ~] = extraire_erp(EEG_fleche, roi, fenetre, func_extreme, triggers_fleche.avec_son_sans_spatial);
                [max_sans_aud, moy_sans_aud, ~] = extraire_erp(EEG_fleche, roi, fenetre, func_extreme, triggers_fleche.sans_son_sans_spatial);
                row{col_idx} = max_sans_aud - max_avec_aud; col_idx = col_idx + 1;
                row{col_idx} = moy_sans_aud - moy_avec_aud; col_idx = col_idx + 1;
                
                % ALERTE VISUELLE
                [max_indice, moy_indice, ~] = extraire_erp(EEG_fleche, roi, fenetre, func_extreme, triggers_fleche.indice_spatial_sans_son);
                [max_sans_ind, moy_sans_ind, ~] = extraire_erp(EEG_fleche, roi, fenetre, func_extreme, triggers_fleche.sans_indice_sans_son);
                row{col_idx} = max_sans_ind - max_indice; col_idx = col_idx + 1;
                row{col_idx} = moy_sans_ind - moy_indice; col_idx = col_idx + 1;
                
                % ORIENTATION
                [max_valide, moy_valide, ~] = extraire_erp(EEG_fleche, roi, fenetre, func_extreme, triggers_fleche.valide);
                [max_invalide, moy_invalide, ~] = extraire_erp(EEG_fleche, roi, fenetre, func_extreme, triggers_fleche.invalide);
                row{col_idx} = max_invalide - max_valide; col_idx = col_idx + 1;
                row{col_idx} = moy_invalide - moy_valide; col_idx = col_idx + 1;
                
                % CONTRÔLE EXÉCUTIF
                [max_cong, moy_cong, ~] = extraire_erp(EEG_fleche, roi, fenetre, func_extreme, triggers_fleche.congruent);
                [max_incong, moy_incong, ~] = extraire_erp(EEG_fleche, roi, fenetre, func_extreme, triggers_fleche.incongruent);
                row{col_idx} = max_incong - max_cong; col_idx = col_idx + 1;
                row{col_idx} = moy_incong - moy_cong; col_idx = col_idx + 1;
            end
            
            %% === EXTRACTION TRIGGERS SONORES ===
            if has_sonore
                EEG_sonore = pop_loadset('filename', fichier_sonore, 'filepath', data_folders.sonore);
                fprintf('  Sonore : %d epochs chargées\n', EEG_sonore.trials);
                
                for comp = {'N1', 'P2'}
                    comp_name = comp{1};
                    fenetre = composantes_sonore.(comp_name).fenetre;
                    func_extreme = composantes_sonore.(comp_name).func;
                    roi = composantes_sonore.(comp_name).roi;
                    
                    % ALERTE STANDARD (Sons vs NoS)
                    [max_avec, moy_avec, ~] = extraire_erp(EEG_sonore, roi, fenetre, func_extreme, triggers_sonore.avec_son);
                    [max_sans, moy_sans, ~] = extraire_erp(EEG_sonore, roi, fenetre, func_extreme, triggers_sonore.sans_son);
                    row{col_idx} = max_sans - max_avec; col_idx = col_idx + 1;
                    row{col_idx} = moy_sans - moy_avec; col_idx = col_idx + 1;
                end
            else
                % Remplir avec NaN si fichier sonore manquant
                for comp = {'N1', 'P2'}
                    row{col_idx} = NaN; col_idx = col_idx + 1;
                    row{col_idx} = NaN; col_idx = col_idx + 1;
                end
                fprintf('  ⚠️ Fichier sonore manquant - NaN insérés\n');
            end
            
            %% === EXTRACTION TRIGGERS INDICE ===
            if has_indice
                EEG_indice = pop_loadset('filename', fichier_indice, 'filepath', data_folders.indice);
                fprintf('  Indice : %d epochs chargées\n', EEG_indice.trials);
                
                fenetre = composantes_indice.N1.fenetre;
                func_extreme = composantes_indice.N1.func;
                roi = composantes_indice.N1.roi;
                
                % ALERTE VISUELLE (sans_indice vs avec_indice)
                [max_avec, moy_avec, ~] = extraire_erp(EEG_indice, roi, fenetre, func_extreme, triggers_indice.avec_indice);
                [max_sans, moy_sans, ~] = extraire_erp(EEG_indice, roi, fenetre, func_extreme, triggers_indice.sans_indice);
                row{col_idx} = max_sans - max_avec; col_idx = col_idx + 1;
                row{col_idx} = moy_sans - moy_avec; col_idx = col_idx + 1;
            else
                % Remplir avec NaN si fichier indice manquant
                row{col_idx} = NaN; col_idx = col_idx + 1;
                row{col_idx} = NaN; col_idx = col_idx + 1;
                fprintf('  ⚠️ Fichier indice manquant - NaN insérés\n');
            end
            
            % Vérifier la cohérence
            if col_idx - 1 ~= n_cols
                warning('Incohérence colonnes : attendu %d, obtenu %d', n_cols, col_idx - 1);
            end
            
            % Ajouter la ligne aux résultats
            if isempty(results)
                results = row;
            else
                results = [results; row];
            end
            
            fprintf('  ✓ Traitement réussi\n');
            
        catch ME
            warning('Erreur lors du traitement de %s : %s', base_name, ME.message);
            fprintf('Stack trace:\n%s\n', getReport(ME));
            continue;
        end
    end
    
    fprintf('\n=== Nombre de sujets traités : %d ===\n', size(results, 1));
end

%% === EXÉCUTION ===
fprintf('\n========================================\n');
fprintf('  EXTRACTION DES INDICATEURS ERP ANT\n');
fprintf('      Version 4.0 - Multi-Triggers\n');
fprintf('========================================\n');

% Lancer l'analyse
results_ANT = analyser_ANT_multi_triggers(data_folders, header, composantes_fleche, composantes_sonore, composantes_indice, triggers_fleche, triggers_sonore, triggers_indice, mapping_conditions);

%% === SAUVEGARDE DES RÉSULTATS ===
if ~isempty(results_ANT)
    header = header(:)';
    
    fprintf('\nTaille header : %d colonnes\n', length(header));
    fprintf('Taille results : %d lignes x %d colonnes\n', size(results_ANT, 1), size(results_ANT, 2));
    
    if size(results_ANT, 2) ~= length(header)
        warning('Incohérence : header a %d colonnes, results a %d colonnes', length(header), size(results_ANT, 2));
    end
    
    final_results = cell(size(results_ANT, 1) + 1, length(header));
    final_results(1, :) = header;
    final_results(2:end, :) = results_ANT;
    
    output_file = fullfile(output_folder, 'ANT_indicateurs_ERP_multi_triggers.xlsx');
    writecell(final_results, output_file);
    
    fprintf('\n✅ Fichier sauvegardé : %s\n', output_file);
    fprintf('Nombre de sujets traités : %d\n', size(results_ANT, 1));
    
    % Afficher un résumé
    fprintf('\n=== RÉSUMÉ ===\n');
    
    fprintf('\n--- Triggers flèches (5 indicateurs × 2 composantes) ---\n');
    fprintf('  N1 [%d-%d ms] : %s\n', composantes_fleche.N1.fenetre(1), composantes_fleche.N1.fenetre(2), strjoin(composantes_fleche.N1.roi, ', '));
    fprintf('  P3 [%d-%d ms] : %s\n', composantes_fleche.P3.fenetre(1), composantes_fleche.P3.fenetre(2), strjoin(composantes_fleche.P3.roi, ', '));
    
    fprintf('\n--- Triggers sonores (1 indicateur × 2 composantes) ---\n');
    fprintf('  N1 [%d-%d ms] : %s\n', composantes_sonore.N1.fenetre(1), composantes_sonore.N1.fenetre(2), strjoin(composantes_sonore.N1.roi, ', '));
    fprintf('  P2 [%d-%d ms] : %s\n', composantes_sonore.P2.fenetre(1), composantes_sonore.P2.fenetre(2), strjoin(composantes_sonore.P2.roi, ', '));
    
    fprintf('\n--- Triggers indice (1 indicateur × 1 composante) ---\n');
    fprintf('  N1 [%d-%d ms] : %s\n', composantes_indice.N1.fenetre(1), composantes_indice.N1.fenetre(2), strjoin(composantes_indice.N1.roi, ', '));
    
    fprintf('\nIndicateurs calculés (différences) :\n');
    fprintf('  Flèches :\n');
    fprintf('    - Alerte Standard : sans_son - avec_son\n');
    fprintf('    - Alerte Auditive : sans_son - avec_son (essais sans indice spatial)\n');
    fprintf('    - Alerte Visuelle : avec_indice - sans_indice (essais sans son)\n');
    fprintf('    - Orientation : invalide - valide\n');
    fprintf('    - Contrôle Exécutif : incongruent - congruent\n');
    fprintf('  Sonores :\n');
    fprintf('    - Alerte Standard : NoS - Sons\n');
    fprintf('  Indice :\n');
    fprintf('    - Alerte Visuelle : avec_indice - sans_indice\n');
    
    % Compter le nombre de colonnes
    n_cols_fleche = 2 * 5 * 2;  % 2 composantes × 5 indicateurs × 2 mesures
    n_cols_sonore = 2 * 1 * 2;  % 2 composantes × 1 indicateur × 2 mesures
    n_cols_indice = 1 * 1 * 2;  % 1 composante × 1 indicateur × 2 mesures
    n_cols_total = 3 + n_cols_fleche + n_cols_sonore + n_cols_indice;
    
    fprintf('\nNombre de colonnes :\n');
    fprintf('  - Info (Sujet, Passation, Condition) : 3\n');
    fprintf('  - Flèches : %d\n', n_cols_fleche);
    fprintf('  - Sonores : %d\n', n_cols_sonore);
    fprintf('  - Indice : %d\n', n_cols_indice);
    fprintf('  - Total : %d colonnes\n', n_cols_total);
    
else
    fprintf('\n⚠️ Aucune donnée à sauvegarder\n');
end

fprintf('\n========================================\n');
fprintf('  FIN DU TRAITEMENT\n');
fprintf('========================================\n');