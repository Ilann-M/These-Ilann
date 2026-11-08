addpath(fullfile(fileparts(mfilename('fullpath')), '..'));  % chemins.m, a la racine du depot
C = chemins();
addpath(C.eeglab);
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% === PARAMÈTRES ===
base_folder = fullfile(C.racine, 'Data', 'Cognition', 'EEG', 'Study_all');
output_folder = fullfile(C.racine, 'Data', 'Cognition', 'EEG', 'Study_all', 'Résultats_all');

% === DÉFINITION DES ROI (uniquement ROI2) ===
ROIs = struct();
ROIs.N1 = {'Oz', 'P7', 'P8', 'O1', 'O2'};
ROIs.P3 = {'Pz', 'P3', 'P4', 'PO3', 'PO4'};

% Triggers d'intérêt
triggers_interet = {'Valide', 'Invalide'};

% Définition des composantes
composantes = struct();
composantes.N1.fenetre = [110 170];
composantes.N1.func = @min;

composantes.P3.fenetre = [300 500];
composantes.P3.func = @max;

comp_names = {'N1', 'P3'};

% === FONCTION D'ANALYSE ===
function results = analyser_dossier(data_folder, ROIs, triggers_interet, composantes, comp_names)
    % Header
    header = {'Sujet', 'Passation', 'Condition'};
    for c = 1:length(comp_names)
        for t = 1:length(triggers_interet)
            header{end+1} = [comp_names{c} '_Pic_' triggers_interet{t}];
            header{end+1} = [comp_names{c} '_Moy_' triggers_interet{t}];
        end
    end
    
    fprintf('Analyse du dossier : %s\n', data_folder);
    
    if ~exist(data_folder, 'dir')
        warning('Le dossier %s n''existe pas', data_folder);
        results = header;
        return;
    end
    
    fichiers = dir(fullfile(data_folder, '*.set'));
    fprintf('Fichiers .set trouvés : %d\n', length(fichiers));
    
    if isempty(fichiers)
        results = header;
        return;
    end
    
    results = cell(length(fichiers)+1, length(header));
    results(1,:) = header;
    row_idx = 2;

    for f = 1:length(fichiers)
        fichier = fichiers(f).name;
        fprintf('Traitement : %s\n', fichier);
        
        try
            EEG = pop_loadset('filename', fichier, 'filepath', data_folder);

            tokens = regexp(fichier, '^P?(\d+)-(1B|2B)-(\d+)-(away|close)\.set$', 'tokens');
            if isempty(tokens)
                warning('Nom non conforme : %s', fichier);
                continue;
            end
            parts = tokens{1};

            row = {parts{1}, parts{3}, parts{4}};
            
            for c = 1:length(comp_names)
                comp_name = comp_names{c};
                fenetre = composantes.(comp_name).fenetre;
                func_extreme = composantes.(comp_name).func;
                roi_electrodes = ROIs.(comp_name);

                [~, tmin_idx] = min(abs(EEG.times - fenetre(1)));
                [~, tmax_idx] = min(abs(EEG.times - fenetre(2)));

                % Indices des électrodes de la ROI
                chan_indices = [];
                for e = 1:length(roi_electrodes)
                    idx = find(strcmp({EEG.chanlocs.labels}, roi_electrodes{e}));
                    if ~isempty(idx), chan_indices = [chan_indices, idx]; end
                end

                for t = 1:length(triggers_interet)
                    trig = triggers_interet{t};
                    
                    % Identifier les epochs
                    idx_epochs = false(1, length(EEG.epoch));
                    for i = 1:length(EEG.epoch)
                        evt = EEG.epoch(i).eventtype;
                        if iscell(evt), evt = strjoin(evt, '_'); end
                        if contains(string(evt), trig)
                            idx_epochs(i) = true;
                        end
                    end

                    if ~isempty(chan_indices) && any(idx_epochs)
                        erp_roi = mean(mean(EEG.data(chan_indices, :, idx_epochs), 3), 1);
                        signal = erp_roi(tmin_idx:tmax_idx);
                        row{end+1} = func_extreme(signal);
                        row{end+1} = mean(signal);
                    else
                        row{end+1} = NaN;
                        row{end+1} = NaN;
                    end
                end
            end

            results(row_idx,:) = row;
            row_idx = row_idx + 1;
            
        catch ME
            warning('Erreur %s : %s', fichier, ME.message);
        end
    end
    
    results = results(1:row_idx-1, :);
    fprintf('Lignes ajoutées : %d\n', row_idx-2);
end

% === CRÉER LE DOSSIER DE SORTIE ===
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% === ANALYSE 1B ===
fprintf('\n=== ANALYSE 1B ===\n');
results_1B = analyser_dossier(fullfile(base_folder, '1B_all', 'processed'), ROIs, triggers_interet, composantes, comp_names);
writecell(results_1B, fullfile(output_folder, '1B_N1_P3_extraction.csv'));
fprintf('✅ 1B_N1_P3_extraction.csv (%d sujets)\n', size(results_1B,1)-1);

% === ANALYSE 2B ===
fprintf('\n=== ANALYSE 2B ===\n');
results_2B = analyser_dossier(fullfile(base_folder, '2B_all', 'processed'), ROIs, triggers_interet, composantes, comp_names);
writecell(results_2B, fullfile(output_folder, '2B_N1_P3_extraction.csv'));
fprintf('✅ 2B_N1_P3_extraction.csv (%d sujets)\n', size(results_2B,1)-1);

% === RÉSUMÉ ===
fprintf('\n=== RÉSUMÉ ===\n');
fprintf('N1 [110-170 ms] : ROI = {Oz, P7, P8, O1, O2}\n');
fprintf('P3 [300-500 ms] : ROI = {Pz, P3, P4, PO3, PO4}\n');
fprintf('Métriques : Pic + Moyenne × 2 triggers = 8 colonnes de données\n');