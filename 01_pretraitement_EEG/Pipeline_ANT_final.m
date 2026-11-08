%% ========================================================================
% SCRIPT : Pipeline ANT v10.0
%
% Comprehensive EEG preprocessing pipeline with MULTI-TRIGGER EPOCHING:
% - THREE OUTPUT TYPES per participant:
%   * trigger_fleche/  : Epochs on arrow stimuli [-200, 1000ms]
%   * trigger_indice/  : Epochs on visual cues [-200, 500ms]
%   * trigger_sonore/  : Epochs on auditory cues [-200, 600ms]
%
% NEW in v10.0:
% - High-pass filter changed from 0.5 Hz to 1 Hz
% - Creates synthetic triggers for indices and sounds based on:
%   * aut24: Derived from fleche_X type (no CSV needed)
%   * print25: Derived from CSV alignment data
% - Three separate epoched files per participant
% - Trigger types created:
%   * Indices: indice_correct, indice_incorrect, indice_absent
%   * Sons: Sons, NoS
%
% TIMING RELATIONSHIPS:
% - Sound → Arrow: 630ms (trigger_sonore at -630ms from arrow)
% - Cue → Arrow: 190ms (trigger_indice at -190ms from arrow)
%
% Inherits all v9.4 features:
% - ADAPTIVE trigger conversion with fallback logic
% - SMART burst detection
% - TWO-PASS trigger cleaning
% - Enhanced ICA with mandatory + conditional logic
% - Proper epoch rejection with logging
% - Automatic file renaming with exposure type
% - Parallel processing: 4 workers
%%========================================================================

clear; clc; close all;

%% ========================================================================
%% PARAMETERS
%% ========================================================================

% GUI and general settings
useGUI = false;
maxInterpolations = 2;

% ICA parameters (from v9.4)
iclabelEyeThreshold = 0.60;
iclabelBrainThreshold = 0.10;
eogCorrThreshold = 0.35;
eogSafetyThreshold = 0.70;
eyeSafetyThreshold = 0.40;
maxEyeICs = 3;

% Noise detection parameters (from v9.3.3)
extremeAmpThresholdUv = 150;
extremeAmpPropThreshold = 0.10;
asrMaxBoundaries = 50;

% Trigger validation parameters (DATASET-SPECIFIC)
triggerTimingTolerance1_print25 = [480, 580]; % ms between Sons/NoS and indice (~530ms)
triggerTimingTolerance2_print25 = [140, 240]; % ms between indice and fleches (~190ms)
triggerTimingTolerance1_aut24 = [550, 630];   % ms between Sons/NoS and fleches (~600ms)
triggerTimingTolerance2_aut24 = [200, 1700];  % ms between fleches and RepJ

% NEW v10: Epoch windows for each trigger type
epochWindow_fleche = [-0.2, 1.0];    % [-200ms, 1000ms] for arrows
epochWindow_indice = [-0.2, 0.5];    % [-200ms, 500ms] for visual cues
epochWindow_sonore = [-0.2, 0.6];    % [-200ms, 600ms] for auditory cues

% NEW v10: Timing offsets for synthetic trigger creation
offset_son_to_fleche_ms = 580;      % Sound appears 630ms before arrow
offset_indice_to_fleche_ms = 190;   % Visual cue appears 190ms before arrow

% ========================================================================
% NEW: MODE SUBSTRAT TEMPS-FRÉQUENCE
%   'erp'  : comportement Article 1 STRICTEMENT INCHANGÉ (3 fichiers époqués)
%   'freq' : sauvegarde le CONTINU prétraité (post-ICA) pour analyses TF
%            - fenêtres de conservation élargies (les époques longues survivront)
%            - indices RÉELS conservés (pas de triggers synthétiques à 190 ms)
%            - corrections de timing son/indice reportées au script d'époquage TF
%              (audit synchro : Sons +50 ms, NoS +110 ms print25 / +50 ms aut24)
% ========================================================================
runMode          = 'erp';    % chaine rapportee dans la these ('freq' = mode exploratoire non rapporte)
keepPreSec_freq  = 2.0;      % vs 0.2 en erp : marge avant 1er événement d'un bloc
keepPostSec_freq = 3.0;      % vs 1.5 en erp : marge après dernier événement d'un bloc

% Paths
% Racine des données : definie dans chemins.m (racine du depot).
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));  % chemins.m, a la racine du depot
C = chemins();
dataRoot = C.racine;
baseCsvPath = fullfile(dataRoot, 'Data', 'Base.csv');
print25TriggerPath = fullfile(dataRoot, 'Campagne_print_25', 'Cognition', 'Tests-cog', 'Triggers_Extracted');

% Parallel configuration
numWorkers = 6;

% Adjacency map for interpolation
adjacencyMap = containers.Map();
adjacencyMap('Fp1') = {'AF3','F7'}; adjacencyMap('AF3') = {'Fp1','F3'}; adjacencyMap('F7') = {'Fp1','F3'};
adjacencyMap('F3') = {'AF3','FC1','F7'}; adjacencyMap('FC1') = {'F3','C3','Fz'}; adjacencyMap('C3') = {'FC1','CP1','T7'};
adjacencyMap('CP1') = {'C3','P3','CP5'}; adjacencyMap('CP5') = {'CP1','P7'}; adjacencyMap('P7') = {'CP5','P3'};
adjacencyMap('P3') = {'P7','Pz','PO3'}; adjacencyMap('Pz') = {'P3','PO3','PO4'}; adjacencyMap('PO3') = {'P3','O1'};
adjacencyMap('O1') = {'PO3','Oz'}; adjacencyMap('Oz') = {'O1','O2'}; adjacencyMap('O2') = {'Oz','PO4'};
adjacencyMap('PO4') = {'O2','P4'}; adjacencyMap('P4') = {'PO4','P8','Pz'}; adjacencyMap('P8') = {'P4','CP6'};
adjacencyMap('CP6') = {'P8','CP2'}; adjacencyMap('CP2') = {'CP6','C4','P4'}; adjacencyMap('C4') = {'CP2','FC2','T8'};
adjacencyMap('T8') = {'C4','FC6'}; adjacencyMap('FC6') = {'T8','F4','C4'}; adjacencyMap('FC2') = {'F4','C4','Fz'};
adjacencyMap('F4') = {'AF4','FC2','F8'}; adjacencyMap('F8') = {'F4','FC6'}; adjacencyMap('AF4') = {'Fp2','F4'};
adjacencyMap('Fp2') = {'AF4','F8'}; adjacencyMap('Fz') = {'FC1','FC2','Cz'}; adjacencyMap('Cz') = {'Fz','Pz'};
adjacencyMap('TP9') = {'TP10'}; adjacencyMap('TP10') = {'TP9'};

%% ========================================================================
%% DATASETS CONFIGURATION
%% ========================================================================

antOutput = fullfile(dataRoot, 'Data', 'Cognition', 'EEG', 'Study_all', 'ANT_all', 'processed');
datasets = {
    struct('input', fullfile(dataRoot, 'Campagne_print_25', 'Cognition', 'EEG', 'Brut'), ...
           'output', antOutput, ...
           'type', 'print25'), ...
    struct('input', fullfile(dataRoot, 'Campagne_aut_24', 'Cognition', 'EEG', 'EEG_Brut'), ...
           'output', antOutput, ...
           'type', 'aut24')
};

%% ========================================================================
%% INITIALIZATION
%% ========================================================================

% Initialize EEGLAB
addpath(fileparts(which('eeglab.m')));
eeglab nogui;

% Load exposure mapping
fprintf('📖 Loading exposure mapping from Base.csv...\n');
exposureMap = load_exposure_mapping(baseCsvPath);
if isempty(exposureMap)
    fprintf('⚠️ Warning: Could not load Base.csv. Files will be saved without exposure suffix.\n');
else
    fprintf('✅ Loaded exposure data for %d participants\n', length(fieldnames(exposureMap)));
end

% Start parallel pool
poolobj = gcp('nocreate');
if isempty(poolobj)
    fprintf('🚀 Démarrage du pool parallèle avec %d workers...\n', numWorkers);
    poolobj = parpool('local', numWorkers);
else
    fprintf('✅ Pool parallèle existant détecté (%d workers).\n', poolobj.NumWorkers);
    if poolobj.NumWorkers ~= numWorkers
        fprintf('⚠️ Pool has %d workers but need %d. Restarting...\n', poolobj.NumWorkers, numWorkers);
        delete(poolobj);
        poolobj = parpool('local', numWorkers);
    end
end

warning('off', 'MATLAB:dispatcher:InexactMatch');
warning('off', 'pop_biosig:warning');

%% ========================================================================
%% MAIN PROCESSING LOOP
%% ========================================================================

for d = 1:length(datasets)
    input_folder = datasets{d}.input;
    output_folder = datasets{d}.output;
    set_type = datasets{d}.type;

    if ~exist(input_folder, 'dir')
        fprintf('❌ Dossier d''entrée introuvable: %s\n', input_folder);
        continue;
    end
    
    % NEW v10: Create output subdirectories
    output_fleche = fullfile(output_folder, 'trigger_fleche');
    output_indice = fullfile(output_folder, 'trigger_indice');
    output_sonore = fullfile(output_folder, 'trigger_sonore');
    
    if ~exist(output_fleche, 'dir'), mkdir(output_fleche); end
    if ~exist(output_indice, 'dir'), mkdir(output_indice); end
    if ~exist(output_sonore, 'dir'), mkdir(output_sonore); end
    
    fprintf('✅ Dossiers de sortie créés:\n');
    fprintf('   - %s\n', output_fleche);
    fprintf('   - %s\n', output_indice);
    fprintf('   - %s\n', output_sonore);

    % NEW: dossier substrat continu (mode freq uniquement)
    output_freq = fullfile(output_folder, 'continuous_freq');
    if strcmp(runMode, 'freq')
        if ~exist(output_freq, 'dir'), mkdir(output_freq); end
        fprintf('   - [FREQ] %s\n', output_freq);
    end

    % Find ANT files
    ant2_files = dir(fullfile(input_folder, '*ANT-2*.bdf'));
    ant3_files = dir(fullfile(input_folder, '*ANT-3*.bdf'));
    all_files = [ant2_files; ant3_files];

    if isempty(all_files)
        fprintf('⚠️ Aucun fichier *ANT-2*.bdf ou *ANT-3*.bdf trouvé dans: %s\n', input_folder);
        continue;
    end

    fprintf('\n========================================\n');
    fprintf('Dataset: %s (v10.0 - Multi-trigger)\n', set_type);
    fprintf('Fichiers à traiter: %d\n', length(all_files));
    fprintf('Workers parallèles: %d\n', numWorkers);
    fprintf('========================================\n\n');

    % Create log file
    logPath = fullfile(output_folder, 'pipeline_log_v10.csv');
    if ~exist(logPath, 'file')
        fid = fopen(logPath, 'w');
        fprintf(fid, 'filename,invalid_triggers,interpolated_channels,removed_ics,rej_fleche,rej_indice,rej_sonore,final_fleche,final_indice,final_sonore,cuts,asr_boundaries,extreme_noise\n');        fclose(fid);
    end

    numFiles = length(all_files);
    filenames = {all_files.name};
    filepaths = cell(numFiles, 1);
    for f = 1:numFiles
        filepaths{f} = fullfile(input_folder, filenames{f});
    end

    % Set timing tolerances based on dataset
    if strcmp(set_type, 'print25')
        timing1 = triggerTimingTolerance1_print25;
        timing2 = triggerTimingTolerance2_print25;
    else
        timing1 = triggerTimingTolerance1_aut24;
        timing2 = triggerTimingTolerance2_aut24;
    end

    % Process files in parallel
    parfor f = 1:numFiles
        try
            process_single_file_v10(filepaths{f}, filenames{f}, ...
                output_fleche, output_indice, output_sonore, ...
                set_type, adjacencyMap, maxInterpolations, ...
                iclabelEyeThreshold, iclabelBrainThreshold, eogCorrThreshold, ...
                eogSafetyThreshold, eyeSafetyThreshold, maxEyeICs, ...
                extremeAmpThresholdUv, extremeAmpPropThreshold, asrMaxBoundaries, ...
                timing1, timing2, exposureMap, logPath, print25TriggerPath, ...
                epochWindow_fleche, epochWindow_indice, epochWindow_sonore, ...
                offset_son_to_fleche_ms, offset_indice_to_fleche_ms, ...
                runMode, output_freq, keepPreSec_freq, keepPostSec_freq);
        catch ME
            fprintf('❌ Erreur fichier %s: %s\n', filenames{f}, ME.message);
            fprintf('   Stack: %s\n', ME.getReport('basic'));
        end
    end

    fprintf('\n✅ Dataset %s terminé (traitement parallèle v10.0).\n', set_type);
end

warning('on', 'all');
fprintf('\n========================================\n');
fprintf('TRAITEMENT PARALLÈLE TERMINÉ (v10.0)\n');
fprintf('========================================\n');

%% ========================================================================
%% MAIN WORKER FUNCTION
%% ========================================================================

function process_single_file_v10(filepath, filename, output_fleche, output_indice, output_sonore, ...
    set_type, adjacencyMap, maxInterpolations, iclabelEyeThreshold, ...
    iclabelBrainThreshold, eogCorrThreshold, eogSafetyThreshold, ...
    eyeSafetyThreshold, maxEyeICs, extremeAmpThresholdUv, ...
    extremeAmpPropThreshold, asrMaxBoundaries, triggerTimingTolerance1, ...
    triggerTimingTolerance2, exposureMap, logPath, print25TriggerPath, ...
    epochWindow_fleche, epochWindow_indice, epochWindow_sonore, ...
    offset_son_ms, offset_indice_ms, ...
    runMode, output_freq, keepPreSec_freq, keepPostSec_freq)

    ensure_eeglab();   % init EEGLAB une seule fois par worker (évite les appels réseau répétés)

    workerID = get_worker_id();

    % NEW: mode substrat TF vs mode ERP (défaut). Fenêtres de conservation adaptées.
    isFreq = strcmp(runMode, 'freq');
    if isFreq
        keepPreSec = keepPreSec_freq; keepPostSec = keepPostSec_freq;
        fprintf('[Worker %d] 🚀 Traitement v10 [MODE FREQ]: %s\n', workerID, filename);
    else
        keepPreSec = 0.2; keepPostSec = 1.5;   % valeurs Article 1 (inchangées)
        fprintf('[Worker %d] 🚀 Traitement v10: %s\n', workerID, filename);
    end

    % Initialize tracking variables
    interpolatedChans = {};
    comps_to_reject = [];
    invalidTriggers = 0;
    extremeNoiseSegments = 0;
    csvData = [];  % Will be populated for print25
    cutSegmentsSec = {};
    asrBoundariesAdded = 0;
    flechesInitial = 0;
    noiseBurstRemoved = 0;

    % [FREQ] Reprise : ignore les fichiers dont le continu existe déjà
    if isFreq
        [pId0, att0] = parse_filename_matlab(filename);
        expo0 = get_exposure_type(pId0, att0, exposureMap);
        [~, nm0, ~] = fileparts(filename);
        if ~isempty(expo0)
            skipName = sprintf('%s-%s_%s_continuous.set', nm0, expo0, set_type);
        else
            skipName = sprintf('%s_%s_continuous.set', nm0, set_type);
        end
        if exist(fullfile(output_freq, skipName), 'file')
            fprintf('[Worker %d] ⏭️ [FREQ] déjà fait, ignoré: %s\n', workerID, skipName);
            return;
        end
    end

    %% === STEP 1: Import BDF ===
    if ~exist(filepath, 'file')
        fprintf('[Worker %d] ❌ Fichier introuvable: %s\n', workerID, filepath);
        return;
    end

    EEG = [];
    try
        EEG = pop_biosig(filepath);
    catch
        try
            EEG = pop_fileio(filepath);
        catch ME
            fprintf('[Worker %d] ❌ Import échoué: %s\n', workerID, ME.message);
            return;
        end
    end
    EEG = eeg_checkset(EEG);
    fprintf('[Worker %d] ✅ Import: %d channels, %.1f sec\n', workerID, EEG.nbchan, EEG.xmax);

    %% === STEP 2: Trigger conversion ===
    [EEG, new_events] = convert_triggers(EEG, set_type, workerID);
    
    %% === STEP 3: Align and rename fleches for print25 ===
    if strcmp(set_type, 'print25')
        [participantId, passationNum] = parse_filename_matlab(filename);
        
        if ~isempty(participantId) && passationNum > 0
            try
                [EEG, alignmentQuality, csvData] = align_and_rename_fleches_print25_v10(EEG, print25TriggerPath, participantId, passationNum, workerID);
                fprintf('[Worker %d] 🎯 Alignment quality: R²=%.3f\n', workerID, alignmentQuality);
            catch ME
                fprintf('[Worker %d] ⚠️ Échec alignment fleches: %s\n', workerID, ME.message);
            end
        end
    end

    %% === STEP 4: Two-pass trigger cleaning ===
    [EEG, invalidTriggers] = clean_triggers_two_pass(EEG, set_type, triggerTimingTolerance1, triggerTimingTolerance2, workerID);

    %% === STEP 5: Create synthetic triggers for indices ===
    % En mode freq : on CONSERVE les indices réels (indice_raw) et on n'applique
    % AUCUNE correction de timing ici — tout est reporté au script d'époquage TF.
    if ~isFreq
        EEG = create_synthetic_triggers(EEG, set_type, offset_indice_ms, workerID);

        %% === STEP 5b: Apply temporal correction for aut_24 sound triggers ===
        if strcmp(set_type, 'aut24')
            EEG = correct_sound_triggers_aut24(EEG, workerID);
        end
    end

    %% === STEP 6: Channel correction and selection ===
    EEG = prepare_channels(EEG);

    %% === STEP 7: Filtering (NEW: 1 Hz instead of 0.5 Hz) ===
    fprintf('[Worker %d] 🔧 Filtering: 1-40 Hz\n', workerID);
    EEG = pop_eegfiltnew(EEG, 1, 40);  % Changed from 0.5 to 1 Hz
    EEG = eeg_checkset(EEG);

    %% === STEP 8: Detect and remove extreme noise ===
    try
        [EEG, extremeNoiseSegments] = detect_and_remove_extreme_noise(EEG, extremeAmpThresholdUv);
        if extremeNoiseSegments > 0
            fprintf('[Worker %d] 🔥 %d segments de bruit extrême supprimés\n', workerID, extremeNoiseSegments);
        end
    catch ME
        fprintf('[Worker %d] ⚠️ Détection de bruit extrême échouée: %s\n', workerID, ME.message);
    end

    %% === STEP 9: ASR repair ===
    try
        if exist('pop_clean_rawdata','file')
            EEG = pop_clean_rawdata(EEG, ...
                'FlatlineCriterion', 5, 'Highpass', 'off', 'ChannelCriterion', 'off', ...
                'LineNoiseCriterion', 'off', 'BurstCriterion', 30, ...
                'BurstRejection', 'off', 'WindowCriterion', 'off');
        end
    catch
    end
    EEG = eeg_checkset(EEG);

    %% === STEP 10: Remove long silent segments ===
    EEG = remove_silent_segments(EEG, keepPreSec, keepPostSec);

    %% === STEP 11: ASR limited rejection ===
    EEG = asr_limited_rejection(EEG, extremeAmpThresholdUv, extremeAmpPropThreshold, asrMaxBoundaries, workerID);

    %% === STEP 12: Re-reference BEFORE ICA ===
    EEG = apply_reference(EEG);

    %% === STEP 13: ICA ===
    try
        fprintf('[Worker %d] 🧠 Running ICA...\n', workerID);
        EEG = pop_runica(EEG, 'icatype', 'runica');
    catch ME
        fprintf('[Worker %d] ⚠️ ICA échoué: %s\n', workerID, ME.message);
    end

    %% === STEP 14: IC rejection (Enhanced logic from v9.4) ===
    [EEG, comps_to_reject] = reject_eye_components(EEG, iclabelEyeThreshold, iclabelBrainThreshold, ...
        eogCorrThreshold, eogSafetyThreshold, eyeSafetyThreshold, maxEyeICs, workerID);

    %% === STEP 15: Interpolation ===
    [EEG, interpolatedChans] = interpolate_bad_channels(EEG, adjacencyMap, maxInterpolations);

    %% === STEP 16: Remove reference electrodes ===
    EEG = remove_reference_channels(EEG);

    %% === STEP 16b [FREQ]: Save continuous substrate and STOP ===
    % En mode freq, on sauvegarde le continu prétraité (post-ICA, montage scalp)
    % et on saute l'époquage ERP : le ré-époquage TF se fait dans un script dédié.
    if isFreq
        [participantId, attemptNumber] = parse_filename_matlab(filename);
        exposureType = get_exposure_type(participantId, attemptNumber, exposureMap);
        [~, name, ~] = fileparts(filename);
        if ~isempty(exposureType)
            outname = sprintf('%s-%s_%s_continuous.set', name, exposureType, set_type);
        else
            outname = sprintf('%s_%s_continuous.set', name, set_type);
        end
        EEG.setname = sprintf('%s_%s_continuous', name, set_type);
        pop_saveset(EEG, 'filename', fullfile(output_freq, outname));
        fprintf('[Worker %d] 💾 [FREQ] Continu sauvegardé (%d canaux, %.0f s): %s\n', ...
            workerID, EEG.nbchan, EEG.xmax, outname);
        return;
    end

    %% === STEP 17: Store preprocessed continuous data ===
    EEG_continuous = EEG;
    fprintf('[Worker %d] 💾 Prétraitement continu terminé, création des 3 fichiers époqués...\n', workerID);

    %% === STEP 18: Create three epoched versions ===
    [participantId, attemptNumber] = parse_filename_matlab(filename);
    exposureType = get_exposure_type(participantId, attemptNumber, exposureMap);
    [~, name, ~] = fileparts(filename);

    % 18a: Epochs on FLECHES (arrows)
    [epochs_fleche, rej_fleche] = create_and_save_epochs_fleche(EEG_continuous, epochWindow_fleche, ...
        output_fleche, name, exposureType, workerID);

    % 18b: Epochs on INDICES (visual cues)
    [epochs_indice, rej_indice] = create_and_save_epochs_indice(EEG_continuous, epochWindow_indice, ...
        output_indice, name, exposureType, workerID);

    % 18c: Epochs on SONS (auditory cues)
    [epochs_sonore, rej_sonore] = create_and_save_epochs_sonore(EEG_continuous, epochWindow_sonore, ...
        output_sonore, name, exposureType, workerID);

    %% === STEP 19: Write log entry ===
    write_log_entry_v10(logPath, filename, invalidTriggers, interpolatedChans, ...
        comps_to_reject, rej_fleche, rej_indice, rej_sonore, ...
        epochs_fleche, epochs_indice, epochs_sonore, ...
        cutSegmentsSec, asrBoundariesAdded, extremeNoiseSegments);
    
    fprintf('[Worker %d] ✅ Terminé: %s (fleche:%d, indice:%d, sonore:%d)\n', ...
        workerID, filename, epochs_fleche, epochs_indice, epochs_sonore);
end

%% ========================================================================
%% TRIGGER CONVERSION FUNCTIONS
%% ========================================================================

function [EEG, new_events] = convert_triggers(EEG, set_type, workerID)
    % Convert raw triggers to standardized event types
    
    new_events = [];
    fprintf('[Worker %d] Triggers AVANT conversion: %d\n', workerID, length(EEG.event));
    
    switch set_type
        case 'print25'
            new_events = convert_triggers_print25(EEG, workerID);
            
        case 'aut24'
            new_events = convert_triggers_aut24(EEG, workerID);
    end
    
    EEG.event = new_events;
    EEG = eeg_checkset(EEG);
    
    % Count fleches
    fleches_count = sum(cellfun(@(x) strcmp(x, 'fleches') || startsWith(x, 'fleche_'), {new_events.type}));
    fprintf('[Worker %d] 🔍 Conversion: %d triggers totaux, %d fleches\n', workerID, length(new_events), fleches_count);
end

function new_events = convert_triggers_print25(EEG, workerID)
    % PRINT25: Convert triggers, fleches will be renamed later via CSV alignment
    new_events = [];
    i = 1;

    % Remove 65280 triggers (usually only one at the beginning)
    removed_65280 = 0;
    for i = length(EEG.event):-1:1
        evt_type = EEG.event(i).type;
        if ischar(evt_type), evt_type = str2double(evt_type); end
        if isequal(evt_type, 65280)
            EEG.event(i) = [];
            removed_65280 = removed_65280 + 1;
        end
    end
    if removed_65280 > 0
        fprintf('[Worker %d] 🗑️ Removed %d trigger(s) 65280\n', workerID, removed_65280);
    end
    
    while i <= length(EEG.event)
        curr_type = EEG.event(i).type;
        if ischar(curr_type), curr_type = str2double(curr_type); end

        % Indice: 65344 OR 65348
        if ismember(curr_type, [65344, 65348])
            EEG.event(i).type = 'indice_raw';  % Will be refined later
            new_events = [new_events EEG.event(i)];
            
        % NoS: 65315 OR 65312
        elseif ismember(curr_type, [65315, 65312])
            EEG.event(i).type = 'NoS';
            new_events = [new_events EEG.event(i)];
            
        % Sons: Try pair 65296+65284, fallback to 65296 alone (with +50ms correction)
        elseif isequal(curr_type, 65296)
            if i < length(EEG.event)
                next_type = EEG.event(i+1).type;
                if ischar(next_type), next_type = str2double(next_type); end
                dt = (EEG.event(i+1).latency - EEG.event(i).latency) / EEG.srate * 1000;
                
                if dt < 200 && isequal(next_type, 65284)
                    % Pair found (triggerBox present) - no correction needed
                    EEG.event(i).type = 'Sons';
                    new_events = [new_events EEG.event(i)];
                    i = i + 1;  % Skip paired trigger
                else
                    % 65296 alone (no triggerBox) - apply +50ms correction
                    EEG.event(i).type = 'Sons';
                    EEG.event(i).latency = EEG.event(i).latency + (50 / 1000 * EEG.srate);
                    new_events = [new_events EEG.event(i)];
                    fprintf('[Worker %d] ⏱️ Sons trigger corrected +50ms (no triggerBox)\n', workerID);
                end
            else
                % 65296 alone (last event) - apply +50ms correction
                EEG.event(i).type = 'Sons';
                EEG.event(i).latency = EEG.event(i).latency + (50 / 1000 * EEG.srate);
                new_events = [new_events EEG.event(i)];
            end
            
        % Fleches: Handle multiple cases
        elseif ismember(curr_type, [65408, 65412])
            EEG.event(i).type = 'fleches';
            new_events = [new_events EEG.event(i)];
            
            % Skip trigger-box follow-up if present
            if i < length(EEG.event)
                next_type = EEG.event(i+1).type;
                if ischar(next_type), next_type = str2double(next_type); end
                dt = (EEG.event(i+1).latency - EEG.event(i).latency) / EEG.srate * 1000;
                
                if dt < 200 && ismember(next_type, [65288, 65416, 65292])
                    i = i + 1;
                end
            end
            
        elseif ismember(curr_type, [65288, 65416, 65292])
            % Check if already skipped
            isPaired = false;
            if i > 1
                prev_type = EEG.event(i-1).type;
                if ischar(prev_type), prev_type = str2double(prev_type); end
                dt_prev = (EEG.event(i).latency - EEG.event(i-1).latency) / EEG.srate * 1000;
                if ismember(prev_type, [65408, 65412]) && dt_prev < 200
                    isPaired = true;
                end
            end
            
            if ~isPaired
                EEG.event(i).type = 'fleches';
                new_events = [new_events EEG.event(i)];
            end
        end
        i = i + 1;
    end
end

function new_events = convert_triggers_aut24(EEG, workerID)
    % AUT24: Convert directly to fleche_X based on trigger code
    new_events = [];
    
    % Mapping: trigger code -> fleche type
    triggerMap = containers.Map(...
        {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}, ...
        {'fleche_1', 'fleche_2', 'fleche_3', 'fleche_4', 'fleche_5', 'fleche_6', ...
         'fleche_7', 'fleche_8', 'fleche_9', 'fleche_10', 'fleche_11', 'fleche_12', ...
         'Sons', 'NoS', 'RepJ'});
    
    for i = 1:length(EEG.event)
        curr_type = EEG.event(i).type;
        if ischar(curr_type), curr_type = str2double(curr_type); end
        
        if isKey(triggerMap, curr_type)
            EEG.event(i).type = triggerMap(curr_type);
            new_events = [new_events EEG.event(i)];
        end
    end
end

%% ========================================================================
%% SYNTHETIC TRIGGER CREATION
%% ========================================================================

function EEG = create_synthetic_triggers(EEG, set_type, offset_indice_ms, workerID)
    % Create synthetic triggers for indices only (sounds kept from original)
    % Indices: placed at -190ms from each fleche
    % Type determined by fleche_type: 1-4=absent, 5-8=correct, 9-12=incorrect
    
    fprintf('[Worker %d] 🔧 Creating synthetic indice triggers...\n', workerID);
    
    % === STEP 1: Remove existing indice_raw triggers ===
    eventsToKeep = [];
    removed_indice = 0;
    for i = 1:length(EEG.event)
        if strcmp(EEG.event(i).type, 'indice_raw')
            removed_indice = removed_indice + 1;
        else
            eventsToKeep = [eventsToKeep, i];
        end
    end
    
    if removed_indice > 0
        fprintf('[Worker %d] 🗑️ Removed %d indice_raw triggers\n', workerID, removed_indice);
        EEG.event = EEG.event(eventsToKeep);
    end
    
    % === STEP 2: Find all fleche events ===
    flecheIndices = [];
    flecheLatencies = [];
    flecheTypes = [];
    for i = 1:length(EEG.event)
        if startsWith(EEG.event(i).type, 'fleche_')
            flecheIndices = [flecheIndices, i];
            flecheLatencies = [flecheLatencies, EEG.event(i).latency];
            typeNum = str2double(EEG.event(i).type(8:end));
            flecheTypes = [flecheTypes, typeNum];
        end
    end
    
    if isempty(flecheLatencies)
        fprintf('[Worker %d] ⚠️ No fleche_X events found, cannot create indice triggers\n', workerID);
        return;
    end
    
    fprintf('[Worker %d] 📊 Found %d fleche events\n', workerID, length(flecheLatencies));
    
    % === STEP 3: Create synthetic indice triggers ===
    new_events = EEG.event;
    indice_count = struct('correct', 0, 'incorrect', 0, 'absent', 0);
    
    for i = 1:length(flecheLatencies)
        ftype = flecheTypes(i);
        fleche_latency = flecheLatencies(i);
        
        % Calculate indice latency (-190ms from fleche)
        indice_latency = fleche_latency - (offset_indice_ms / 1000 * EEG.srate);
        
        if indice_latency < 1
            continue;
        end
        
        % Determine indice type
        if ftype >= 1 && ftype <= 4
            indice_type = 'indice_absent';
            indice_count.absent = indice_count.absent + 1;
        elseif ftype >= 5 && ftype <= 8
            indice_type = 'indice_correct';
            indice_count.correct = indice_count.correct + 1;
        elseif ftype >= 9 && ftype <= 12
            indice_type = 'indice_incorrect';
            indice_count.incorrect = indice_count.incorrect + 1;
        else
            continue;
        end
        
        % Create indice event
        indice_event = EEG.event(flecheIndices(i));
        indice_event.type = indice_type;
        indice_event.latency = indice_latency;
        if isfield(indice_event, 'urevent')
            indice_event.urevent = [];
        end
        new_events = [new_events, indice_event];
    end
    
    % === STEP 4: Sort events by latency ===
    [~, sortIdx] = sort([new_events.latency]);
    EEG.event = new_events(sortIdx);
    EEG = eeg_checkset(EEG);
    
    fprintf('[Worker %d] ✅ Created %d indices: %d correct, %d incorrect, %d absent\n', ...
        workerID, indice_count.correct + indice_count.incorrect + indice_count.absent, ...
        indice_count.correct, indice_count.incorrect, indice_count.absent);
end

%% ========================================================================
%% EPOCH CREATION AND SAVING FUNCTIONS
%% ========================================================================

function [epochs_count, rejected_count] = create_and_save_epochs_fleche(EEG, epochWindow, output_folder, name, exposureType, workerID)
    % Create epochs on arrow triggers (fleche_1 to fleche_12)
    
    fprintf('[Worker %d] 📦 Creating FLECHE epochs...\n', workerID);
    
    % Define epoch types
    epochTypes = {};
    for i = 1:12
        epochTypes{end+1} = sprintf('fleche_%d', i);
    end
    
    % Check if any fleche triggers exist
    fleche_count = 0;
    for i = 1:length(EEG.event)
        if startsWith(EEG.event(i).type, 'fleche_')
            fleche_count = fleche_count + 1;
        end
    end
    
    if fleche_count == 0
        fprintf('[Worker %d] ⚠️ No fleche triggers found, skipping fleche epochs\n', workerID);
        epochs_count = 0;
        rejected_count = 0;
        return;
    end

    try
        EEG_epoched = pop_epoch(EEG, epochTypes, epochWindow, 'epochinfo', 'yes');
        EEG_epoched = eeg_checkset(EEG_epoched);
        EEG_epoched = pop_rmbase(EEG_epoched, [epochWindow(1)*1000, 0]);
        EEG_epoched = eeg_checkset(EEG_epoched);
        
        % Epoch rejection
        [EEG_epoched, rejectedEpochs] = reject_bad_epochs(EEG_epoched, workerID, 'fleche');
        rejected_count = length(rejectedEpochs);
        
        epochs_count = EEG_epoched.trials;
        
        % Save
        if ~isempty(exposureType)
            outname = sprintf('%s-%s_fleche.set', name, exposureType);
        else
            outname = sprintf('%s_fleche.set', name);
        end
        
        EEG_epoched.setname = [name '_fleche'];
        pop_saveset(EEG_epoched, 'filename', fullfile(output_folder, outname));
        
        fprintf('[Worker %d] ✅ Saved %d fleche epochs: %s\n', workerID, epochs_count, outname);
        
    catch ME
        fprintf('[Worker %d] ❌ Fleche epoching failed: %s\n', workerID, ME.message);
        epochs_count = 0;
        rejected_count = 0;
    end
end

function [epochs_count, rejected_count] = create_and_save_epochs_indice(EEG, epochWindow, output_folder, name, exposureType, workerID)
% Create epochs on visual cue triggers (indice_correct, indice_incorrect, indice_absent)
    
    fprintf('[Worker %d] 📦 Creating INDICE epochs...\n', workerID);
    
    epochTypes = {'indice_correct', 'indice_incorrect', 'indice_absent'};
    
    % Check if any indice triggers exist
    indice_count = 0;
    for i = 1:length(EEG.event)
        if startsWith(EEG.event(i).type, 'indice_') && ~strcmp(EEG.event(i).type, 'indice_raw')
            indice_count = indice_count + 1;
        end
    end
    
    if indice_count == 0
        fprintf('[Worker %d] ⚠️ No indice triggers found, skipping indice epochs\n', workerID);
        epochs_count = 0;
        rejected_count = 0;
        return;
    end
    
    try
        EEG_epoched = pop_epoch(EEG, epochTypes, epochWindow, 'epochinfo', 'yes');
        EEG_epoched = eeg_checkset(EEG_epoched);
        EEG_epoched = pop_rmbase(EEG_epoched, [epochWindow(1)*1000, 0]);
        EEG_epoched = eeg_checkset(EEG_epoched);
        
        % Epoch rejection
        [EEG_epoched, rejectedEpochs] = reject_bad_epochs(EEG_epoched, workerID, 'indice');
        rejected_count = length(rejectedEpochs);        
        epochs_count = EEG_epoched.trials;
        
        % Save
        if ~isempty(exposureType)
            outname = sprintf('%s-%s_indice.set', name, exposureType);
        else
            outname = sprintf('%s_indice.set', name);
        end
        
        EEG_epoched.setname = [name '_indice'];
        pop_saveset(EEG_epoched, 'filename', fullfile(output_folder, outname));
        
        fprintf('[Worker %d] ✅ Saved %d indice epochs: %s\n', workerID, epochs_count, outname);
        
    catch ME
        fprintf('[Worker %d] ❌ Indice epoching failed: %s\n', workerID, ME.message);
        epochs_count = 0;
        rejected_count = 0;
    end
end

function [epochs_count, rejected_count] = create_and_save_epochs_sonore(EEG, epochWindow, output_folder, name, exposureType, workerID)
% Create epochs on auditory cue triggers (Sons, NoS)
    
    fprintf('[Worker %d] 📦 Creating SONORE epochs...\n', workerID);
    
    epochTypes = {'Sons', 'NoS'};
    
    % Check if any son triggers exist
    son_count = 0;
    for i = 1:length(EEG.event)
        if ismember(EEG.event(i).type, epochTypes)
            son_count = son_count + 1;
        end
    end
    
    if son_count == 0
        fprintf('[Worker %d] ⚠️ No son triggers found, skipping sonore epochs\n', workerID);
        epochs_count = 0;
        rejected_count = 0;
        return;
    end
    
    try
        EEG_epoched = pop_epoch(EEG, epochTypes, epochWindow, 'epochinfo', 'yes');
        EEG_epoched = eeg_checkset(EEG_epoched);
        EEG_epoched = pop_rmbase(EEG_epoched, [epochWindow(1)*1000, 0]);
        EEG_epoched = eeg_checkset(EEG_epoched);
        
        % Epoch rejection
        [EEG_epoched, rejectedEpochs] = reject_bad_epochs(EEG_epoched, workerID, 'sonore');        
        rejected_count = length(rejectedEpochs);        
        epochs_count = EEG_epoched.trials;
        
        % Save
        if ~isempty(exposureType)
            outname = sprintf('%s-%s_sonore.set', name, exposureType);
        else
            outname = sprintf('%s_sonore.set', name);
        end
        
        EEG_epoched.setname = [name '_sonore'];
        pop_saveset(EEG_epoched, 'filename', fullfile(output_folder, outname));
        
        fprintf('[Worker %d] ✅ Saved %d sonore epochs: %s\n', workerID, epochs_count, outname);
        
    catch ME
        fprintf('[Worker %d] ❌ Sonore epoching failed: %s\n', workerID, ME.message);
        epochs_count = 0;
        rejected_count = 0;
    end
end

function [EEG, rejectedEpochs] = reject_bad_epochs(EEG, workerID, epochType)
    % Reject bad epochs with FULL TRACKING (from v9.4)
    
    rejectedEpochs = [];
    epochsBeforeRejection = EEG.trials;
    allRejectedEpochs = [];
    
    % Step 1: Reject epochs with early boundaries
    earlyBoundaryEpochs = [];
    for ep = 1:EEG.trials
        epochEvents = EEG.epoch(ep).eventtype;
        epochLatencies = EEG.epoch(ep).eventlatency;
        
        if ~iscell(epochEvents), epochEvents = {epochEvents}; end
        if ~iscell(epochLatencies), epochLatencies = {epochLatencies}; end
        
        for ev = 1:length(epochEvents)
            if strcmp(epochEvents{ev}, 'boundary')
                latMs = epochLatencies{ev} * 1000;
                if latMs < 500  % CORRECTED: 500ms threshold
                    earlyBoundaryEpochs = [earlyBoundaryEpochs, ep];
                    break;
                end
            end
        end
    end
    
    if ~isempty(earlyBoundaryEpochs)
        allRejectedEpochs = earlyBoundaryEpochs;
        EEG = pop_rejepoch(EEG, earlyBoundaryEpochs, 0);
        EEG = eeg_checkset(EEG);
        fprintf('[Worker %d] %s: %d boundary epochs rejected\n', workerID, epochType, length(earlyBoundaryEpochs));
    end
    
    % Step 2: Automatic rejection
    try
        EEG = pop_autorej(EEG, 'nogui', 'on', 'eegthresh', [-75 75]);
        EEG = eeg_checkset(EEG);
        
        autorejEpochs = [];
        if isfield(EEG,'reject') && isfield(EEG.reject,'rejglobal')
            mask = EEG.reject.rejglobal;
            if islogical(mask)
                autorejEpochs = find(mask==1);
            elseif isnumeric(mask)
                autorejEpochs = find(mask>0);
            end
        end
        
        if ~isempty(autorejEpochs)
            % Map to original indices BEFORE removing
            originalIndices = map_to_original_indices(autorejEpochs, allRejectedEpochs, epochsBeforeRejection);
            allRejectedEpochs = [allRejectedEpochs, originalIndices];
            EEG = pop_rejepoch(EEG, autorejEpochs, 0);
            EEG = eeg_checkset(EEG);
            fprintf('[Worker %d] %s: %d autorej epochs rejected\n', workerID, epochType, length(autorejEpochs));
        end
        
    catch
        % Fallback to jointprob
        try
            threshold = iif(epochsBeforeRejection < 200, 6, iif(epochsBeforeRejection < 250, 5, 4));
            
            [EEG, ~, ~, ~] = pop_jointprob(EEG, 1, 1:EEG.nbchan, threshold, threshold, 0, 0, 0, [], 0);
            
            jointprobEpochs = [];
            if isfield(EEG, 'reject') && isfield(EEG.reject, 'rejjp')
                mask = EEG.reject.rejjp;
                if islogical(mask)
                    jointprobEpochs = find(mask == 1);
                elseif isnumeric(mask)
                    jointprobEpochs = find(mask > 0);
                end
            end
            
            if ~isempty(jointprobEpochs)
                maxReject = floor(0.20 * EEG.trials);
                if length(jointprobEpochs) > maxReject
                    jointprobEpochs = jointprobEpochs(1:maxReject);
                end
                originalIndices = map_to_original_indices(jointprobEpochs, allRejectedEpochs, epochsBeforeRejection);
                allRejectedEpochs = [allRejectedEpochs, originalIndices];
                EEG = pop_rejepoch(EEG, jointprobEpochs, 0);
                EEG = eeg_checkset(EEG);
                fprintf('[Worker %d] %s: %d jointprob epochs rejected\n', workerID, epochType, length(jointprobEpochs));
            end
        catch
        end
    end
    
    rejectedEpochs = sort(unique(allRejectedEpochs));
    epochsAfterRejection = EEG.trials;
    rejectedCount = epochsBeforeRejection - epochsAfterRejection;
    
    fprintf('[Worker %d] %s: Total %d/%d epochs rejected\n', workerID, epochType, rejectedCount, epochsBeforeRejection);
end

%% ========================================================================
%% PRINT25 CSV ALIGNMENT (Modified to return csvData)
%% ========================================================================

function [EEG, alignmentQuality, csvData] = align_and_rename_fleches_print25_v10(EEG, csvFolder, participantId, passationNum, workerID)
    % Align fleches with CSV data and return csvData for synthetic trigger creation
    
    csvData = [];
    alignmentQuality = 0;
    
    % Load CSV file
    csvFilename = sprintf('%s_ANT%d_triggers.csv', participantId, passationNum);
    csvPath = fullfile(csvFolder, csvFilename);
    
    if ~exist(csvPath, 'file')
        error('CSV file not found: %s', csvPath);
    end
    
    fprintf('[Worker %d] 📂 Loading CSV: %s\n', workerID, csvFilename);
    csvData = readtable(csvPath, 'Delimiter', ';');
    
    % Extract fleches events
    flechesIdx = find(strcmp({EEG.event.type}, 'fleches'));
    
    if isempty(flechesIdx)
        error('No fleches triggers found in EEG');
    end
    
    expectedCount = height(csvData);
    actualCount = length(flechesIdx);
    
    fprintf('[Worker %d] 🔍 Found %d fleches in EEG, %d in CSV\n', workerID, actualCount, expectedCount);
    
    % Extract latencies and intervals
    eegLatencies = [EEG.event(flechesIdx).latency];
    eegIntervals = diff(eegLatencies) / EEG.srate;
    
    if ismember('interval_to_next_sec', csvData.Properties.VariableNames)
        csvIntervals = csvData.interval_to_next_sec(1:end-1);
        csvIntervals = csvIntervals(~isnan(csvIntervals));
    else
        error('Column interval_to_next_sec not found in CSV');
    end
    
    % Try alignment strategies
    percentDiff = abs(actualCount - expectedCount) / expectedCount;
    
    if percentDiff <= 0.05
        % Clean excess fleches if needed
        if actualCount > expectedCount
            [EEG, flechesIdx, eegIntervals] = clean_excess_fleches(EEG, flechesIdx, eegIntervals, expectedCount, workerID);
        end
        
        % Try full alignment
        [bestOffset, alignmentQuality] = find_best_alignment(eegIntervals, csvIntervals);
        
        if alignmentQuality >= 0.85
            EEG = apply_alignment(EEG, flechesIdx, csvData, bestOffset, workerID);
            return;
        end
    end
    
    % Segment-based alignment
    fprintf('[Worker %d] 🔀 Using SEGMENT-BASED alignment\n', workerID);
    
    longGaps = find(eegIntervals > 8);
    
    if isempty(longGaps)
        segments = {1:length(flechesIdx)};
        csvSegments = {1:height(csvData)};
    else
        segments = {};
        csvSegments = {};
        csvBlockSize = round(height(csvData) / (length(longGaps) + 1));
        
        segments{1} = 1:longGaps(1);
        csvSegments{1} = 1:csvBlockSize;
        
        for i = 1:length(longGaps)-1
            segments{end+1} = longGaps(i)+1:longGaps(i+1);
            csvSegments{end+1} = csvBlockSize*i+1:csvBlockSize*(i+1);
        end
        
        segments{end+1} = longGaps(end)+1:length(flechesIdx);
        csvSegments{end+1} = csvBlockSize*length(longGaps)+1:height(csvData);
    end
    
    % Align each segment
    totalAligned = 0;
    segmentQualities = [];
    
    for seg = 1:length(segments)
        eegSeg = segments{seg};
        csvSeg = csvSegments{seg};
        
        % Clean excess if needed
        if length(eegSeg) > length(csvSeg)
            numExcess = length(eegSeg) - length(csvSeg);
            segStartIdx = eegSeg(1);
            segEndIdx = min(eegSeg(end)-1, length(eegIntervals));
            
            if segEndIdx >= segStartIdx
                segIntervals = eegIntervals(segStartIdx:segEndIdx);
                [~, sortIdx] = sort(segIntervals);
                
                toRemove = [];
                for k = 1:min(numExcess, length(sortIdx))
                    relIdx = sortIdx(k) + 1;
                    if relIdx <= length(eegSeg)
                        toRemove = [toRemove, relIdx];
                    end
                end
                
                for idx = toRemove
                    if idx <= length(eegSeg)
                        eegIdx = flechesIdx(eegSeg(idx));
                        EEG.event(eegIdx).type = 'fleches_artifact';
                    end
                end
                
                eegSeg(toRemove) = [];
                segments{seg} = eegSeg;
            end
        end
        
        csvSegAdjusted = csvSeg(1):min(csvSeg(1) + length(eegSeg) - 1, csvSeg(end));
        
        % Apply alignment
        for i = 1:length(eegSeg)
            if i <= length(csvSegAdjusted)
                eegIdx = flechesIdx(eegSeg(i));
                csvIdx = csvSegAdjusted(i);
                
                if csvIdx >= 1 && csvIdx <= height(csvData)
                    flecheType = csvData.fleche_type(csvIdx);
                    EEG.event(eegIdx).type = sprintf('fleche_%d', flecheType);
                    totalAligned = totalAligned + 1;
                end
            end
        end
    end
    
    % Mark unaligned fleches as artifacts
    for i = 1:length(flechesIdx)
        eegIdx = flechesIdx(i);
        if strcmp(EEG.event(eegIdx).type, 'fleches')
            EEG.event(eegIdx).type = 'fleches_artifact';
        end
    end
    
    alignmentQuality = totalAligned / expectedCount;
    fprintf('[Worker %d] 📊 Aligned %d/%d fleches (%.1f%%)\n', workerID, totalAligned, expectedCount, alignmentQuality*100);
    
    EEG = eeg_checkset(EEG);
end

function [bestOffset, bestCorr] = find_best_alignment(eegIntervals, csvIntervals)
    minLength = min(length(eegIntervals), length(csvIntervals));
    maxOffset = abs(length(eegIntervals) - length(csvIntervals));
    
    bestCorr = -Inf;
    bestOffset = 0;
    
    for offset = -min(maxOffset, 20):min(maxOffset, 20)
        if offset >= 0
            csvSeg = csvIntervals(offset+1:min(offset+minLength, end));
            eegSeg = eegIntervals(1:length(csvSeg));
        else
            eegSeg = eegIntervals(abs(offset)+1:min(abs(offset)+minLength, end));
            csvSeg = csvIntervals(1:length(eegSeg));
        end
        
        if length(eegSeg) > 10 && length(csvSeg) > 10 && length(eegSeg) == length(csvSeg)
            eegNorm = eegSeg / mean(eegSeg);
            csvNorm = csvSeg / mean(csvSeg);
            
            corrMatrix = corrcoef(eegNorm, csvNorm);
            corr = corrMatrix(1, 2);
            
            if corr > bestCorr
                bestCorr = corr;
                bestOffset = offset;
            end
        end
    end
    
    bestCorr = bestCorr^2;
end

function EEG = apply_alignment(EEG, flechesIdx, csvData, bestOffset, workerID)
    numAligned = min(length(flechesIdx), height(csvData) - bestOffset);
    
    for i = 1:numAligned
        eegIdx = flechesIdx(i);
        csvIdx = bestOffset + i;
        
        if csvIdx >= 1 && csvIdx <= height(csvData)
            flecheType = csvData.fleche_type(csvIdx);
            EEG.event(eegIdx).type = sprintf('fleche_%d', flecheType);
        end
    end
    
    fprintf('[Worker %d] 📊 Applied standard alignment (offset: %d)\n', workerID, bestOffset);
end

function [EEG, flechesIdx, eegIntervals] = clean_excess_fleches(EEG, flechesIdx, eegIntervals, expectedCount, workerID)
    actualCount = length(flechesIdx);
    
    fprintf('[Worker %d] ⚠️ Found %d extra fleches, cleaning...\n', workerID, actualCount - expectedCount);
    
    toRemove = [];
    shortIntervals = find(eegIntervals < 1.0);
    
    for i = 1:length(shortIntervals)
        idx = shortIntervals(i);
        toRemove = [toRemove, idx+1];
    end
    
    numToRemove = actualCount - expectedCount;
    if length(toRemove) > numToRemove
        [~, sortIdx] = sort(eegIntervals(min(toRemove, length(eegIntervals))));
        toRemove = toRemove(sortIdx(1:numToRemove));
    end
    
    if ~isempty(toRemove)
        toRemove = unique(toRemove);
        toRemove = toRemove(toRemove <= length(flechesIdx));
        
        flechesIdxToRemove = flechesIdx(toRemove);
        for idx = flechesIdxToRemove
            EEG.event(idx).type = 'fleches_artifact';
        end
        
        flechesIdx = find(strcmp({EEG.event.type}, 'fleches'));
        eegLatencies = [EEG.event(flechesIdx).latency];
        eegIntervals = diff(eegLatencies) / EEG.srate;
        
        fprintf('[Worker %d] 🧹 Removed %d parasites\n', workerID, length(toRemove));
    end
end

%% ========================================================================
%% TRIGGER CLEANING FUNCTIONS
%% ========================================================================

function [EEG, invalidTriggers] = clean_triggers_two_pass(EEG, set_type, timing1, timing2, workerID)
    % Two-pass trigger cleaning
    
    triggersInitial = length(EEG.event);
    
    % Define valid types
    if strcmp(set_type, 'print25')
        validTypes = {'indice_raw', 'Sons', 'NoS', 'fleches', 'boundary'};
        for i = 1:12
            validTypes{end+1} = sprintf('fleche_%d', i);
        end
    else
        validTypes = {'Sons', 'NoS', 'RepJ', 'boundary'};
        for i = 1:12
            validTypes{end+1} = sprintf('fleche_%d', i);
        end
    end
    
    % Pass 1: Remove noise bursts
    EEG.event = remove_noise_burst_triggers_smart(EEG.event, EEG.srate, validTypes);
    EEG = eeg_checkset(EEG);
    
    noiseBurstRemoved = triggersInitial - length(EEG.event);
    if noiseBurstRemoved > 0
        fprintf('[Worker %d] Pass 1: Removed %d noise-burst triggers\n', workerID, noiseBurstRemoved);
    end
    
    % Pass 2: Validate sequences (lenient for print25)
    [validatedEvents, invalidTriggers] = validate_trigger_sequences_lenient(EEG.event, EEG.srate, timing1, timing2, true, set_type);
    
    % Count fleches before and after
    flechesBefore = sum(cellfun(@(x) strcmp(x, 'fleches') || startsWith(x, 'fleche_'), {EEG.event.type}));
    flechesAfter = sum(cellfun(@(x) strcmp(x, 'fleches') || startsWith(x, 'fleche_'), {validatedEvents.type}));
    
    if flechesAfter >= 20 && (flechesAfter / max(1, flechesBefore)) >= 0.3
        EEG.event = validatedEvents;
        EEG = eeg_checkset(EEG);
        fprintf('[Worker %d] Pass 2: Validated sequences, kept %d/%d fleches\n', workerID, flechesAfter, flechesBefore);
    else
        fprintf('[Worker %d] ⚠️ Validation too aggressive, SKIPPING Pass 2\n', workerID);
        invalidTriggers = 0;
    end
end

function cleanEvents = remove_noise_burst_triggers_smart(events, srate, validTypes)
    cleanEvents = [];
    i = 1;
    
    while i <= length(events)
        evt = events(i);
        
        if strcmp(evt.type, 'boundary')
            cleanEvents = [cleanEvents, evt];
            i = i + 1;
            continue;
        end
        
        if ~ismember(evt.type, validTypes) && ~startsWith(evt.type, 'fleche_')
            i = i + 1;
            continue;
        end
        
        if i < length(events)
            nextLat = events(i+1).latency;
            dtMs = (nextLat - evt.latency) / srate * 1000;
            
            if dtMs < 10 && ~strcmp(events(i+1).type, 'boundary')
                i = i + 2;
                continue;
            end
            
            if i+2 <= length(events) && dtMs < 50
                thirdLat = events(i+2).latency;
                dt2Ms = (thirdLat - evt.latency) / srate * 1000;
                if dt2Ms < 50 && ~strcmp(events(i+2).type, 'boundary')
                    i = i + 3;
                    continue;
                end
            end
        end
        
        cleanEvents = [cleanEvents, evt];
        i = i + 1;
    end
end

function [validatedEvents, invalidCount] = validate_trigger_sequences_lenient(events, srate, timing1, timing2, allowPartialTriplets, setType)
    validatedEvents = [];
    invalidCount = 0;
    
    timing1Lenient = [timing1(1) - 50, timing1(2) + 50];
    timing2Lenient = [timing2(1) - 50, timing2(2) + 200];
    
    if strcmp(setType, 'print25')
        validatedEvents = events;
        invalidCount = 0;
    else
        i = 1;
        while i <= length(events)
            if ismember(events(i).type, {'Sons', 'NoS'})
                if i+1 <= length(events) && (strcmp(events(i+1).type, 'fleches') || startsWith(events(i+1).type, 'fleche_'))
                    dt1 = (events(i+1).latency - events(i).latency) / srate * 1000;
                    
                    isValidPair = (dt1 >= timing1(1) && dt1 <= timing1(2)) || ...
                                  (dt1 >= timing1Lenient(1) && dt1 <= timing1Lenient(2));
                    
                    if isValidPair
                        if i+2 <= length(events) && strcmp(events(i+2).type, 'RepJ')
                            dt2 = (events(i+2).latency - events(i+1).latency) / srate * 1000;
                            
                            if (dt2 >= timing2(1) && dt2 <= timing2(2)) || ...
                               (dt2 >= timing2Lenient(1) && dt2 <= timing2Lenient(2))
                                validatedEvents = [validatedEvents, events(i), events(i+1), events(i+2)];
                                i = i + 3;
                            else
                                validatedEvents = [validatedEvents, events(i), events(i+1)];
                                invalidCount = invalidCount + 1;
                                i = i + 3;
                            end
                        else
                            validatedEvents = [validatedEvents, events(i), events(i+1)];
                            i = i + 2;
                        end
                    else
                        invalidCount = invalidCount + 2;
                        i = i + 2;
                    end
                else
                    invalidCount = invalidCount + 1;
                    i = i + 1;
                end
            else
                validatedEvents = [validatedEvents, events(i)];
                i = i + 1;
            end
        end
    end
end

%% ========================================================================
%% PREPROCESSING HELPER FUNCTIONS
%% ========================================================================

function EEG = prepare_channels(EEG)
    % Channel correction and selection
    for i = 1:length(EEG.chanlocs)
        if strcmpi(EEG.chanlocs(i).labels, 'Ma1'), EEG.chanlocs(i).labels = 'TP9'; end
        if strcmpi(EEG.chanlocs(i).labels, 'Ma2'), EEG.chanlocs(i).labels = 'TP10'; end
    end
    
    selectedChans = {'Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5', ...
                     'P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8', ...
                     'CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4', ...
                     'Fp2','Fz','Cz','TP9','TP10','EOG','ref'};
    EEG = pop_select(EEG, 'channel', selectedChans);
    
    try
        eeglab_path = fileparts(which('eeglab.m'));
        standard_locs = find_standard_locs(eeglab_path);
        if ~isempty(standard_locs)
            EEG = pop_chanedit(EEG, 'lookup', standard_locs);
        end
    catch
    end
end

function [EEG, extremeNoiseSegments] = detect_and_remove_extreme_noise(EEG, thresholdUv)
    extremeNoiseSegments = 0;
    minSegmentSec = 0.1;
    minSegmentSamples = round(minSegmentSec * EEG.srate);
    
    chanLabels = {EEG.chanlocs.labels};
    excludeChans = {'TP9', 'TP10', 'EOG', 'ref'};
    scalpChans = true(1, numel(chanLabels));
    for exc = 1:numel(excludeChans)
        scalpChans = scalpChans & ~strcmpi(chanLabels, excludeChans{exc});
    end
    scalpIndices = find(scalpChans);
    
    if isempty(scalpIndices)
        return;
    end
    
    X = double(EEG.data(scalpIndices, :));
    extremeMask = any(abs(X) > thresholdUv, 1);
    
    diff_mask = [0, diff(extremeMask)];
    starts = find(diff_mask == 1);
    ends = find(diff_mask == -1);
    
    if extremeMask(1) == 1
        starts = [1, starts];
    end
    if extremeMask(end) == 1
        ends = [ends, length(extremeMask)];
    end
    
    rej = [];
    for si = 1:length(starts)
        segLength = ends(si) - starts(si) + 1;
        if segLength >= minSegmentSamples
            rej(end+1, :) = [starts(si), ends(si)];
            extremeNoiseSegments = extremeNoiseSegments + 1;
        end
    end
    
    if ~isempty(rej)
        EEG = eeg_eegrej(EEG, rej);
        EEG = eeg_checkset(EEG);
    end
end

function EEG = remove_silent_segments(EEG, keepPreSec, keepPostSec)
    % keepPreSec/keepPostSec optionnels : défaut = valeurs Article 1 (0.2 / 1.5 s)
    if nargin < 2 || isempty(keepPreSec),  keepPreSec  = 0.2; end
    if nargin < 3 || isempty(keepPostSec), keepPostSec = 1.5; end
    try
        triggerTypes = {'fleches','NoS','Sons','indice_raw','RepJ'};
        for i = 1:12
            triggerTypes{end+1} = sprintf('fleche_%d', i);
        end

        evLat = [];
        for ei = 1:numel(EEG.event)
            et = EEG.event(ei).type;
            if ischar(et) && (any(strcmpi(et, triggerTypes)) || startsWith(et, 'fleche_'))
                evLat(end+1) = round(EEG.event(ei).latency);
            end
        end
        evLat = unique(evLat);

        keepPre = max(1, round(keepPreSec * EEG.srate));
        keepPost = max(1, round(keepPostSec * EEG.srate));
        rej = compute_rejection_segments(evLat, EEG.pnts, EEG.srate, keepPre, keepPost);
        
        if ~isempty(rej)
            EEG = eeg_eegrej(EEG, rej);
            EEG = eeg_checkset(EEG);
        end
    catch
    end
end

function rej = compute_rejection_segments(evLat, pnts, srate, keepPre, keepPost)
    rej = [];
    leadTailSilenceSec = 2;
    pauseMinSec = 5;
    if isempty(evLat), return; end
    evLat = sort(evLat);
    
    if evLat(1) > leadTailSilenceSec * srate
        startCutEnd = max(1, evLat(1) - keepPre);
        rej(end+1, :) = [1, startCutEnd];
    end
    
    for k = 1:numel(evLat)-1
        gap = evLat(k+1) - evLat(k);
        if gap >= pauseMinSec * srate
            a = min(pnts, max(1, evLat(k) + keepPost));
            b = min(pnts, max(1, evLat(k+1) - keepPre));
            if b > a
                rej(end+1, :) = [a, b];
            end
        end
    end
    
    if (pnts - evLat(end)) > leadTailSilenceSec * srate
        endCutStart = min(pnts, evLat(end) + keepPost);
        rej(end+1, :) = [endCutStart, pnts];
    end
end

function EEG = asr_limited_rejection(EEG, extremeAmpThresholdUv, extremeAmpPropThreshold, asrMaxBoundaries, workerID)
    try
        if exist('pop_clean_rawdata','file')
            X = double(EEG.data);
            ampMask = any(abs(X) > extremeAmpThresholdUv, 1);
            ampProp = sum(ampMask) / size(X,2);
            
            if ampProp >= extremeAmpPropThreshold
                EEG_before = EEG;
                boundaries_before = count_boundaries(EEG);
                
                EEG_tmp = pop_clean_rawdata(EEG, ...
                    'FlatlineCriterion', 'off', 'Highpass', 'off', ...
                    'ChannelCriterion', 'off', 'LineNoiseCriterion', 'off', ...
                    'BurstCriterion', 25, 'BurstRejection', 'on', 'WindowCriterion', 0.20);
                EEG_tmp = eeg_checkset(EEG_tmp);
                
                boundaries_after = count_boundaries(EEG_tmp);
                added = boundaries_after - boundaries_before;
                
                if added > asrMaxBoundaries
                    fprintf('[Worker %d] ⚠️ ASR rejected: %d boundaries > limit %d\n', workerID, added, asrMaxBoundaries);
                    EEG = EEG_before;
                else
                    EEG = EEG_tmp;
                end
            end
        end
    catch
    end
    EEG = eeg_checkset(EEG);
end

function count = count_boundaries(EEG)
    count = 0;
    if isfield(EEG,'event') && ~isempty(EEG.event)
        try
            count = sum(strcmpi({EEG.event.type}, 'boundary'));
        catch
        end
    end
end

function EEG = apply_reference(EEG)
    try
        chanLabels = {EEG.chanlocs.labels};
        refIdx = find(strcmpi(chanLabels, 'ref'), 1);
        if ~isempty(refIdx)
            EEG = pop_reref(EEG, refIdx, 'keepref', 'on');
            EEG = eeg_checkset(EEG);
        end
    catch
    end
end

function [EEG, comps_to_reject] = reject_eye_components(EEG, iclabelEyeThreshold, iclabelBrainThreshold, ...
    eogCorrThreshold, eogSafetyThreshold, eyeSafetyThreshold, maxEyeICs, workerID)
    
    comps_to_reject = [];
    
    if isempty(EEG.icaweights)
        return;
    end
    
    try
        if exist('iclabel','file')
            EEG = iclabel(EEG);
            
            if isfield(EEG,'etc') && isfield(EEG.etc,'ic_classification')
                probs = EEG.etc.ic_classification.ICLabel.classifications;
                eyeProbs = probs(:,3);
                brainProbs = probs(:,1);
                [~, order] = sort(eyeProbs, 'descend');
                
                % IC1: MANDATORY
                comps_to_reject = order(1);
                fprintf('[Worker %d] 🔍 IC%d: %.1f%% Eye → OBLIGATOIRE ✅\n', workerID, comps_to_reject(1), eyeProbs(comps_to_reject(1))*100);
                
                % IC2-3: CONDITIONAL
                eogIdx = find(strcmpi({EEG.chanlocs.labels}, 'EOG'), 1);
                if ~isfield(EEG,'icaact') || isempty(EEG.icaact)
                    EEG = eeg_checkset(EEG, 'ica');
                end
                
                hasEOG = ~isempty(eogIdx);
                
                for candidateIdx = 2:min(3, length(order))
                    ic = order(candidateIdx);
                    
                    criterion1 = eyeProbs(ic) >= iclabelEyeThreshold;
                    criterion2 = brainProbs(ic) < iclabelBrainThreshold;
                    
                    criterion3 = false;
                    eogCorr = NaN;
                    if hasEOG
                        icSig = double(squeeze(EEG.icaact(ic,:,:))); icSig = icSig(:);
                        eogSig = double(squeeze(EEG.data(eogIdx,:,:))); eogSig = eogSig(:);
                        if numel(icSig) == numel(eogSig) && numel(icSig) > 10
                            c = corrcoef(icSig, eogSig);
                            eogCorr = abs(c(1,2));
                            criterion3 = eogCorr >= eogCorrThreshold;
                        end
                    end
                    
                    safetyNet = false;
                    if hasEOG && ~isnan(eogCorr)
                        if eogCorr >= eogSafetyThreshold && eyeProbs(ic) >= eyeSafetyThreshold
                            safetyNet = true;
                        end
                    end
                    
                    numCriteriaMet = sum([criterion1, criterion2, criterion3]);
                    shouldReject = (numCriteriaMet >= 2) || (~hasEOG && criterion1 && criterion2) || safetyNet;
                    
                    if shouldReject
                        comps_to_reject(end+1) = ic;
                        fprintf('[Worker %d] ✅ IC%d: Eye=%.1f%%, Brain=%.1f%% → REJETÉ\n', workerID, ic, eyeProbs(ic)*100, brainProbs(ic)*100);
                    end
                end
                
                fprintf('[Worker %d] 📊 Total composantes rejetées: %d\n', workerID, length(comps_to_reject));
            end
        end
    catch ME
        fprintf('[Worker %d] ⚠️ ICLabel échoué: %s\n', workerID, ME.message);
    end
    
    if ~isempty(comps_to_reject)
        EEG = pop_subcomp(EEG, comps_to_reject, 0);
        EEG = eeg_checkset(EEG);
    end
end

function [EEG, interpolatedChans] = interpolate_bad_channels(EEG, adjacencyMap, maxInterpolations)
    interpolatedChans = {};
    
    try
        chanLabels = {EEG.chanlocs.labels};
        excludeFromInterp = {'TP9', 'TP10', 'EOG', 'ref'};
        scalpChans = true(1, numel(chanLabels));
        for exc = 1:numel(excludeFromInterp)
            scalpChans = scalpChans & ~strcmpi(chanLabels, excludeFromInterp{exc});
        end
        scalpIndices = find(scalpChans);
        
        if ~isempty(scalpIndices)
            data2d = double(EEG.data(scalpIndices, :));
            chVar = var(data2d, 0, 2);
            medianVar = median(chVar);
            varRatio = chVar / medianVar;
            candidates = scalpIndices(varRatio > 3 | varRatio < 0.1);
            candLabels = chanLabels(candidates);
            
            selected = {};
            for ci = 1:numel(candLabels)
                c = candLabels{ci};
                if numel(selected) >= maxInterpolations, break; end
                isAdjacent = false;
                if isKey(adjacencyMap, c)
                    for sj = 1:numel(selected)
                        if isKey(adjacencyMap, selected{sj})
                            if any(strcmpi(adjacencyMap(c), selected{sj})) || ...
                               any(strcmpi(adjacencyMap(selected{sj}), c))
                                isAdjacent = true; break;
                            end
                        end
                    end
                end
                if ~isAdjacent
                    selected{end+1} = c;
                end
            end
            
            if ~isempty(selected)
                interpolatedChans = selected;
                EEG = pop_interp(EEG, find(ismember(chanLabels, interpolatedChans)), 'spherical');
                EEG = eeg_checkset(EEG);
            end
        end
    catch
    end
end

function EEG = remove_reference_channels(EEG)
    try
        chansToRemove = {'TP9', 'TP10', 'EOG', 'ref'};
        labels = {EEG.chanlocs.labels};
        removeIdx = [];
        for ch = 1:numel(chansToRemove)
            idx = find(strcmpi(labels, chansToRemove{ch}));
            if ~isempty(idx)
                removeIdx = [removeIdx, idx];
            end
        end
        if ~isempty(removeIdx)
            EEG = pop_select(EEG, 'nochannel', removeIdx);
            EEG = eeg_checkset(EEG);
        end
    catch
    end
end

%% ========================================================================
%% UTILITY FUNCTIONS
%% ========================================================================

function ensure_eeglab()
    % Initialise EEGLAB UNE SEULE FOIS par processus worker (persistent),
    % au lieu d'une fois par fichier : supprime les appels réseau répétés
    % ('Retrieving plugin versions from server...') et accélère le run.
    persistent inited
    if isempty(inited)
        if ~exist('eeglab', 'file')
            addpath(fileparts(which('eeglab.m')));
        end
        eeglab nogui;
        try, pop_editoptions('option_checkversion', 0); catch, end
        inited = true;
    end
end

function worker_id = get_worker_id()
    try
        t = getCurrentTask();
        worker_id = t.ID;
    catch
        worker_id = 0;
    end
end

function standard_locs = find_standard_locs(eeglab_path)
    standard_locs = '';
    cand1 = fullfile(eeglab_path, 'sample_locs', 'standard-10-5-cap385.elp');
    if exist(cand1, 'file')
        standard_locs = cand1;
        return;
    end
    plugdir = fullfile(eeglab_path, 'plugins');
    if exist(plugdir, 'dir')
        dlist = dir(fullfile(plugdir, 'dipfit*'));
        for di = 1:numel(dlist)
            cand2 = fullfile(dlist(di).folder, dlist(di).name, 'standard_BESA', 'standard-10-5-cap385.elp');
            if exist(cand2, 'file')
                standard_locs = cand2;
                return;
            end
        end
    end
    w = which('standard-10-5-cap385.elp');
    if ~isempty(w)
        standard_locs = w;
    end
end

function exposureType = get_exposure_type(participantId, attemptNumber, exposureMap)
    exposureType = '';
    
    if ~isempty(participantId) && attemptNumber > 1 && ~isempty(exposureMap)
        if isfield(exposureMap, participantId)
            attemptField = sprintf('attempt%d', attemptNumber);
            if isfield(exposureMap.(participantId), attemptField)
                exposureType = exposureMap.(participantId).(attemptField);
            end
        end
    end
end

function [participantId, attemptNumber] = parse_filename_matlab(filename)
    participantId = [];
    attemptNumber = 1;
    
    [~, nameWithoutExt, ~] = fileparts(filename);
    
    participantMatch = regexp(nameWithoutExt, '^(P\d+)', 'match');
    if isempty(participantMatch)
        return;
    end
    participantId = participantMatch{1};
    
    antMatch = regexp(nameWithoutExt, 'ANT[-_]?(\d)', 'tokens');
    if ~isempty(antMatch)
        attemptNumber = str2double(antMatch{1}{1});
    else
        antPos = strfind(nameWithoutExt, 'ANT');
        if ~isempty(antPos)
            afterAnt = nameWithoutExt(antPos(1)+3:end);
            digits = regexp(afterAnt, '\d', 'match');
            if ~isempty(digits)
                attemptNumber = str2double(digits{1});
            end
        end
    end
end

function exposureMap = load_exposure_mapping(csvPath)
    exposureMap = struct();
    
    if ~exist(csvPath, 'file')
        warning('Base.csv non trouvé: %s', csvPath);
        return;
    end
    
    try
        try
            data = readtable(csvPath, 'Delimiter', ',', 'ReadVariableNames', true);
            if width(data) < 3
                data = readtable(csvPath, 'Delimiter', ';', 'ReadVariableNames', true);
            end
        catch
            data = readtable(csvPath, 'Delimiter', ';', 'ReadVariableNames', true);
        end
        
        if width(data) < 3
            error('Base.csv doit avoir au moins 3 colonnes');
        end
        
        for i = 1:height(data)
            participant = data{i, 1};
            if iscell(participant), participant = participant{1}; end
            if ~ischar(participant) && ~isstring(participant)
                participant = num2str(participant);
            end
            participant = char(participant);
            
            if strcmpi(participant, 'Participant'), continue; end
            
            if ~isfield(exposureMap, participant)
                exposureMap.(participant) = struct();
            end
            
            expo2 = data{i, 2};
            if iscell(expo2), expo2 = expo2{1}; end
            if ~ischar(expo2) && ~isstring(expo2), expo2 = num2str(expo2); end
            expo2 = char(expo2);
            
            if strcmpi(expo2, 'C')
                exposureMap.(participant).attempt2 = 'close';
            elseif strcmpi(expo2, 'A')
                exposureMap.(participant).attempt2 = 'away';
            end
            
            if width(data) >= 3
                expo3 = data{i, 3};
                if iscell(expo3), expo3 = expo3{1}; end
                if ~ischar(expo3) && ~isstring(expo3), expo3 = num2str(expo3); end
                expo3 = char(expo3);
                
                if strcmpi(expo3, 'C')
                    exposureMap.(participant).attempt3 = 'close';
                elseif strcmpi(expo3, 'A')
                    exposureMap.(participant).attempt3 = 'away';
                end
            end
        end
        
        fprintf('✅ Base.csv chargé: %d participants\n', length(fieldnames(exposureMap)));
        
    catch ME
        warning('Erreur lecture Base.csv: %s', ME.message);
    end
end

function write_log_entry_v10(logPath, filename, invalidTriggers, interpolatedChans, ...
    comps_to_reject, rej_fleche, rej_indice, rej_sonore, ...
    final_fleche, final_indice, final_sonore, ...
    cutSegmentsSec, asrBoundariesAdded, extremeNoiseSegments)
    
    max_attempts = 10;
    attempt = 0;
    success = false;
    
    interpStr = strjoin(interpolatedChans, '|');
    icStr = strjoin(arrayfun(@num2str, comps_to_reject, 'UniformOutput', false), '|');
    cutsStr = strjoin(cutSegmentsSec, '|');
    
    logLine = sprintf('%s,%d,%s,%s,%d,%d,%d,%d,%d,%d,%s,%d,%d\n', ...
        filename, invalidTriggers, interpStr, icStr, ...
        rej_fleche, rej_indice, rej_sonore, ...
        final_fleche, final_indice, final_sonore, ...
        cutsStr, asrBoundariesAdded, extremeNoiseSegments);
    
    while ~success && attempt < max_attempts
        try
            fid = fopen(logPath, 'a');
            if fid ~= -1
                fprintf(fid, '%s', logLine);
                fclose(fid);
                success = true;
            else
                attempt = attempt + 1;
                pause(0.1 * rand());
            end
        catch
            attempt = attempt + 1;
            pause(0.1 * rand());
        end
    end
end

function originalIndices = map_to_original_indices(currentIndices, alreadyRejected, totalOriginal)
    % Maps current epoch indices back to original numbering
    originalIndices = [];
    currentToOriginal = 1:totalOriginal;
    if ~isempty(alreadyRejected)
        currentToOriginal(alreadyRejected) = [];
    end
    for i = 1:length(currentIndices)
        if currentIndices(i) <= length(currentToOriginal)
            originalIndices(end+1) = currentToOriginal(currentIndices(i));
        end
    end
end

function result = iif(condition, trueVal, falseVal)
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end

function EEG = correct_sound_triggers_aut24(EEG, workerID)
    % Apply +50ms temporal correction to Sons/NoS triggers for aut_24 campaign
    
    correctionMs = +50;
    correctionSamples = correctionMs / 1000 * EEG.srate;
    
    corrected_count = 0;
    for i = 1:length(EEG.event)
        if strcmp(EEG.event(i).type, 'Sons') || strcmp(EEG.event(i).type, 'NoS')
            EEG.event(i).latency = EEG.event(i).latency + correctionSamples;
            corrected_count = corrected_count + 1;
        end
    end
    
    if corrected_count > 0
        fprintf('[Worker %d] ⏱️ Applied +50ms correction to %d sound triggers (aut_24)\n', workerID, corrected_count);
        EEG = eeg_checkset(EEG);
    end
end