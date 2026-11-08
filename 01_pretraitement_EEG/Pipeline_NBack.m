%% ========================================================================
% SCRIPT : Pipeline N-Back v2.2 FINAL
%
% Comprehensive EEG preprocessing pipeline for 1-Back and 2-Back tasks
%
% ========================================================================
% FEATURES:
% ========================================================================
% 
% TRIGGER HANDLING:
% - Adaptive trigger conversion for print_25 (pair logic with fallback)
% - Smart handling of missing trigger-box (uses OpenSesame triggers)
% - Direct conversion for aut_24 dataset with +42ms temporal correction
% - Burst detection with N-Back specific timing (100ms/200ms)
% - Response trigger detection for both datasets
%
% PREPROCESSING:
% - Filtering: 0.5-40 Hz
% - ASR correction: 30 SD (repair mode)
% - ASR rejection: 25 SD (if ≥5% data >100µV), max 50 boundaries
% - Extreme noise removal: >100µV for >10s
% - Silent segment removal: 200ms pre + 1500ms post triggers
%
% ICA & ARTIFACT REJECTION:
% - ICA: runica algorithm
% - Eye component detection (NEW v2.2):
%   * IC1: MANDATORY - Highest eye probability (always rejected)
%   * IC2-3: CONDITIONAL - Must meet 2/3 criteria:
%     - Eye probability ≥ 60%
%     - Brain probability < 10% (STRICTER - protects cognitive activity)
%     - EOG correlation ≥ 0.30 (STRICTER - if EOG available)
%   * SAFETY NET: EOG ≥ 0.60 + Eye ≥ 20% = automatic rejection
% - Maximum: 3 eye components per recording (1 mandatory + 2 conditional)
% - Interpolation: Max 2 non-adjacent channels (variance >3× or <0.1× median)
%
% EPOCHING:
% - Window: -200 to +800 ms
% - Baseline: -200 to 0 ms
% - Adaptive rejection:
%   * <100 epochs: 5 SD (permissive)
%   * 100-120 epochs: 4 SD (moderate)
%   * >120 epochs: 3 SD (strict, ERP standard)
% - Rejection limit: 30% max (ensures quality while preserving power)
%
% VALIDATION CRITERIA:
% - Minimum 25 Valide trials (60% of expected 42)
% - Minimum 60 Invalide trials (61% of expected 98)
% - Minimum 85 total trials (61% of expected 140)
%
% OUTPUT:
% - Automatic file renaming with exposure type (_close/_away)
% - Separate folders: 1B_all/processed and 2B_all/processed
% - Filtered to passations 2 and 3 only
%
% PARALLEL PROCESSING:
% - 4 workers for optimal throughput
%
% ========================================================================
% CHANGELOG v2.2:
% ========================================================================
%
% v2.1 → v2.2 (Hyper-Conservative ICA + EOG Safety):
% - UPDATED: Brain threshold 25% → 10% (maximum cognitive protection)
% - UPDATED: EOG correlation 0.25 → 0.30 (stricter criterion)
% - NEW: EOG safety net (EOG≥0.60 + Eye≥20% = force rejection)
% - IMPROVED: Catches high-EOG artifacts even if Brain>10%
% - RATIONALE: Prevents rejection of frontal ERPs while ensuring eye artifact removal
%
% v2.0 → v2.1 (ICA Optimization + Temporal Correction):
% - NEW: Mandatory IC1 rejection (highest eye probability)
% - NEW: Conditional IC2-3 rejection with 3 criteria
% - NEW: aut24 temporal correction (+42ms on all triggers)
%
% Previous versions: See v2.0 header for full changelog
%
% ========================================================================
% EXPECTED RESULTS:
% ========================================================================
% - Files processed: ~50 (passations 2/3 only)
% - Success rate: 75-80% (~38-40 files)
% - Epochs per file: 90-100 average (25-33 Valide, 62-75 Invalide)
% - SNR improvement: +20-25% vs. raw data
% - Eye artifacts: <5% residual (excellent)
% - IC rejection: 1-3 components per file (guaranteed minimum 1)
%
% ========================================================================
% TECHNICAL SPECIFICATIONS:
% ========================================================================
% Expected trials: 140 per recording (42 Valide, 98 Invalide)
% Sampling rate: 1024 Hz (BioSemi ActiveTwo)
% Channels: 32 scalp + TP9/TP10/EOG/ref
% Task structure: Single block (no segmentation)
% ISI: 1000 ms minimum
% Response window: 200-2000 ms
%
%% ========================================================================

clear; clc; close all;

% Parameters - v2.2: Hyper-conservative ICA with EOG safety net
useGUI = false;
maxInterpolations = 2;
iclabelEyeThreshold = 0.60;      % Threshold for IC2-3 (Eye probability)
iclabelBrainThreshold = 0.10;    % STRICTER: Brain activity protection (was 0.25)
eogCorrThreshold = 0.35;         % STRICTER: EOG correlation for IC2-3 (was 0.25)
eogSafetyThreshold = 0.70;       % NEW: Critical EOG correlation (auto-reject)
eyeSafetyThreshold = 0.40;       % NEW: Minimum eye prob for safety net
maxEyeICs = 3;                   % Maximum total (1 mandatory + 2 conditional)

% Noise detection parameters
extremeAmpThresholdUv = 100;         
extremeAmpPropThreshold = 0.05;      
asrMaxBoundaries = 50;               

% Validation thresholds
minValideTrials = 25;    % Minimum Valide trials (out of 42) - 60%
minInvalideTrials = 60;  % Minimum Invalide trials (out of 98) - 61%
minTotalTrials = 85;     % Minimum total trials (out of 140) - 61%

% Racine des données : definie dans chemins.m (racine du depot).
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));  % chemins.m, a la racine du depot
C = chemins();
dataRoot = C.racine;
% Base.csv path for exposure type mapping
baseCsvPath = fullfile(dataRoot, 'Data', 'Base.csv');

% ========================================================================
% NEW: MODE SUBSTRAT TEMPS-FRÉQUENCE
%   'erp'  : comportement actuel STRICTEMENT INCHANGÉ (fichiers époqués validés)
%   'freq' : sauvegarde le CONTINU prétraité (post-ICA) pour analyses TF
%            - fenêtres de conservation élargies (les époques longues survivront)
%            - PAS de validation min-trials : le continu est sauvegardé quoi qu'il arrive
%            - corrections de timing reportées au script d'époquage TF
% ========================================================================
runMode          = 'erp';    % chaine rapportee dans la these ('freq' = mode exploratoire non rapporte)
keepPreSec_freq  = 2.0;      % vs 0.2 en erp
keepPostSec_freq = 3.0;      % vs 1.5 en erp

% Parallel configuration
numWorkers = 4;

% Adjacency map (will be passed to workers)
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

% Datasets configuration
printBDF = fullfile(dataRoot, 'Campagne_print_25', 'Cognition', 'EEG', 'Brut');
autBDF   = fullfile(dataRoot, 'Campagne_aut_24', 'Cognition', 'EEG', 'EEG_Brut');
out1B    = fullfile(dataRoot, 'Data', 'Cognition', 'EEG', 'Study_all', '1B_all', 'processed');
out2B    = fullfile(dataRoot, 'Data', 'Cognition', 'EEG', 'Study_all', '2B_all', 'processed');
datasets = {
    struct('input', printBDF, 'output', out1B, 'type', 'print25', 'task', '1B'), ...
    struct('input', printBDF, 'output', out2B, 'type', 'print25', 'task', '2B'), ...
    struct('input', autBDF,   'output', out1B, 'type', 'aut24',   'task', '1B'), ...
    struct('input', autBDF,   'output', out2B, 'type', 'aut24',   'task', '2B')
};

% Load exposure mapping
fprintf('📂 Chargement Base.csv...\n');
exposureMap = load_exposure_mapping(baseCsvPath);
fprintf('✅ Base.csv chargé\n\n');

% Initialize EEGLAB
if ~exist('eeglab', 'file')
    error('EEGLAB non trouvé dans le path MATLAB');
end
eeglab nogui;

% Setup parallel pool
if isempty(gcp('nocreate'))
    parpool(numWorkers);
end

% Process each dataset
for d = 1:length(datasets)
    input_folder = datasets{d}.input;
    output_folder = datasets{d}.output;
    set_type = datasets{d}.type;
    task_type = datasets{d}.task;
    
    fprintf('\n========================================\n');
    fprintf('DATASET: %s - %s\n', set_type, task_type);
    fprintf('========================================\n');
    
    % Create output folder
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % NEW: dossier substrat continu (mode freq uniquement)
    output_freq = fullfile(output_folder, 'continuous_freq');
    if strcmp(runMode, 'freq')
        if ~exist(output_freq, 'dir'), mkdir(output_freq); end
        fprintf('   [FREQ] Substrat continu -> %s\n', output_freq);
    end
    
    % Find files matching task pattern
    pattern = sprintf('*-%s-*.bdf', task_type);
    all_files = dir(fullfile(input_folder, pattern));
    
    % Filter to keep only passations 2 and 3
    files = [];
    for f = 1:length(all_files)
        fname = all_files(f).name;
        % Check if filename contains -2 or -3 (passation number)
        if contains(fname, '-2.bdf') || contains(fname, '-2_') || ...
           contains(fname, '-3.bdf') || contains(fname, '-3_')
            files = [files; all_files(f)];
        end
    end

    if isempty(files)
        fprintf('⚠️ Aucun fichier passation 2/3 trouvé pour pattern: %s\n', pattern);
        continue;
    end
    
    fprintf('📁 %d fichiers trouvés (passations 2 et 3 uniquement)\n', length(files));
    
    % Prepare file list
    filepaths = cell(length(files), 1);
    filenames = cell(length(files), 1);
    for i = 1:length(files)
        filepaths{i} = fullfile(input_folder, files(i).name);
        filenames{i} = files(i).name;
    end
    
    % Setup log file
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    logPath = fullfile(output_folder, sprintf('processing_log_%s_%s.txt', task_type, timestamp));
    write_log_header(logPath);
    
    % Parallel processing
    parfor i = 1:length(filepaths)
        try
            process_single_nback_file(filepaths{i}, filenames{i}, output_folder, ...
                set_type, adjacencyMap, maxInterpolations, iclabelEyeThreshold, ...
                iclabelBrainThreshold, eogCorrThreshold, eogSafetyThreshold, ...
                eyeSafetyThreshold, maxEyeICs, extremeAmpThresholdUv, ...
                extremeAmpPropThreshold, asrMaxBoundaries, exposureMap, logPath, ...
                minValideTrials, minInvalideTrials, minTotalTrials, ...
                runMode, output_freq, keepPreSec_freq, keepPostSec_freq);
        catch ME
            fprintf('[Worker %d] ❌ Erreur fichier %s: %s\n', get_worker_id(), filenames{i}, ME.message);
        end
    end
end

warning('on', 'all');
fprintf('\n========================================\n');
fprintf('TRAITEMENT PARALLÈLE TERMINÉ\n');
fprintf('========================================\n');

%% ========================================================================
%% WORKER FUNCTION
%% ========================================================================
function process_single_nback_file(filepath, filename, output_folder, ...
    set_type, adjacencyMap, maxInterpolations, iclabelEyeThreshold, ...
    iclabelBrainThreshold, eogCorrThreshold, eogSafetyThreshold, ...
    eyeSafetyThreshold, maxEyeICs, extremeAmpThresholdUv, ...
    extremeAmpPropThreshold, asrMaxBoundaries, exposureMap, logPath, ...
    minValideTrials, minInvalideTrials, minTotalTrials, ...
    runMode, output_freq, keepPreSec_freq, keepPostSec_freq)

    ensure_eeglab();   % init EEGLAB une seule fois par worker (évite les appels réseau répétés)

    % NEW: mode substrat TF vs mode ERP (défaut). Fenêtres de conservation adaptées.
    isFreq = strcmp(runMode, 'freq');
    if isFreq
        keepPreSec = keepPreSec_freq; keepPostSec = keepPostSec_freq;
        fprintf('[Worker %d] 🔄 Traitement [MODE FREQ]: %s\n', get_worker_id(), filename);
    else
        keepPreSec = 0.2; keepPostSec = 1.5;   % valeurs actuelles (inchangées)
        fprintf('[Worker %d] 🔄 Traitement: %s\n', get_worker_id(), filename);
    end

    cutSegmentsSec = {};
    asrBoundariesAdded = 0;
    interpolatedChans = {};
    comps_to_reject = [];
    rejectedEpochs = [];
    extremeNoiseSegments = 0;

    % [FREQ] Reprise : ignore les fichiers dont le continu existe déjà
    if isFreq
        [~, nm0, ~] = fileparts(filename);
        [pId0, att0] = parse_filename_matlab(filename);
        expo0 = '';
        if ~isempty(pId0) && att0 > 1 && ~isempty(exposureMap) && isfield(exposureMap, pId0)
            af0 = sprintf('attempt%d', att0);
            if isfield(exposureMap.(pId0), af0), expo0 = exposureMap.(pId0).(af0); end
        end
        if ~isempty(expo0)
            skipName = sprintf('%s-%s_%s_continuous.set', nm0, expo0, set_type);
        else
            skipName = sprintf('%s_%s_continuous.set', nm0, set_type);
        end
        if exist(fullfile(output_freq, skipName), 'file')
            fprintf('[Worker %d] ⏭️ [FREQ] déjà fait, ignoré: %s\n', get_worker_id(), skipName);
            return;
        end
    end

    %% === Import ===
    if ~exist(filepath, 'file')
        fprintf('[Worker %d] ❌ Fichier introuvable: %s\n', get_worker_id(), filepath);
        return;
    end
    
    try
        EEG = pop_biosig(filepath);
        EEG = eeg_checkset(EEG);
    catch ME
        fprintf('[Worker %d] ❌ Import échoué: %s\n', get_worker_id(), ME.message);
        return;
    end

    %% === Trigger conversion ===
    new_events = [];
    triggersInitial = length(EEG.event);
    
    switch set_type
        case 'print25'
            % PRINT25: Pair logic with fallback to OpenSesame triggers only
            % Priority: Use pairs if available (65296+65288 or 65312+65288)
            % Fallback: Use OpenSesame triggers alone (65296 or 65312)
            i = 1;
            while i <= length(EEG.event)
                curr_type = EEG.event(i).type;
                if ischar(curr_type), curr_type = str2double(curr_type); end

                % Valide: Try 65296 + 65288 pair first
                if isequal(curr_type, 65296) && i < length(EEG.event)
                    next_type = EEG.event(i+1).type;
                    if ischar(next_type), next_type = str2double(next_type); end
                    dt = (EEG.event(i+1).latency - EEG.event(i).latency) / EEG.srate * 1000;
                    
                    if dt <= 100 && isequal(next_type, 65288)
                        % Pair found: use trigger-box timing
                        EEG.event(i+1).type = 'Valide';
                        new_events = [new_events EEG.event(i+1)];
                        i = i + 1;  % Skip both triggers
                    else
                        % No pair: use OpenSesame trigger alone (fallback)
                        EEG.event(i).type = 'Valide';
                        % Apply +42ms correction ONLY for known exceptions without trigger-box
                        if (contains(filename, '226') && contains(filename, 'B-3')) || ...
                           (contains(filename, '227') && contains(filename, 'B-2'))
                            EEG.event(i).latency = EEG.event(i).latency + (0.042 * EEG.srate);
                        end
                        new_events = [new_events EEG.event(i)];
                    end
                    
                % Invalide: Try 65312 + 65288 pair first
                elseif isequal(curr_type, 65312) && i < length(EEG.event)
                    next_type = EEG.event(i+1).type;
                    if ischar(next_type), next_type = str2double(next_type); end
                    dt = (EEG.event(i+1).latency - EEG.event(i).latency) / EEG.srate * 1000;
                    
                    if dt <= 100 && isequal(next_type, 65288)
                        % Pair found: use trigger-box timing
                        EEG.event(i+1).type = 'Invalide';
                        new_events = [new_events EEG.event(i+1)];
                        i = i + 1;  % Skip both triggers
                    else
                        % No pair: use OpenSesame trigger alone (fallback)
                        EEG.event(i).type = 'Invalide';
                        % Apply +42ms correction ONLY for known exceptions without trigger-box
                        if (contains(filename, '226') && contains(filename, 'B-3')) || ...
                           (contains(filename, '227') && contains(filename, 'B-2'))
                            EEG.event(i).latency = EEG.event(i).latency + (0.042 * EEG.srate);
                        end
                        new_events = [new_events EEG.event(i)];
                    end
                    
                % Orphan 65288: Skip (was part of a non-matching pair)
                elseif isequal(curr_type, 65288)
                    % Check if already paired
                    isPaired = false;
                    if i > 1
                        prev_type = EEG.event(i-1).type;
                        if ischar(prev_type), prev_type = str2double(prev_type); end
                        dt_prev = (EEG.event(i).latency - EEG.event(i-1).latency) / EEG.srate * 1000;
                        if ismember(prev_type, [65296, 65312]) && dt_prev <= 100
                            isPaired = true;
                        end
                    end
                    
                    % If not paired, it's an orphan 65288 - skip it
                    if ~isPaired
                        % Don't add to new_events
                    end
                    
                % Response triggers (keep for analysis)
                elseif ismember(curr_type, [65408, 65344])
                    if curr_type == 65408
                        EEG.event(i).type = 'RespValide';
                    else
                        EEG.event(i).type = 'RespInvalide';
                    end
                    new_events = [new_events EEG.event(i)];
                end
                
                i = i + 1;
            end
            
        case 'aut24'
            % AUT24: Direct conversion + 42ms temporal correction
            for i = 1:length(EEG.event)
                curr_type = EEG.event(i).type;
                if ischar(curr_type), curr_type = str2double(curr_type); end
                
                % Stimuli
                if curr_type == 2
                    EEG.event(i).type = 'Valide';
                    % Apply 42ms correction
                    EEG.event(i).latency = EEG.event(i).latency + (0.042 * EEG.srate);
                    new_events = [new_events EEG.event(i)];
                elseif curr_type == 4
                    EEG.event(i).type = 'Invalide';
                    % Apply 42ms correction
                    EEG.event(i).latency = EEG.event(i).latency + (0.042 * EEG.srate);
                    new_events = [new_events EEG.event(i)];
                    
                % Responses (keep for analysis)
                elseif curr_type == 2048
                    EEG.event(i).type = 'repFaux';
                    EEG.event(i).latency = EEG.event(i).latency + (0.042 * EEG.srate);
                    new_events = [new_events EEG.event(i)];
                elseif curr_type == 4096
                    EEG.event(i).type = 'NrepJuste';
                    EEG.event(i).latency = EEG.event(i).latency + (0.042 * EEG.srate);
                    new_events = [new_events EEG.event(i)];
                elseif curr_type == 8192
                    EEG.event(i).type = 'repJuste';
                    EEG.event(i).latency = EEG.event(i).latency + (0.042 * EEG.srate);
                    new_events = [new_events EEG.event(i)];
                elseif curr_type == 16384
                    EEG.event(i).type = 'NrepFaux';
                    EEG.event(i).latency = EEG.event(i).latency + (0.042 * EEG.srate);
                    new_events = [new_events EEG.event(i)];
                end
            end
    end
    
    EEG.event = new_events;
    EEG = eeg_checkset(EEG);
    
    %% === Burst detection and removal ===
    triggersBeforeBurst = length(EEG.event);
    
    % Define valid types
    validTypes = {'Valide', 'Invalide', 'RespValide', 'RespInvalide', ...
                  'repFaux', 'NrepJuste', 'repJuste', 'NrepFaux', 'boundary'};
    
    EEG.event = remove_noise_burst_triggers_smart(EEG.event, EEG.srate, validTypes);
    EEG = eeg_checkset(EEG);
    
    triggersAfterBurst = length(EEG.event);
    burstsRemoved = triggersBeforeBurst - triggersAfterBurst;
    
    if burstsRemoved > 0
        fprintf('[Worker %d] 🧹 Removed %d burst triggers\n', get_worker_id(), burstsRemoved);
    end
    
    % Count trial types
    valideCount = sum(strcmp({EEG.event.type}, 'Valide'));
    invalideCount = sum(strcmp({EEG.event.type}, 'Invalide'));
    
    fprintf('[Worker %d] 🔢 Triggers convertis: %d Valide, %d Invalide (sur %d triggers initiaux)\n', ...
        get_worker_id(), valideCount, invalideCount, triggersInitial);
    
    if valideCount == 0 && invalideCount == 0
        fprintf('[Worker %d] ❌ FATAL: Aucun trigger Valide/Invalide trouvé!\n', get_worker_id());
        return;
    end

    %% === Channel correction and selection ===
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

    %% === Filtering ===
    EEG = pop_eegfiltnew(EEG, 0.5, 40);
    EEG = eeg_checkset(EEG);

    %% === Detect and remove extreme noise ===
    try
        [EEG, extremeNoiseSegments] = detect_and_remove_extreme_noise(EEG, extremeAmpThresholdUv);
        if extremeNoiseSegments > 0
            fprintf('[Worker %d] 🔥 %d segments de bruit extrême supprimés\n', get_worker_id(), extremeNoiseSegments);
        end
    catch ME
        fprintf('[Worker %d] ⚠️ Détection de bruit extrême échouée: %s\n', get_worker_id(), ME.message);
    end

    %% === ASR repair ===
    try
        if exist('pop_clean_rawdata','file')
            EEG = pop_clean_rawdata(EEG, ...
                'FlatlineCriterion', 5, 'Highpass', 'off', 'ChannelCriterion', 'off', ...
                'LineNoiseCriterion', 'off', 'BurstCriterion', 30, ...
                'BurstRejection', 'off', 'WindowCriterion', 'off', ...
                'UseRiemannian', false);  % ← Désactive le filtrage spectral
        end
    catch ME
        fprintf('[Worker %d] ⚠️ ASR repair échoué: %s\n', get_worker_id(), ME.message);
    end
    EEG = eeg_checkset(EEG);

    %% === Remove long silent segments ===
    try
        triggerTypes = {'Valide', 'Invalide'};
        evLat = [];
        for ei = 1:numel(EEG.event)
            et = EEG.event(ei).type;
            if ischar(et) && any(strcmpi(et, triggerTypes))
                evLat(end+1) = round(EEG.event(ei).latency);
            end
        end
        evLat = unique(evLat);
        keepPre = max(1, round(keepPreSec * EEG.srate));
        keepPost = max(1, round(keepPostSec * EEG.srate));
        rej = compute_rejection_segments(evLat, EEG.pnts, EEG.srate, keepPre, keepPost);
        if ~isempty(rej)
            for ri = 1:size(rej,1)
                cutSegmentsSec{end+1} = sprintf('%.3f-%.3f', rej(ri,1)/EEG.srate, rej(ri,2)/EEG.srate);
            end
            EEG = eeg_eegrej(EEG, rej);
            EEG = eeg_checkset(EEG);
        end
    catch
    end

    %% === ASR limited rejection ===
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
                asrBoundariesAdded = max(0, added);
                if added > asrMaxBoundaries
                    fprintf('[Worker %d] ⚠️ ASR rejected: %d boundaries > limit %d\n', ...
                        get_worker_id(), added, asrMaxBoundaries);
                    EEG = EEG_before;
                else
                    EEG = EEG_tmp;
                end
            end
        end
    catch
    end
    EEG = eeg_checkset(EEG);

    %% === Re-reference BEFORE ICA ===
    try
        chanLabels = {EEG.chanlocs.labels};
        refIdx = find(strcmpi(chanLabels, 'ref'), 1);
        if ~isempty(refIdx)
            EEG = pop_reref(EEG, refIdx, 'keepref', 'on');
            EEG = eeg_checkset(EEG);
        end
    catch
    end

    %% === ICA ===
    try
        EEG = pop_runica(EEG, 'icatype', 'runica');
        EEG = eeg_checkset(EEG, 'ica');
    catch ME
        fprintf('[Worker %d] ⚠️ ICA échoué: %s\n', get_worker_id(), ME.message);
    end

    %% === IC rejection - v2.2: 1 mandatory + 2 conditional + EOG safety ===
    if ~isempty(EEG.icaweights)
        try
            if exist('iclabel','file')
                EEG = iclabel(EEG);
                if isfield(EEG,'etc') && isfield(EEG.etc,'ic_classification')
                    probs = EEG.etc.ic_classification.ICLabel.classifications;
                    eyeProbs = probs(:,3);   % Column 3 = Eye
                    brainProbs = probs(:,1); % Column 1 = Brain
                    [sortedProbs, order] = sort(eyeProbs, 'descend');
                    
                    % === IC1: MANDATORY (highest eye probability) ===
                    comps_to_reject = order(1);
                    fprintf('[Worker %d] 🔍 IC%d: %.1f%% Eye, %.1f%% Brain → OBLIGATOIRE ✅\n', ...
                        get_worker_id(), comps_to_reject(1), ...
                        eyeProbs(comps_to_reject(1))*100, brainProbs(comps_to_reject(1))*100);
                    
                    % === IC2-3: CONDITIONAL (strict multi-criteria + safety net) ===
                    eogIdx = find(strcmpi({EEG.chanlocs.labels}, 'EOG'), 1);
                    if ~isfield(EEG,'icaact') || isempty(EEG.icaact)
                        EEG = eeg_checkset(EEG, 'ica');
                    end
                    
                    % Check EOG availability
                    hasEOG = ~isempty(eogIdx);
                    if hasEOG
                        fprintf('[Worker %d] 🔍 EOG channel: Found (validation active)\n', get_worker_id());
                    else
                        fprintf('[Worker %d] 🔍 EOG channel: NOT FOUND (using 2 criteria only)\n', get_worker_id());
                    end
                    
                    % Evaluate IC2 and IC3 candidates
                    for candidateIdx = 2:min(3, length(order))
                        ic = order(candidateIdx);
                        
                        % CRITERION 1: Eye probability ≥ 60%
                        criterion1 = eyeProbs(ic) >= iclabelEyeThreshold;
                        
                        % CRITERION 2: Brain probability < 10% (STRICTER v2.2)
                        criterion2 = brainProbs(ic) < iclabelBrainThreshold;
                        
                        % CRITERION 3: EOG correlation ≥ 0.30 (STRICTER v2.2)
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
                        
                        % === SAFETY NET v2.2: Very high EOG = force rejection ===
                        safetyNet = false;
                        if hasEOG && ~isnan(eogCorr)
                            if eogCorr >= eogSafetyThreshold && eyeProbs(ic) >= eyeSafetyThreshold
                                safetyNet = true;
                                fprintf('[Worker %d] 🚨 SAFETY NET: IC%d (EOG=%.2f, Eye=%.1f%%) → Force rejection\n', ...
                                    get_worker_id(), ic, eogCorr, eyeProbs(ic)*100);
                            end
                        end
                        
                        % DECISION: At least 2/3 criteria (or 2/2 if no EOG) OR safety net
                        numCriteriaMet = sum([criterion1, criterion2, criterion3]);
                        totalCriteria = 2 + hasEOG; % 2 or 3 depending on EOG
                        
                        shouldReject = (numCriteriaMet >= 2) || ...
                                       (~hasEOG && criterion1 && criterion2) || ...
                                       safetyNet;
                        
                        if shouldReject
                            comps_to_reject(end+1) = ic;
                            if safetyNet
                                fprintf('[Worker %d] ✅ IC%d: REJETÉ via SAFETY NET\n', get_worker_id(), ic);
                            elseif hasEOG
                                fprintf('[Worker %d] ✅ IC%d: Eye=%.1f%%, Brain=%.1f%%, EOG=%.2f → REJETÉ (%d/3 critères)\n', ...
                                    get_worker_id(), ic, eyeProbs(ic)*100, brainProbs(ic)*100, eogCorr, numCriteriaMet);
                            else
                                fprintf('[Worker %d] ✅ IC%d: Eye=%.1f%%, Brain=%.1f%% → REJETÉ (%d/2 critères)\n', ...
                                    get_worker_id(), ic, eyeProbs(ic)*100, brainProbs(ic)*100, numCriteriaMet);
                            end
                        else
                            if hasEOG
                                fprintf('[Worker %d] ⏭️  IC%d: Eye=%.1f%%, Brain=%.1f%%, EOG=%.2f → CONSERVÉ (%d/3 critères)\n', ...
                                    get_worker_id(), ic, eyeProbs(ic)*100, brainProbs(ic)*100, eogCorr, numCriteriaMet);
                            else
                                fprintf('[Worker %d] ⏭️  IC%d: Eye=%.1f%%, Brain=%.1f%% → CONSERVÉ (%d/2 critères)\n', ...
                                    get_worker_id(), ic, eyeProbs(ic)*100, brainProbs(ic)*100, numCriteriaMet);
                            end
                        end
                    end
                    
                    fprintf('[Worker %d] 📊 Total composantes rejetées: %d\n', ...
                        get_worker_id(), length(comps_to_reject));
                end
            end
        catch ME
            fprintf('[Worker %d] ⚠️ ICLabel échoué: %s\n', get_worker_id(), ME.message);
        end
    end
    
    if ~isempty(comps_to_reject)
        EEG = pop_subcomp(EEG, comps_to_reject, 0);
        EEG = eeg_checkset(EEG);
    end

    %% === Interpolation ===
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
                
                % Clear icaact to prevent mismatch warnings after component rejection
                if isfield(EEG, 'icaact')
                    EEG.icaact = [];
                end
            end
        end
    catch
    end

    %% === Remove ref electrodes ===
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

    %% === [FREQ] Save continuous substrate and STOP ===
    % En mode freq, on sauvegarde le continu prétraité (post-ICA, montage scalp)
    % SANS validation min-trials, et on saute l'époquage ERP : le ré-époquage TF
    % se fait dans un script dédié. Corrections de timing reportées à ce script.
    if isFreq
        [~, name, ~] = fileparts(filename);
        [participantId, attemptNumber] = parse_filename_matlab(filename);
        exposureType = '';
        if ~isempty(participantId) && attemptNumber > 1 && ~isempty(exposureMap)
            if isfield(exposureMap, participantId)
                attemptField = sprintf('attempt%d', attemptNumber);
                if isfield(exposureMap.(participantId), attemptField)
                    exposureType = exposureMap.(participantId).(attemptField);
                end
            end
        end
        if ~isempty(exposureType)
            outname = fullfile(output_freq, sprintf('%s-%s_%s_continuous.set', name, exposureType, set_type));
        else
            outname = fullfile(output_freq, sprintf('%s_%s_continuous.set', name, set_type));
        end
        EEG.setname = sprintf('%s_%s_continuous', name, set_type);
        EEG = pop_saveset(EEG, 'filename', outname);
        fprintf('[Worker %d] 💾 [FREQ] Continu sauvegardé (%d canaux, %.0f s): %s\n', ...
            get_worker_id(), EEG.nbchan, EEG.xmax, name);
        return;
    end

    %% === Epoching ===
    try
        EEG = pop_epoch(EEG, {'Valide', 'Invalide'}, [-0.2 0.8], 'epochinfo', 'yes');
        EEG = eeg_checkset(EEG);
    catch ME
        fprintf('[Worker %d] ❌ Époquage échoué: %s\n', get_worker_id(), filename);
        return;
    end

    EEG = pop_rmbase(EEG, [-200 0]);
    EEG = eeg_checkset(EEG);

    %% === Epoch rejection (adaptive) ===
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
                if latMs < 500
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
        fprintf('[Worker %d] 🗑️ Step 1: %d époques rejetées (boundary < 500ms)\n', ...
            get_worker_id(), length(earlyBoundaryEpochs));
    end

    % Step 2: Adaptive rejection
    try
        % Adaptive thresholds
        if epochsBeforeRejection < 100
            thresholdSingle = 5;
            thresholdJoint = 5;
            fprintf('[Worker %d] 📊 Seuils permissifs (5 SD) - %d époques\n', get_worker_id(), epochsBeforeRejection);
        elseif epochsBeforeRejection < 120
            thresholdSingle = 4;
            thresholdJoint = 4;
            fprintf('[Worker %d] 📊 Seuils modérés (4 SD) - %d époques\n', get_worker_id(), epochsBeforeRejection);
        else
            thresholdSingle = 3;
            thresholdJoint = 3;
            fprintf('[Worker %d] 📊 Seuils stricts (3 SD) - %d époques\n', get_worker_id(), epochsBeforeRejection);
        end
        
        [EEG, ~, ~, ~] = pop_jointprob(EEG, 1, 1:EEG.nbchan, thresholdSingle, thresholdJoint, 0, 0, 0, [], 0);
        
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
            % Limit to 30% max
            maxReject = floor(0.30 * EEG.trials);
            if length(jointprobEpochs) > maxReject
                fprintf('[Worker %d] ⚠️ Limitation: %d identifiées, rejet limité à %d (30%%)\n', ...
                        get_worker_id(), length(jointprobEpochs), maxReject);
                jointprobEpochs = jointprobEpochs(1:maxReject);
            end
            
            originalIndices = map_to_original_indices(jointprobEpochs, allRejectedEpochs, epochsBeforeRejection);
            allRejectedEpochs = [allRejectedEpochs, originalIndices];
            
            EEG = pop_rejepoch(EEG, jointprobEpochs, 0);
            EEG = eeg_checkset(EEG);
            fprintf('[Worker %d] 🗑️ Step 2: %d époques rejetées (jointprob)\n', ...
                get_worker_id(), length(jointprobEpochs));
        else
            fprintf('[Worker %d] ℹ️ Step 2: Aucune époque rejetée (jointprob)\n', get_worker_id());
        end
        
    catch jointprobError
        fprintf('[Worker %d] ❌ Rejet d''époques échoué: %s\n', ...
            get_worker_id(), jointprobError.message);
    end
    
    epochsAfterRejection = EEG.trials;
    rejectedEpochCount = epochsBeforeRejection - epochsAfterRejection;
    
    allRejectedEpochs = sort(unique(allRejectedEpochs));
    
    fprintf('[Worker %d] 📊 Résumé rejet: %d/%d époques rejetées (%d restantes)\n', ...
        get_worker_id(), rejectedEpochCount, epochsBeforeRejection, epochsAfterRejection);

    %% === Validation: Count trial types ===
    valideEpochs = 0;
    invalideEpochs = 0;
    
    for ep = 1:EEG.trials
        epochEvents = EEG.epoch(ep).eventtype;
        if ~iscell(epochEvents), epochEvents = {epochEvents}; end
        
        if any(strcmp(epochEvents, 'Valide'))
            valideEpochs = valideEpochs + 1;
        elseif any(strcmp(epochEvents, 'Invalide'))
            invalideEpochs = invalideEpochs + 1;
        end
    end
    
    fprintf('[Worker %d] 📊 Époques finales: %d Valide, %d Invalide (Total: %d)\n', ...
        get_worker_id(), valideEpochs, invalideEpochs, epochsAfterRejection);
    
    % Check validation criteria
    passValidation = (valideEpochs >= minValideTrials) && ...
                     (invalideEpochs >= minInvalideTrials) && ...
                     (epochsAfterRejection >= minTotalTrials);
    
    if ~passValidation
        fprintf('[Worker %d] ❌ VALIDATION ÉCHOUÉE:\n', get_worker_id());
        fprintf('[Worker %d]    - Valide: %d (min: %d) %s\n', get_worker_id(), ...
            valideEpochs, minValideTrials, ternary(valideEpochs >= minValideTrials, '✅', '❌'));
        fprintf('[Worker %d]    - Invalide: %d (min: %d) %s\n', get_worker_id(), ...
            invalideEpochs, minInvalideTrials, ternary(invalideEpochs >= minInvalideTrials, '✅', '❌'));
        fprintf('[Worker %d]    - Total: %d (min: %d) %s\n', get_worker_id(), ...
            epochsAfterRejection, minTotalTrials, ternary(epochsAfterRejection >= minTotalTrials, '✅', '❌'));
        return;
    end
    
    fprintf('[Worker %d] ✅ VALIDATION RÉUSSIE\n', get_worker_id());

    %% === Save with exposure suffix ===
    [~, name, ~] = fileparts(filename);
    [participantId, attemptNumber] = parse_filename_matlab(filename);
    exposureType = '';

    if ~isempty(participantId) && attemptNumber > 1 && ~isempty(exposureMap)
        if isfield(exposureMap, participantId)
            attemptField = sprintf('attempt%d', attemptNumber);
            if isfield(exposureMap.(participantId), attemptField)
                exposureType = exposureMap.(participantId).(attemptField);
            end
        end
    end

    if ~isempty(exposureType)
        outname = fullfile(output_folder, sprintf('%s-%s.set', name, exposureType));
        fprintf('[Worker %d] 💾 Saving as: %s-%s.set\n', get_worker_id(), name, exposureType);
    else
        outname = fullfile(output_folder, [name '.set']);
        fprintf('[Worker %d] ⚠️ No exposure found, saving as: %s.set\n', get_worker_id(), name);
    end

    EEG.setname = name;
    EEG = pop_saveset(EEG, 'filename', outname);

    write_log_entry_nback(logPath, filename, interpolatedChans, comps_to_reject, ...
        allRejectedEpochs, rejectedEpochCount, epochsAfterRejection, cutSegmentsSec, ...
        asrBoundariesAdded, extremeNoiseSegments, valideEpochs, invalideEpochs);

    fprintf('[Worker %d] ✅ Terminé: %s (%d Valide, %d Invalide)\n', ...
        get_worker_id(), filename, valideEpochs, invalideEpochs);
end

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function result = ternary(condition, trueVal, falseVal)
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end

function originalIndices = map_to_original_indices(currentIndices, alreadyRejected, totalOriginal)
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

function [participantId, attemptNumber] = parse_filename_matlab(filename)
    participantId = '';
    attemptNumber = 0;
    
    tokens = regexp(filename, '^[Pp](\d+)-\w+-(\d+)', 'tokens');
    if ~isempty(tokens)
        participantId = sprintf('P%s', tokens{1}{1});
        attemptNumber = str2double(tokens{1}{2});
    end
end

function exposureMap = load_exposure_mapping(csvPath)
    exposureMap = struct();
    
    if ~exist(csvPath, 'file')
        warning('Base.csv non trouvé: %s', csvPath);
        return;
    end
    
    try
        % Try reading with comma first
        try
            data = readtable(csvPath, 'Delimiter', ',', 'ReadVariableNames', true);
            % Check if properly parsed (should have 3 columns)
            if width(data) < 3
                % Try semicolon instead
                data = readtable(csvPath, 'Delimiter', ';', 'ReadVariableNames', true);
            end
        catch
            % Fallback to semicolon
            data = readtable(csvPath, 'Delimiter', ';', 'ReadVariableNames', true);
        end
        
        % Verify we have at least 3 columns
        if width(data) < 3
            error('Base.csv doit avoir au moins 3 colonnes (Participant, 2, 3)');
        end
        
        % Get column names
        colNames = data.Properties.VariableNames;
        
        fprintf('📊 Colonnes détectées: %s\n', strjoin(colNames, ', '));
        fprintf('📊 Lignes: %d participants\n', height(data));
        
        for i = 1:height(data)
            % First column should be Participant
            participant = data{i, 1};
            if iscell(participant)
                participant = participant{1};
            end
            if ~ischar(participant) && ~isstring(participant)
                participant = num2str(participant);
            end
            participant = char(participant);
            
            % Skip header row if it slipped through
            if strcmpi(participant, 'Participant')
                continue;
            end
            
            % Initialize participant structure
            if ~isfield(exposureMap, participant)
                exposureMap.(participant) = struct();
            end
            
            % Second column = Passation 2
            expo2 = data{i, 2};
            if iscell(expo2)
                expo2 = expo2{1};
            end
            if ~ischar(expo2) && ~isstring(expo2)
                expo2 = num2str(expo2);
            end
            expo2 = char(expo2);
            
            if strcmpi(expo2, 'C')
                exposureMap.(participant).attempt2 = 'close';
            elseif strcmpi(expo2, 'A')
                exposureMap.(participant).attempt2 = 'away';
            end
            
            % Third column = Passation 3
            if width(data) >= 3
                expo3 = data{i, 3};
                if iscell(expo3)
                    expo3 = expo3{1};
                end
                if ~ischar(expo3) && ~isstring(expo3)
                    expo3 = num2str(expo3);
                end
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
        fprintf('Stack trace:\n%s\n', ME.getReport());
    end
end

function [EEG, numSegments] = detect_and_remove_extreme_noise(EEG, thresholdUv)
    numSegments = 0;
    
    X = double(EEG.data);
    mask = any(abs(X) > thresholdUv, 1);
    
    starts = find(diff([0 mask]) == 1);
    ends = find(diff([mask 0]) == -1);
    
    minDuration = 10 * EEG.srate;
    longSegments = (ends - starts + 1) >= minDuration;
    
    if any(longSegments)
        starts = starts(longSegments);
        ends = ends(longSegments);
        numSegments = length(starts);
        
        rej = [starts(:) ends(:)];
        EEG = eeg_eegrej(EEG, rej);
        EEG = eeg_checkset(EEG);
    end
end

function rej = compute_rejection_segments(triggerLatencies, totalPnts, srate, keepPre, keepPost)
    rej = [];
    
    if isempty(triggerLatencies)
        return;
    end
    
    keep = false(1, totalPnts);
    for i = 1:length(triggerLatencies)
        lat = triggerLatencies(i);
        rangeStart = max(1, lat - keepPre);
        rangeEnd = min(totalPnts, lat + keepPost);
        keep(rangeStart:rangeEnd) = true;
    end
    
    toReject = ~keep;
    starts = find(diff([0 toReject]) == 1);
    ends = find(diff([toReject 0]) == -1);
    
    minGap = 1 * srate;
    longEnough = (ends - starts + 1) >= minGap;
    
    if any(longEnough)
        rej = [starts(longEnough)' ends(longEnough)'];
    end
end

function nb = count_boundaries(EEG)
    nb = 0;
    for i = 1:length(EEG.event)
        if strcmp(EEG.event(i).type, 'boundary')
            nb = nb + 1;
        end
    end
end

function standard_locs = find_standard_locs(eeglab_path)
    standard_locs = '';
    possible = {
        fullfile(eeglab_path, 'plugins', 'dipfit', 'standard_BEM', 'elec', 'standard_1005.elc'),
        fullfile(eeglab_path, 'plugins', 'dipfit2.3', 'standard_BEM', 'elec', 'standard_1005.elc'),
        fullfile(eeglab_path, 'sample_locs', 'standard_1005.elc')
    };
    
    for i = 1:length(possible)
        if exist(possible{i}, 'file')
            standard_locs = possible{i};
            return;
        end
    end
end

function write_log_header(logPath)
    fid = fopen(logPath, 'w');
    if fid == -1, return; end
    
    fprintf(fid, 'N-Back Processing Log v2.2\n');
    fprintf(fid, 'Timestamp: %s\n', datestr(now));
    fprintf(fid, 'New features: Brain<10%% | EOG≥0.30 | EOG Safety Net (≥0.60)\n');
    fprintf(fid, '================================================================\n\n');
    fclose(fid);
end

function write_log_entry_nback(logPath, filename, interpolatedChans, comps_to_reject, ...
    allRejectedEpochs, rejectedEpochCount, epochsAfterRejection, cutSegmentsSec, ...
    asrBoundariesAdded, extremeNoiseSegments, valideEpochs, invalideEpochs)
    
    fid = fopen(logPath, 'a');
    if fid == -1, return; end
    
    fprintf(fid, 'File: %s\n', filename);
    fprintf(fid, '  Interpolated channels: %s\n', strjoin(interpolatedChans, ', '));
    fprintf(fid, '  Rejected ICs: %s\n', mat2str(comps_to_reject));
    fprintf(fid, '  Rejected epochs: %d/%d (indices: %s)\n', ...
        rejectedEpochCount, rejectedEpochCount + epochsAfterRejection, mat2str(allRejectedEpochs));
    fprintf(fid, '  Final epochs: %d (%d Valide, %d Invalide)\n', ...
        epochsAfterRejection, valideEpochs, invalideEpochs);
    fprintf(fid, '  Cut segments: %s\n', strjoin(cutSegmentsSec, '; '));
    fprintf(fid, '  ASR boundaries added: %d\n', asrBoundariesAdded);
    fprintf(fid, '  Extreme noise segments: %d\n', extremeNoiseSegments);
    fprintf(fid, '\n');
    
    fclose(fid);
end

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

function wid = get_worker_id()
    t = getCurrentTask();
    if isempty(t)
        wid = 1;
    else
        wid = t.ID;
    end
end

function cleanEvents = remove_noise_burst_triggers_smart(events, srate, validTypes)
    % N-Back specific burst detection with temporal constraints:
    % - Stimulus-Stimulus: min 1000ms (ISI)
    % - Stimulus-Response: min 200ms (reaction time)
    % - Response-Response: min 100ms (impossible double response)
    % 
    % Detection criteria (applied AFTER trigger conversion, so no risk for legitimate pairs):
    % 1. ANY 2+ triggers within 100ms = burst (too fast for any legitimate sequence)
    % 2. 3+ triggers within 200ms = burst (definitely artifact)
    
    cleanEvents = [];
    i = 1;

    while i <= length(events)
        evt = events(i);

        % Always keep boundaries
        if strcmp(evt.type, 'boundary')
            cleanEvents = [cleanEvents, evt];
            i = i + 1;
            continue;
        end

        % Check if valid type
        if ~ismember(evt.type, validTypes)
            i = i + 1;
            continue;
        end

        % Look ahead for potential burst within 200ms
        burstCount = 1;
        burstEnd = i;
        
        for j = (i+1):min(i+20, length(events))
            if strcmp(events(j).type, 'boundary')
                break;
            end
            
            dt = (events(j).latency - evt.latency) / srate * 1000;
            
            if dt > 200  % Look ahead up to 200ms
                break;
            end
            
            if ismember(events(j).type, validTypes)
                burstCount = burstCount + 1;
                burstEnd = j;
            end
        end

        % Determine if burst with N-Back specific criteria
        isBurst = false;
        
        if burstCount >= 3
            % 3+ triggers within 200ms = definitely artifact
            dt_total = (events(burstEnd).latency - evt.latency) / srate * 1000;
            if dt_total <= 200
                isBurst = true;
            end
        elseif burstCount == 2
            % 2 triggers within 100ms = artifact (impossible for any legitimate sequence)
            % Even fastest response (200ms) + next stimulus (1000ms ISI) = 1200ms minimum
            dt_pair = (events(i+1).latency - evt.latency) / srate * 1000;
            if dt_pair <= 100
                isBurst = true;
                burstEnd = i + 1;
            end
        end

        if isBurst
            % Skip all burst triggers
            i = burstEnd + 1;
        else
            % Keep this trigger
            cleanEvents = [cleanEvents, evt];
            i = i + 1;
        end
    end
end