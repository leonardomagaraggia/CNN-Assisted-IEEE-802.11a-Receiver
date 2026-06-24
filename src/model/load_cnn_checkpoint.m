function [cnn, modelPath, selection] = load_cnn_checkpoint( ...
        baseParams, requestedPath, checkpointDir, allowFallback)
% Carica un checkpoint compatibile; se richiesto, ripiega sul piu recente.

    paths = project_paths(false);
    projectRoot = paths.Root;
    if nargin < 2 || isempty(requestedPath)
        requestedPath = "";
    end
    if nargin < 3 || isempty(checkpointDir)
        checkpointDir = paths.Checkpoints;
    end
    if nargin < 4 || isempty(allowFallback)
        allowFallback = true;
    end

    requestedPath = string(requestedPath);
    checkpointDir = string(checkpointDir);
    if ~isscalar(requestedPath) || ~isscalar(checkpointDir)
        error('requestedPath e checkpointDir devono essere stringhe scalari.');
    end

    candidates = strings(0, 1);
    if strlength(requestedPath) > 0
        requestedCandidates = requested_path_candidates( ...
            requestedPath, projectRoot, checkpointDir);
        candidates = [candidates; requestedCandidates];
    end
    if allowFallback
        candidates = [candidates; list_checkpoints_newest_first(checkpointDir)];
    end
    candidates = unique(candidates, 'stable');

    failures = strings(0, 1);
    for candidateIdx = 1:numel(candidates)
        candidate = candidates(candidateIdx);
        if ~isfile(candidate)
            failures(end+1, 1) = candidate + ": file inesistente"; %#ok<AGROW>
            continue;
        end

        try
            loaded = load(candidate, 'net');
            if ~isfield(loaded, 'net') || ~isa(loaded.net, 'dlnetwork')
                error('la variabile net manca o non e un dlnetwork');
            end

            candidateCNN = LOS_CNN( ...
                'NFFT', baseParams.FFTLength, ...
                'ModOrder', baseParams.ModulationOrder, ...
                'NumInputChannels', baseParams.NumCNNInputChannels);
            candidateCNN.Net = loaded.net;
            validate_network_shape(candidateCNN);

            cnn = candidateCNN;
            modelPath = candidate;
            selection = struct();
            selection.RequestedPath = requestedPath;
            selection.UsedFallback = strlength(requestedPath) == 0 || ...
                ~any(candidate == requested_path_candidates( ...
                requestedPath, projectRoot, checkpointDir));
            selection.Failures = failures;

            if selection.UsedFallback && strlength(requestedPath) > 0
                warning('Checkpoint richiesto non valido. Uso il piu recente compatibile: %s', ...
                    char(modelPath));
            end
            return;
        catch exception
            failures(end+1, 1) = candidate + ": " + string(exception.message); %#ok<AGROW>
        end
    end

    if allowFallback
        error('Nessun checkpoint CNN compatibile trovato in %s. Dettagli: %s', ...
            char(checkpointDir), char(strjoin(failures, ' | ')));
    end
    error('Checkpoint CNN non valido: %s. Dettagli: %s', ...
        char(requestedPath), char(strjoin(failures, ' | ')));
end

function candidates = requested_path_candidates(requestedPath, projectRoot, checkpointDir)
    if strlength(requestedPath) == 0
        candidates = strings(0, 1);
        return;
    end

    candidates = requestedPath;
    if ~is_absolute_path(requestedPath)
        candidates(end+1, 1) = string(fullfile(projectRoot, requestedPath));
        candidates(end+1, 1) = string(fullfile(checkpointDir, requestedPath));
    end
    candidates = unique(candidates, 'stable');
end

function checkpoints = list_checkpoints_newest_first(checkpointDir)
    if ~isfolder(checkpointDir)
        checkpoints = strings(0, 1);
        return;
    end

    files = dir(fullfile(checkpointDir, '*.mat'));
    if isempty(files)
        checkpoints = strings(0, 1);
        return;
    end
    [~, order] = sort([files.datenum], 'descend');
    files = files(order);
    checkpoints = strings(numel(files), 1);
    for fileIdx = 1:numel(files)
        checkpoints(fileIdx) = string(fullfile(files(fileIdx).folder, files(fileIdx).name));
    end
end

function tf = is_absolute_path(pathValue)
    pathText = char(pathValue);
    tf = startsWith(pathText, filesep) || ...
        (~isempty(regexp(pathText, '^[A-Za-z]:[\\/]', 'once')));
end

function validate_network_shape(cnn)
    sample = zeros(cnn.NumInputChannels, cnn.NFFT, 1, 'single');
    scores = gather(extractdata(forward(cnn.Net, dlarray(sample, 'CTB'))));
    scoreSize = size(scores);
    if numel(scoreSize) < 2 || scoreSize(1) ~= cnn.ModOrder || ...
            ~any(scoreSize(2:end) == cnn.NFFT)
        error('output CNN incompatibile: [%s]', num2str(scoreSize));
    end
end
