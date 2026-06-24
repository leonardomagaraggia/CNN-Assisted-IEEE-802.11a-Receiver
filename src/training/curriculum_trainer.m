% Addestra LOS_CNN con curriculum 4-QAM IEEE 802.11a Rician-only.

clc; close all; rng(172);

paths = project_paths(true);

fprintf("=== LOS CNN TRAINER 4-QAM IEEE 802.11a RICIAN ===\n");

%% Parametri base

baseParams = simulation_parameters();
baseParams = baseParams.refreshDerived();

if baseParams.ModulationOrder ~= 4
    error("Il curriculum e vincolato a segnali 4-QAM.");
end

carrierMap = baseParams.getCarrierMap();
NC = carrierMap.NumSubcarriers;
NActive = carrierMap.NumActiveCarriers;
NS = baseParams.NumOFDMSymbols;

cnn = LOS_CNN( ...
    'NFFT', NC, ...
    'ModOrder', baseParams.ModulationOrder, ...
    'NumInputChannels', baseParams.NumCNNInputChannels, ...
    'MiniBatchSize', baseParams.TrainingMiniBatchSize, ...
    'MaxEpochs', 1, ...
    'InitialLearnRate', 1e-3);

useGPU = false;

frameTool = IEEE80211aFrame(baseParams);

%% Piloti

pilotActiveIdx = carrierMap.PilotActiveIdx;
pilotIdx = carrierMap.PilotGlobalIdx;
dataIdx = carrierMap.PayloadGlobalIdx;

pilotClass = baseParams.PilotClass;
fprintf("Sottoportanti FFT: %d | Carrier OFDM utili: %d | Payload data: %d | Piloti: %d\n", ...
    NC, NActive, numel(dataIdx), numel(pilotIdx));
fprintf("Piloti IEEE 802.11a subcarrier: %s | active-index: %s | global-index: %s\n", ...
    mat2str(carrierMap.PilotSubcarrierIdx), mat2str(pilotActiveIdx), mat2str(pilotIdx));
fprintf("Input CNN: %d canali pilot-aware | MiniBatchSize: %d | Esecuzione: CPU/RAM\n", ...
    cnn.NumInputChannels, cnn.MiniBatchSize);
fprintf("Frame: L-STF + L-LTF + %d simboli QPSK uncoded dinamici | preambolo %d campioni\n", ...
    NS, frameTool.PreambleLength);
print_ofdm_channel_report(baseParams, "BASE");

legacyCheckpoint = latest_phase08_checkpoint(paths.Checkpoints);
if strlength(legacyCheckpoint) > 0
    warmStart = load(legacyCheckpoint, 'net');
    cnn.Net = warmStart.net;
    fprintf("Warm start dai pesi esistenti: %s\n", legacyCheckpoint);
end

%% Curriculum

phases = build_curriculum();
validate_curriculum(phases);

%% Training

avgGrad = [];
avgSqGrad = [];
iter = 0;
checkpointDir = paths.Checkpoints;
if ~exist(checkpointDir, 'dir')
    mkdir(checkpointDir);
end

for p = 1:numel(phases)
    ph = phases(p);
    phaseEpochs = ph.epochs;
    phaseLearnRate = ph.learnRate;
    iterationsPerEpoch = ceil(ph.n / cnn.MiniBatchSize);
    nearZeroEpochs = 0;
    completedEpochs = 0;
    stopReason = "max_epochs";
    lastEpochLoss = Inf;

    fprintf("\n--- Macrofase %d/%d: %s ---\n", p, numel(phases), char(ph.name));
    fprintf("%s\n", describe_phase(ph));
    validate_phase_ofdm(baseParams, ph);

    for ep = 1:phaseEpochs
        lossEp = 0;
        seen = 0;

        for b = 1:iterationsPerEpoch
            iter = iter + 1;
            batchCount = min(cnn.MiniBatchSize, ph.n - (b-1) * cnn.MiniBatchSize);

            [X, T] = generate_batch_integrated(cnn, baseParams, frameTool, ph, ...
                batchCount, carrierMap, pilotClass);
            X = cnn.normalizeInputBatch(X, carrierMap.ActiveGlobalIdx);
            T = cnn.normalizeTargetShape(T);

            dlX = make_dlarray(X, useGPU);
            dlT = make_dlarray(T, useGPU);

            [loss, grad] = dlfeval(@LOS_CNN.modelGradients, cnn.Net, dlX, dlT, dataIdx);

            [cnn.Net, avgGrad, avgSqGrad] = adamupdate( ...
                cnn.Net, grad, avgGrad, avgSqGrad, iter, phaseLearnRate);

            lossEp = lossEp + gather_scalar(loss) * batchCount;
            seen = seen + batchCount;
        end

        completedEpochs = ep;
        lastEpochLoss = lossEp / max(seen, 1);
        fprintf("Epoca %d/%d | Esempi: %d | Loss data media: %.6f\n", ...
            ep, phaseEpochs, seen, lastEpochLoss);

        if ep >= ph.minEpochs && lastEpochLoss <= ph.earlyStopLoss
            nearZeroEpochs = nearZeroEpochs + 1;
        else
            nearZeroEpochs = 0;
        end

        if nearZeroEpochs >= ph.earlyStopPatience
            stopReason = "near_zero_loss";
            fprintf("Early stop fase: loss <= %.2e per %d epoche consecutive dopo almeno %d epoche.\n", ...
                ph.earlyStopLoss, ph.earlyStopPatience, ph.minEpochs);
            break;
        end
    end

    ph.completedEpochs = completedEpochs;
    ph.lastEpochLoss = lastEpochLoss;
    ph.stopReason = stopReason;

    phTrainEval = ph;
    phTrainEval.n = ph.nTrainEval;
    [XTrainEval, TTrainEval] = generate_dataset_integrated(cnn, baseParams, frameTool, ...
        phTrainEval, carrierMap, pilotClass);
    XTrainEval = cnn.normalizeInputBatch(XTrainEval, carrierMap.ActiveGlobalIdx);
    trainMetrics = evaluate_dataset(cnn, XTrainEval, TTrainEval, dataIdx, useGPU);
    print_metrics(sprintf("Macrofase %d TRAIN-SAMPLE", p), trainMetrics);

    phVal = ph;
    phVal.n = ph.nVal;
    [XVal, TVal] = generate_dataset_integrated(cnn, baseParams, frameTool, phVal, ...
        carrierMap, pilotClass);
    XVal = cnn.normalizeInputBatch(XVal, carrierMap.ActiveGlobalIdx);
    valMetrics = evaluate_dataset(cnn, XVal, TVal, dataIdx, useGPU);
    print_metrics(sprintf("Macrofase %d VAL", p), valMetrics);

    savePath = save_phase_checkpoint(checkpointDir, p, ph, cnn, ...
        trainMetrics, valMetrics, carrierMap, avgGrad, avgSqGrad, ...
        iter, baseParams);
    fprintf("Checkpoint salvato: %s\n", savePath);
end

disp("ADDESTRAMENTO COMPLETATO.");

%% Funzioni di supporto

function phases = build_curriculum()
% Ogni fase usa lo stesso numero di esempi per epoca. Il generatore
% bilancia separatamente strati SNR, strati K e profili multipath: le
% numerosita differiscono al massimo di un frame all'interno di un batch.
examplesPerEpoch = 165536;

phases = make_phase( ...
    "MACRO_01_RICIAN_CONSTELLATION_LOCK", ...
    "Aggancio costellazione ad alto SNR con fase e guadagno LOS variabili", ...
    [36 40; 40 45; 45 50], ...
    [50 80; 80 120; 120 160], ...
    [0 2], path_profiles("flat_mild"), ...
    examplesPerEpoch, 12, 4, 8e-4, 5e-4, 2);

phases(end+1) = make_phase( ...
    "MACRO_02_RICIAN_HIGH_SNR_SELECTIVE", ...
    "Alta SNR con crescente selettivita: denoising della stima L-LTF", ...
    [28 32; 32 36; 36 40; 40 44], ...
    [20 35; 35 60; 60 100], ...
    [0 12], path_profiles("mild_medium"), ...
    examplesPerEpoch, 18, 6, 5e-4, 5e-4, 2);

phases(end+1) = make_phase( ...
    "MACRO_03_RICIAN_HIGH_SNR_CHANNEL_INVERSION", ...
    "Inversione del canale tra 22 e 32 dB con profili LOS mild/medium", ...
    [22 24; 24 26; 26 28; 28 30; 30 32], ...
    [15 25; 25 45; 45 80], ...
    [7 25], path_profiles("mild_medium"), ...
    examplesPerEpoch, 35, 8, 3e-4, 4e-4, 2);

phases(end+1) = make_phase( ...
    "MACRO_04_RICIAN_TARGET_16_26_DB", ...
    "Addestramento intensivo nella regione dove il BER e basso ma misurabile", ...
    [18 20; 20 22; 22 24; 24 26; 26 30], ...
    [12 20; 20 35; 35 60], ...
    [0 40], [path_profiles("mild"), path_profiles("medium")], ...
    examplesPerEpoch, 50, 10, 2e-4, 3e-4, 3);

% phases(end+1) = make_phase( ...
%     "MACRO_05_RICIAN_BOUNDARY_13_20_DB", ...
%     "Copertura fitta e bilanciata attorno alla soglia osservata di 13 dB", ...
%     [15 16; 16 17; 17 18; 18 20], ...
%     [10 18; 18 30; 30 50], ...
%     [0 60], [path_profiles("mild_medium"), path_profiles("medium")], ...
%     examplesPerEpoch, 20, 12, 1.4e-4, 3e-4, 3);
% 
% phases(end+1) = make_phase( ...
%     "MACRO_06_RICIAN_HIGH_SNR_HARD_CHANNELS", ...
%     "SNR 14-36 dB con basso K e notch profondi per mantenere gradienti utili", ...
%     [16 18; 18 22; 22 26; 26 31; 31 36], ...
%     [6 12; 12 20; 20 35], ...
%     [0 80], path_profiles("medium_deep"), ...
%     examplesPerEpoch, 20, 14, 9e-5, 2e-4, 3);
% 
% phases(end+1) = make_phase( ...
%     "MACRO_07_RICIAN_BALANCED_REPLAY", ...
%     "Replay globale bilanciato con il 75 percento degli strati sopra 13 dB", ...
%     [13 16; 16 20; 20 24; 24 30; 30 36; 36 44], ...
%     [6 12; 12 25; 25 50; 50 100], ...
%     [0 140], path_profiles("all"), ...
%     examplesPerEpoch, 50, 12, 6e-5, 2e-4, 3);
% 
% phases(end+1) = make_phase( ...
%     "MACRO_08_RICIAN_HIGH_SNR_FINE_TUNE", ...
%     "Fine tuning finale esclusivamente sopra 13 dB, bilanciato per SNR/K/profilo", ...
%     [15 18; 18 22; 22 26; 26 31; 31 37; 37 44], ...
%     [8 15; 15 30; 30 60; 60 100], ...
%     [0 100], path_profiles("all"), ...
%     examplesPerEpoch, 32, 16, 3e-5, 1e-4, 4);
end

function ph = make_phase(name, macroPhase, snrStrata, kFactorStrata, ...
    dopplerRange, profiles, nExamples, epochs, minEpochs, learnRate, ...
    earlyStopLoss, earlyStopPatience)

    ph = struct();
    ph.name = string(name);
    ph.macroPhase = string(macroPhase);
    ph.channel = "Rician";
    ph.snrStrata = double(snrStrata);
    ph.kFactorStrata = double(kFactorStrata);
    ph.snr = [min(ph.snrStrata(:, 1)), max(ph.snrStrata(:, 2))];
    ph.kFactor = [min(ph.kFactorStrata(:, 1)), ...
        max(ph.kFactorStrata(:, 2))];
    ph.doppler = dopplerRange;
    ph.pathProfiles = profiles;
    ph.n = nExamples;
    ph.nVal = min(12000, max(4096, round(0.02 * nExamples)));
    ph.nTrainEval = min(8192, max(2048, round(0.01 * nExamples)));
    ph.epochs = epochs;
    ph.minEpochs = minEpochs;
    ph.learnRate = learnRate;
    ph.earlyStopLoss = earlyStopLoss;
    ph.earlyStopPatience = earlyStopPatience;
    ph.hpaEnabled = false;
    ph.hpaProbability = 0;
end

function profiles = path_profiles(kind)
switch string(kind)
    case "flat_only"
        profiles = make_profile("flat_1tap", [1 1], [0 0], [-30 -30], 0);

    case "flat_mild"
        profiles = [ ...
            make_profile("flat_1tap", [1 1], [0 0], [-30 -30], 0), ...
            make_profile("mild_2_3tap", [2 3], [10e-9 40e-9], [-30 -14], 5e-9)];

    case "mild"
        profiles = [ ...
            make_profile("mild_2_3tap", [2 3], [25e-9 70e-9], [-28 -12], 8e-9), ...
            make_profile("mild_3_4tap", [3 4], [50e-9 90e-9], [-30 -13], 10e-9)];

    case "mild_medium"
        profiles = [ ...
            make_profile("mild_medium_3_4tap", [3 4], [70e-9 120e-9], [-28 -10], 12e-9), ...
            make_profile("medium_sparse_3_5tap", [3 5], [90e-9 140e-9], [-30 -12], 12e-9)];

    case "medium"
        profiles = [ ...
            make_profile("medium_3_5tap", [3 5], [110e-9 160e-9], [-30 -11], 15e-9), ...
            make_profile("medium_long_4_6tap", [4 6], [130e-9 180e-9], [-32 -13], 15e-9)];

    case "medium_deep"
        profiles = [ ...
            make_profile("medium_deep_4_6tap", [4 6], [150e-9 190e-9], [-34 -14], 15e-9), ...
            make_profile("low_k_sparse_3_6tap", [3 6], [140e-9 200e-9], [-36 -15], 15e-9), ...
            make_profile("late_weak_tail_5_7tap", [5 7], [160e-9 210e-9], [-38 -17], 15e-9)];

    case "deep"
        % Il limite fisico assoluto di Nyquist per i 4 piloti dell'802.11a è ~228 ns.
        % Limitiamo i casi peggiori a 220-225 ns per garantire coerenza nell'equalizzazione SDR.
        profiles = [ ...
            make_profile("edge_weak_tail_5_7tap", [5 7], [180e-9 220e-9], [-40 -18], 15e-9), ...
            make_profile("edge_sparse_4_7tap", [4 7], [170e-9 225e-9], [-42 -20], 15e-9)];

    case "all"
        profiles = [ ...
            path_profiles("flat_only"), ...
            path_profiles("flat_mild"), ...
            path_profiles("mild"), ...
            path_profiles("mild_medium"), ...
            path_profiles("medium"), ...
            path_profiles("medium_deep"), ...
            path_profiles("deep")];
        [~, uniqueIdx] = unique([profiles.name], 'stable');
        profiles = profiles(uniqueIdx);

    otherwise
        error("Profilo multipath non supportato: %s", kind);
end
end

function profile = make_profile(name, numPathsRange, maxExcessDelayRange, ...
    relativeGainRange, minTapSpacing)

    profile = struct();
    profile.name = string(name);
    profile.numPathsRange = double(numPathsRange);
    profile.maxExcessDelayRange = double(maxExcessDelayRange);
    profile.relativeGainRange = double(relativeGainRange);
    profile.minTapSpacing = double(minTapSpacing);
end

function [X, T] = generate_dataset_integrated(cnn, baseParams, frameTool, ph, ...
    carrierMap, pilotClass)

    [X, T] = generate_batch_integrated(cnn, baseParams, frameTool, ph, ph.n, ...
        carrierMap, pilotClass);
end

function [X, T] = generate_batch_integrated(cnn, baseParams, frameTool, ph, ...
    batchCount, carrierMap, pilotClass)

    NC = carrierMap.NumSubcarriers;
    dataGlobalIdx = carrierMap.PayloadGlobalIdx;

    X = zeros(cnn.NumInputChannels, NC, batchCount, 'single');
    allLabels = repmat(uint16(pilotClass), NC, batchCount);

    examplesPerFrame = min(baseParams.NumTrainingSymbolsPerFrame, ...
        baseParams.NumOFDMSymbols);
    framesPerBatch = ceil(batchCount / examplesPerFrame);
    snrStratumSchedule = balanced_schedule( ...
        size(ph.snrStrata, 1), framesPerBatch);
    kStratumSchedule = balanced_schedule( ...
        size(ph.kFactorStrata, 1), framesPerBatch);
    profileSchedule = balanced_schedule( ...
        numel(ph.pathProfiles), framesPerBatch);

    sampleIdx = 0;
    frameIdx = 0;
    while sampleIdx < batchCount
        frameIdx = frameIdx + 1;
        frameParams = sample_frame_params(baseParams, ph, ...
            snrStratumSchedule(frameIdx), ...
            kStratumSchedule(frameIdx), ...
            profileSchedule(frameIdx));
        [tx, txFrame] = frameTool.createRandomFrame();
        rx = multipath_channel(tx, frameParams);
        rx = rx .* exp(1i * 2 * pi * rand());
        rxFrame = frameTool.receive(rx);

        symbolsPerFrame = min([baseParams.NumTrainingSymbolsPerFrame, ...
            baseParams.NumOFDMSymbols, batchCount - sampleIdx]);
        symbolOrder = randperm(baseParams.NumOFDMSymbols, symbolsPerFrame);
        cnnChannelEstimate = regularize_ltf_channel_estimate( ...
            rxFrame.ChannelEstimate, carrierMap, baseParams.FFTLength, 8);

        for symbolIdx = symbolOrder
            sampleIdx = sampleIdx + 1;
            X(:, :, sampleIdx) = build_ofdm_cnn_features( ...
                rxFrame.ActiveGrid(:, symbolIdx), carrierMap, ...
                txFrame.PilotSymbols(:, symbolIdx), cnn.NumInputChannels, ...
                cnnChannelEstimate);

            dataLabels = cnn.symbolsToClasses( ...
                txFrame.PayloadSymbols(:, symbolIdx));
            fullLabels = repmat(uint16(pilotClass), NC, 1);
            fullLabels(dataGlobalIdx) = uint16(dataLabels);
            allLabels(:, sampleIdx) = fullLabels;
        end
    end

    T = cnn.classesToOneHot(allLabels);
end

function frameParams = sample_frame_params(baseParams, ph, ...
        snrStratumIdx, kStratumIdx, profileIdx)
    if ~strcmpi(char(ph.channel), 'Rician')
        error('Il trainer supporta solo fasi Rician. Fase %s: %s.', ...
            char(ph.name), char(ph.channel));
    end

    frameParams = baseParams;
    frameParams.ChannelType = 'Rician';
    frameParams.SNR_dB = sample_range(ph.snrStrata(snrStratumIdx, :));
    frameParams.KFactor = sample_range( ...
        ph.kFactorStrata(kStratumIdx, :));
    frameParams.MaximumDopplerShift = sample_range(ph.doppler);

    [pathDelays, averagePathGains] = sample_path_profile( ...
        baseParams, ph, profileIdx);
    frameParams.PathDelays = pathDelays;
    frameParams.AveragePathGains = averagePathGains;
    frameParams = frameParams.refreshDerived();
end

function [pathDelays, averagePathGains] = sample_path_profile( ...
        baseParams, ph, idx)
    [pathDelays, averagePathGains] = ...
        sample_dynamic_path_profile(baseParams, ph.pathProfiles(idx));
end

function schedule = balanced_schedule(numSituations, scheduleLength)
    if numSituations < 1
        error('Il numero di situazioni bilanciate deve essere positivo.');
    end
    schedule = repmat(1:numSituations, 1, ...
        ceil(scheduleLength / numSituations));
    schedule = schedule(1:scheduleLength);
    schedule = schedule(randperm(scheduleLength));
end

function [pathDelays, averagePathGains] = sample_dynamic_path_profile(baseParams, profile)
    params = baseParams.refreshDerived();
    for attempt = 1:80
        [pathDelays, averagePathGains] = propose_path_profile(params, profile);
        if is_path_profile_admissible(params, pathDelays, averagePathGains)
            return;
        end
    end

    [pathDelays, averagePathGains] = conservative_path_profile(params, profile);
end

function [pathDelays, averagePathGains] = propose_path_profile(params, profile)
    numPaths = randi([profile.numPathsRange(1), profile.numPathsRange(end)]);
    if numPaths <= 1 || profile.maxExcessDelayRange(end) <= 0
        pathDelays = 0;
        averagePathGains = 0;
        return;
    end

    maxDelayAllowed = params.CPDuration * params.MaxExcessDelayOverCPLimit;
    delayHi = min(profile.maxExcessDelayRange(end), maxDelayAllowed);
    delayLo = min(profile.maxExcessDelayRange(1), delayHi);
    maxExcessDelay = sample_range([delayLo delayHi]);

    rawDelays = sort((rand(1, numPaths - 1) .^ 1.25) * maxExcessDelay);
    minSpacing = min(profile.minTapSpacing, ...
        maxExcessDelay / max(numPaths - 1, 1) * 0.35);
    if minSpacing > 0
        rawDelays = max(rawDelays, (1:numPaths - 1) * minSpacing);
        if rawDelays(end) > maxExcessDelay
            rawDelays = rawDelays ./ rawDelays(end) .* maxExcessDelay;
        end
    end
    pathDelays = [0 rawDelays];

    tailGain = sample_range(profile.relativeGainRange);
    decay = linspace(0, abs(tailGain), numPaths);
    jitterLimit = min(3, max(abs(tailGain) / max(numPaths, 1), 0.5));
    jitter = (2 * rand(1, numPaths) - 1) * jitterLimit;
    averagePathGains = -decay + jitter;
    averagePathGains(1) = 0;
    averagePathGains(2:end) = min(averagePathGains(2:end), ...
        -3 - 2 * rand(1, numPaths - 1));
    averagePathGains(2:end) = sort(averagePathGains(2:end), 'descend');
end

function tf = is_path_profile_admissible(baseParams, pathDelays, averagePathGains)
    checkParams = baseParams;
    checkParams.PathDelays = pathDelays;
    checkParams.AveragePathGains = averagePathGains;
    checkParams = checkParams.refreshDerived();
    report = checkParams.ofdmChannelReport();
    tf = report.IsValid;
end

function [pathDelays, averagePathGains] = conservative_path_profile(params, profile)
    numPaths = min(max(profile.numPathsRange(1), 1), min(profile.numPathsRange(end), 4));
    if numPaths <= 1 || profile.maxExcessDelayRange(end) <= 0
        pathDelays = 0;
        averagePathGains = 0;
        return;
    end

    maxDelay = min([profile.maxExcessDelayRange(end), ...
        params.CPDuration * 0.55, 320e-9]);
    tailGain = min(profile.relativeGainRange(1), -24);

    for attempt = 1:20
        pathDelays = linspace(0, maxDelay, numPaths);
        averagePathGains = linspace(0, tailGain, numPaths);
        if is_path_profile_admissible(params, pathDelays, averagePathGains)
            return;
        end
        maxDelay = maxDelay * 0.70;
        tailGain = tailGain - 3;
    end

    error('Impossibile generare un profilo multipath ammissibile per %s.', ...
        char(profile.name));
end

function metrics = evaluate_dataset(cnn, X, T, dataIdx, useGPU)
    X = cnn.normalizeInputShape(X);
    T = cnn.normalizeTargetShape(T);

    N = size(X, 3);
    nb = ceil(N / cnn.MiniBatchSize);

    lossTotal = 0;
    seen = 0;
    correctSymbols = 0;
    totalSymbols = 0;
    bitErrors = 0;
    totalBits = 0;

    for b = 1:nb
        i1 = (b-1) * cnn.MiniBatchSize + 1;
        i2 = min(b * cnn.MiniBatchSize, N);
        batchCount = i2 - i1 + 1;

        dlX = make_dlarray(X(:, :, i1:i2), useGPU);
        dlT = make_dlarray(T(:, :, i1:i2), useGPU);
        dlY = forward(cnn.Net, dlX);

        batchLoss = LOS_CNN.crossEntropyLoss(dlY, dlT, dataIdx);
        lossTotal = lossTotal + gather_scalar(batchLoss) * batchCount;
        seen = seen + batchCount;

        trueClasses = cnn.oneHotToClasses(T(:, :, i1:i2));
        predScores = normalize_score_shape(gather(extractdata(dlY)), cnn, batchCount);
        predClasses = cnn.oneHotToClasses(predScores);

        trueData = trueClasses(dataIdx, :);
        predData = predClasses(dataIdx, :);

        correctSymbols = correctSymbols + sum(trueData(:) == predData(:));
        totalSymbols = totalSymbols + numel(trueData);

        batchBits = cnn.classesToBits(trueData);
        bitErrors = bitErrors + cnn.computeBER(trueData, predData) * numel(batchBits);
        totalBits = totalBits + numel(batchBits);
    end

    metrics.loss = lossTotal / max(seen, 1);
    metrics.accuracy = correctSymbols / max(totalSymbols, 1);
    metrics.berData = bitErrors / max(totalBits, 1);
end

function print_metrics(label, metrics)
    fprintf("%s | Loss data: %.4f | Accuracy data: %.2f%% | BER_data: %.4e\n", ...
        label, metrics.loss, 100 * metrics.accuracy, metrics.berData);
end

function txt = describe_phase(ph)
    profileNames = strjoin([ph.pathProfiles.name], ",");
    snrStrataText = strata_to_text(ph.snrStrata);
    kStrataText = strata_to_text(ph.kFactorStrata);
    txt = sprintf('%s | Canale=%s | SNR bilanciati=%s dB | K bilanciati=%s | Doppler %.1f..%.1f Hz | Path bilanciati=%s | N/epoca=%d | Epoche min/max=%d/%d | EarlyStop loss<=%.1e x%d | LR=%.1e | HPA off', ...
        ph.macroPhase, ph.channel, snrStrataText, kStrataText, ...
        ph.doppler(1), ph.doppler(end), profileNames, ph.n, ...
        ph.minEpochs, ph.epochs, ph.earlyStopLoss, ...
        ph.earlyStopPatience, ph.learnRate);
end

function txt = strata_to_text(strata)
    pieces = strings(size(strata, 1), 1);
    for idx = 1:size(strata, 1)
        pieces(idx) = sprintf('%.1f-%.1f', strata(idx, 1), strata(idx, 2));
    end
    txt = strjoin(pieces, ",");
end

function validate_curriculum(phases)
    expectedExamples = [phases.n];
    if any(expectedExamples ~= expectedExamples(1))
        error('Tutte le fasi devono usare lo stesso numero di esempi per epoca.');
    end

    for idx = 1:numel(phases)
        ph = phases(idx);
        validate_strata(ph.snrStrata, sprintf('%s SNR', ph.name));
        validate_strata(ph.kFactorStrata, sprintf('%s K', ph.name));
        if ph.minEpochs < 1 || ph.minEpochs > ph.epochs
            error('%s: minEpochs deve essere tra 1 ed epochs.', ph.name);
        end
        if ph.earlyStopLoss <= 0 || ph.earlyStopPatience < 1
            error('%s: configurazione early stopping non valida.', ph.name);
        end
    end
end

function validate_strata(strata, label)
    if size(strata, 2) ~= 2 || any(strata(:, 2) < strata(:, 1))
        error('%s: gli strati devono essere intervalli [min max] validi.', label);
    end
    if size(strata, 1) > 1 && any(strata(2:end, 1) < strata(1:end-1, 2))
        error('%s: gli strati non devono sovrapporsi.', label);
    end
end

function validate_phase_ofdm(baseParams, ph)
    worstParams = baseParams;
    worstParams.ChannelType = char(ph.channel);
    worstParams.SNR_dB = ph.snr(1);
    worstParams.KFactor = ph.kFactor(1);
    worstParams.MaximumDopplerShift = ph.doppler(end);

    for k = 1:numel(ph.pathProfiles)
        for trial = 1:24
            [pathDelays, averagePathGains] = ...
                sample_dynamic_path_profile(baseParams, ph.pathProfiles(k));
            worstParams.PathDelays = pathDelays;
            worstParams.AveragePathGains = averagePathGains;
            worstParams = worstParams.refreshDerived();
            label = sprintf('%s profilo %s trial %d', ...
                ph.name, ph.pathProfiles(k).name, trial);
            worstParams.assertOFDMChannelConsistency(label);
        end
    end

    print_ofdm_channel_report(worstParams, ph.name);
end

function print_ofdm_channel_report(params, label)
    report = params.ofdmChannelReport();
    fprintf("OFDM %s | df %.2f kHz | T_OFDM %.3f us | Tc %.3f ms | Bc %.2f kHz | RMS %.1f ns | tauMax %.1f ns | Tcoh/Tsym %.1f | df/Bc %.3f | tau/CP %.3f\n", ...
        char(label), report.SubcarrierBandwidthHz / 1e3, ...
        report.OFDMSymbolDuration * 1e6, report.CoherenceTime * 1e3, ...
        report.CoherenceBandwidthHz / 1e3, report.RMSDelaySpread * 1e9, ...
        report.MaxExcessDelay * 1e9, report.OFDMSymbolsPerCoherenceTime, ...
        report.SubcarrierSpacingOverCoherenceBandwidth, report.MaxExcessDelayOverCP);

    if ~report.IsValid
        for k = 1:numel(report.Issues)
            fprintf("  AVVISO OFDM: %s\n", report.Issues(k));
        end
    end
end

function value = sample_range(rangeValue)
    if isscalar(rangeValue)
        value = rangeValue;
    else
        value = rangeValue(1) + rand() * (rangeValue(end) - rangeValue(1));
    end
end

function savePath = save_phase_checkpoint(checkpointDir, phaseNumber, ph, cnn, ...
    trainMetrics, valMetrics, carrierMap, avgGrad, avgSqGrad, iter, baseParams)

    timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss_SSS"));
    net = gather_network(cnn.Net);
    learnables = net.Learnables;

    optimizerState = struct();
    optimizerState.avgGrad = gather_state(avgGrad);
    optimizerState.avgSqGrad = gather_state(avgSqGrad);
    optimizerState.iteration = iter;

    trainingState = struct();
    trainingState.timestamp = timestamp;
    trainingState.NFFT = cnn.NFFT;
    trainingState.ModOrder = cnn.ModOrder;
    trainingState.NumInputChannels = cnn.NumInputChannels;
    trainingState.MiniBatchSize = cnn.MiniBatchSize;
    trainingState.networkArchitecture = LOS_CNN.architectureVersion();
    trainingState.phaseNumber = phaseNumber;
    trainingState.phase = ph;
    trainingState.trainMetrics = trainMetrics;
    trainingState.valMetrics = valMetrics;
    trainingState.carrierMap = carrierMap;
    trainingState.activeCarrierIdx = carrierMap.ActiveGlobalIdx;
    trainingState.pilotActiveIdx = carrierMap.PilotActiveIdx;
    trainingState.dataActiveIdx = carrierMap.PayloadActiveIdx;
    trainingState.pilotIdx = carrierMap.PilotGlobalIdx;
    trainingState.dataIdx = carrierMap.PayloadGlobalIdx;
    trainingState.numActiveCarriers = carrierMap.NumActiveCarriers;
    trainingState.numPilots = numel(carrierMap.PilotGlobalIdx);
    trainingState.frameFormat = ...
        "legacy_80211a_LSTF_LLTF_QPSK_uncoded_dynamic_payload";
    trainingState.numTrainingSymbolsPerFrame = ...
        baseParams.NumTrainingSymbolsPerFrame;
    trainingState.RandomCarrierPhaseAugmentation = true;
    trainingState.parameters = baseParams.refreshDerived();
    trainingState.optimizerState = optimizerState;

    savePath = fullfile(checkpointDir, ...
        sprintf('ckpt_p%02d_%s.mat', phaseNumber, char(timestamp)));
    save(savePath, 'net', 'learnables', 'trainingState', ...
        'optimizerState', 'trainMetrics', 'valMetrics', 'ph', '-v7.3');
end

function checkpoint = latest_phase08_checkpoint(checkpointDir)
    checkpoint = "";
    if ~exist(checkpointDir, 'dir')
        return;
    end
    files = dir(fullfile(checkpointDir, 'ckpt_p08_*.mat'));
    if isempty(files)
        return;
    end
    [~, idx] = max([files.datenum]);
    checkpoint = string(fullfile(checkpointDir, files(idx).name));
end

function scores = normalize_score_shape(scores, cnn, batchCount)
    % Le metriche usano sempre score [M x NFFT x batch].
    if ismatrix(scores)
        scores = reshape(scores, size(scores, 1), size(scores, 2), 1);
    end

    if size(scores, 1) ~= cnn.ModOrder
        error('Output CNN non valido. Attesa prima dimensione M=%d.', cnn.ModOrder);
    end

    if size(scores, 2) == cnn.NFFT && size(scores, 3) == batchCount
        return;
    end

    if size(scores, 2) == batchCount && size(scores, 3) == cnn.NFFT
        scores = permute(scores, [1 3 2]);
        return;
    end

    error('Output CNN non valido. Atteso [M x NFFT x batch], trovato [%s].', ...
        num2str(size(scores)));
end

function dlX = make_dlarray(X, ~)
    X = single(X);
    dlX = dlarray(X, 'CTB');
end

function value = gather_scalar(dlValue)
    rawValue = extractdata(dlValue);
    try
        rawValue = gather(rawValue);
    catch
    end
    value = double(rawValue);
end

function net = gather_network(net)
    try
        net = dlupdate(@gather_value, net);
    catch
    end
end

function state = gather_state(state)
    if isempty(state)
        return;
    end

    if istable(state) && ismember('Value', state.Properties.VariableNames)
        for k = 1:height(state)
            state.Value{k} = gather_value(state.Value{k});
        end
        return;
    end

    try
        state = dlupdate(@gather_value, state);
    catch
    end
end

function value = gather_value(value)
    try
        value = gather(value);
    catch
    end
end
