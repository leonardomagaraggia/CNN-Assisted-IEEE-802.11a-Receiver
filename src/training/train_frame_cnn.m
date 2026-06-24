function modelPath = train_frame_cnn(varargin)
% Addestramento frame-aware mirato alla generalizzazione Rician/LOS.

    parser = inputParser();
    addParameter(parser, 'NumIterations', 400);
    addParameter(parser, 'MiniBatchSize', 256);
    addParameter(parser, 'InitialLearnRate', 8e-4);
    addParameter(parser, 'WarmStart', false);
    addParameter(parser, 'ValidationFrames', 80);
    addParameter(parser, 'SNRRangeDb', [0 35]);
    addParameter(parser, 'KFactorRange', [8 100]);
    parse(parser, varargin{:});
    cfg = parser.Results;

    paths = project_paths(true);
    rng(172);
    baseParams = simulation_parameters().refreshDerived();
    baseParams.TrainingMiniBatchSize = cfg.MiniBatchSize;
    carrierMap = baseParams.getCarrierMap();
    frameTool = IEEE80211aFrame(baseParams);

    cnn = LOS_CNN( ...
        'NFFT', baseParams.FFTLength, ...
        'ModOrder', baseParams.ModulationOrder, ...
        'NumInputChannels', baseParams.NumCNNInputChannels, ...
        'MiniBatchSize', cfg.MiniBatchSize);

    if cfg.WarmStart
        warmPath = latest_checkpoint(paths.Checkpoints);
        if strlength(warmPath) > 0
            loaded = load(warmPath, 'net');
            cnn.Net = loaded.net;
            fprintf("Warm start: %s\n", warmPath);
        end
    end

    avgGrad = [];
    avgSqGrad = [];
    lossHistory = zeros(cfg.NumIterations, 1);
    dataIdx = carrierMap.PayloadGlobalIdx;

    fprintf("Training frame-aware: %d iterazioni, batch %d, %d simboli/frame\n", ...
        cfg.NumIterations, cfg.MiniBatchSize, ...
        baseParams.NumTrainingSymbolsPerFrame);

    for iteration = 1:cfg.NumIterations
        [X, T] = generate_batch(cnn, baseParams, frameTool, carrierMap, ...
            cfg.MiniBatchSize, cfg.SNRRangeDb, cfg.KFactorRange);
        X = cnn.normalizeInputBatch(X, carrierMap.ActiveGlobalIdx);
        T = cnn.normalizeTargetShape(T);

        dlX = dlarray(single(X), 'CTB');
        dlT = dlarray(single(T), 'CTB');
        [loss, gradients] = dlfeval(@LOS_CNN.modelGradients, ...
            cnn.Net, dlX, dlT, dataIdx);

        progress = (iteration - 1) / max(cfg.NumIterations - 1, 1);
        learnRate = cfg.InitialLearnRate * (0.08 ^ progress);
        [cnn.Net, avgGrad, avgSqGrad] = adamupdate( ...
            cnn.Net, gradients, avgGrad, avgSqGrad, iteration, learnRate);
        lossHistory(iteration) = double(gather(extractdata(loss)));

        if iteration == 1 || mod(iteration, 20) == 0
            fprintf("Iter %4d/%4d | loss %.4f | LR %.2e\n", ...
                iteration, cfg.NumIterations, lossHistory(iteration), learnRate);
        end
    end

    validation = validate_los(cnn, baseParams, frameTool, carrierMap, ...
        cfg.ValidationFrames);
    fprintf("Validazione LOS K=15 | Eb/No 10/15/20 dB: %s\n", ...
        mat2str(validation.BER, 4));

    checkpointDir = paths.Checkpoints;
    if ~exist(checkpointDir, 'dir')
        mkdir(checkpointDir);
    end
    timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss_SSS"));
    modelPath = fullfile(checkpointDir, sprintf( ...
        'ckpt_ft_%s.mat', timestamp));

    net = cnn.Net;
    trainingState = struct();
    trainingState.timestamp = timestamp;
    trainingState.networkArchitecture = LOS_CNN.architectureVersion();
    trainingState.frameFormat = ...
        "legacy_80211a_LSTF_LLTF_QPSK_uncoded_dynamic_payload";
    trainingState.NumInputChannels = cnn.NumInputChannels;
    trainingState.NumIterations = cfg.NumIterations;
    trainingState.MiniBatchSize = cfg.MiniBatchSize;
    trainingState.InitialLearnRate = cfg.InitialLearnRate;
    trainingState.LOSKRange = cfg.KFactorRange;
    trainingState.SNRRangeDb = cfg.SNRRangeDb;
    trainingState.RandomCarrierPhaseAugmentation = true;
    trainingState.validation = validation;
    trainingState.parameters = baseParams;
    save(modelPath, 'net', 'trainingState', 'lossHistory', 'validation', '-v7.3');
    fprintf("Checkpoint frame-aware salvato: %s\n", modelPath);
end

function [X, T] = generate_batch(cnn, baseParams, frameTool, carrierMap, ...
        batchCount, snrRangeDb, kFactorRange)
    X = zeros(cnn.NumInputChannels, carrierMap.NumSubcarriers, ...
        batchCount, 'single');
    labels = repmat(uint16(baseParams.PilotClass), ...
        carrierMap.NumSubcarriers, batchCount);

    sampleIdx = 0;
    while sampleIdx < batchCount
        params = random_los_params(baseParams, snrRangeDb, kFactorRange);
        [txWaveform, txFrame] = frameTool.createRandomFrame();
        rxWaveform = multipath_channel(txWaveform, params);
        rxWaveform = rxWaveform .* exp(1i * 2 * pi * rand());
        rxFrame = frameTool.receive(rxWaveform);

        takeCount = min([baseParams.NumTrainingSymbolsPerFrame, ...
            baseParams.NumOFDMSymbols, batchCount - sampleIdx]);
        selected = randperm(baseParams.NumOFDMSymbols, takeCount);
        frameFeatures = build_frame_cnn_batch( ...
            rxFrame, txFrame, carrierMap, cnn.NumInputChannels);

        for symbolIdx = selected
            sampleIdx = sampleIdx + 1;
            X(:, :, sampleIdx) = frameFeatures(:, :, symbolIdx);
            payloadLabels = cnn.symbolsToClasses( ...
                txFrame.PayloadSymbols(:, symbolIdx));
            labels(carrierMap.PayloadGlobalIdx, sampleIdx) = ...
                uint16(payloadLabels);
        end
    end

    T = cnn.classesToOneHot(labels);
end

function params = random_los_params(baseParams, snrRangeDb, kFactorRange)
    params = baseParams;
    params.ChannelType = 'Rician';
    params.SNR_dB = snrRangeDb(1) + rand() * diff(snrRangeDb);
    params.KFactor = exp(log(kFactorRange(1)) + rand() * ...
        (log(kFactorRange(2)) - log(kFactorRange(1))));
    params.MaximumDopplerShift = 40 * rand();

    draw = rand();
    if draw < 0.25
        params.PathDelays = 0;
        params.AveragePathGains = 0;
    elseif draw < 0.75
        params.PathDelays = [0, (20 + 100 * rand()) * 1e-9];
        params.AveragePathGains = [0, -(6 + 14 * rand())];
    else
        delays = sort((20 + 130 * rand(1, 2)) * 1e-9);
        params.PathDelays = [0, delays];
        params.AveragePathGains = [0, -(5 + 8 * rand()), -(13 + 12 * rand())];
    end
    params = params.refreshDerived();
end

function validation = validate_los(cnn, baseParams, frameTool, carrierMap, numFrames)
    ebNoDb = [10 15 20];
    ber = zeros(size(ebNoDb));

    for snrIdx = 1:numel(ebNoDb)
        params = configure_test_channel(baseParams, ebNoDb(snrIdx), 'LOS', 15);
        bitErrors = 0;
        totalBits = 0;

        for frameIdx = 1:numFrames
            [txWaveform, txFrame] = frameTool.createRandomFrame();
            rxWaveform = multipath_channel(txWaveform, params);
            rxFrame = frameTool.receive(rxWaveform);
            X = build_frame_cnn_batch(rxFrame, txFrame, carrierMap, ...
                cnn.NumInputChannels);
            X = cnn.normalizeInputBatch(X, carrierMap.ActiveGlobalIdx);
            scores = gather(extractdata(forward(cnn.Net, ...
                dlarray(single(X), 'CTB'))));
            if size(scores, 2) == baseParams.NumOFDMSymbols
                scores = permute(scores, [1 3 2]);
            end
            predicted = cnn.oneHotToClasses(scores);
            predicted = predicted(carrierMap.PayloadGlobalIdx, :);
            truth = cnn.symbolsToClasses(txFrame.PayloadSymbols);
            trueBits = cnn.classesToBits(truth);
            predictedBits = cnn.classesToBits(predicted);
            bitErrors = bitErrors + sum(trueBits(:) ~= predictedBits(:));
            totalBits = totalBits + numel(trueBits);
        end
        ber(snrIdx) = bitErrors / totalBits;
    end

    validation = struct();
    validation.EbNoDb = ebNoDb;
    validation.BER = ber;
    validation.NumFrames = numFrames;
end

function checkpoint = latest_checkpoint(checkpointDir)
    checkpoint = "";
    files = dir(fullfile(checkpointDir, 'ckpt_*.mat'));
    if isempty(files)
        return;
    end
    [~, idx] = max([files.datenum]);
    checkpoint = string(fullfile(checkpointDir, files(idx).name));
end
