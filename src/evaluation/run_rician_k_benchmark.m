function results = run_rician_k_benchmark(varargin)
%RUN_RICIAN_K_BENCHMARK BER CNN/MMSE su griglia SNR e fattore K Rician.
%   L'inferenza CNN viene raggruppata su piu frame per limitare l'overhead.

    parser = inputParser();
    parser.FunctionName = 'run_rician_k_benchmark';
    addParameter(parser, 'KFactorDb', [0 10 20 30]);
    addParameter(parser, 'SNRdB', 0:2:34);
    addParameter(parser, 'MinFrames', 300);
    addParameter(parser, 'MaxFrames', 600);
    addParameter(parser, 'TargetBitErrors', 400);
    addParameter(parser, 'FrameBatchSize', 4);
    addParameter(parser, 'ModelPath', "");
    addParameter(parser, 'MaximumDopplerShift', 0);
    addParameter(parser, 'RandomSeed', 352);
    addParameter(parser, 'CommonRandomNumbers', true);
    addParameter(parser, 'ExportChart', true);
    addParameter(parser, 'WriteLog', true);
    addParameter(parser, 'Visible', true);
    parse(parser, varargin{:});
    cfg = parser.Results;
    validate_configuration(cfg);

    paths = project_paths(true);
    timestamp = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'));
    rng(cfg.RandomSeed, 'twister');

    baseParams = simulation_parameters().refreshDerived();
    carrierMap = baseParams.getCarrierMap();
    frameTool = IEEE80211aFrame(baseParams);
    [cnn, modelPath, checkpointSelection] = load_cnn_checkpoint( ...
        baseParams, cfg.ModelPath, paths.Checkpoints);

    snrDb = double(cfg.SNRdB(:).');
    kDb = double(cfg.KFactorDb(:).');
    kLinear = 10.^(kDb / 10);
    methodNames = ["CNN" "MMSE"];
    errors = zeros(numel(kDb), numel(snrDb), numel(methodNames));
    totalBits = zeros(numel(kDb), numel(snrDb));
    numFrames = zeros(numel(kDb), numel(snrDb));
    syncFailures = zeros(numel(kDb), numel(snrDb));

    fprintf('Checkpoint: %s\n', modelPath);
    fprintf('K [dB]: %s | SNR [dB]: %s\n', mat2str(kDb), mat2str(snrDb));

    for kIdx = 1:numel(kDb)
        for snrIdx = 1:numel(snrDb)
            params = configure_test_channel(baseParams, snrDb(snrIdx), ...
                'LOS', kLinear(kIdx), ...
                'KFactor', kLinear(kIdx), ...
                'MaximumDopplerShift', cfg.MaximumDopplerShift);
            params.assertOFDMChannelConsistency(sprintf( ...
                'Rician K=%g dB, SNR=%g dB', kDb(kIdx), snrDb(snrIdx)));

            pointErrors = zeros(1, numel(methodNames));
            framesSeen = 0;
            bitsSeen = 0;
            badSync = 0;

            while framesSeen < cfg.MaxFrames
                batchFrames = min(cfg.FrameBatchSize, ...
                    cfg.MaxFrames - framesSeen);
                samplesPerFrame = baseParams.NumOFDMSymbols;
                xBatch = zeros(cnn.NumInputChannels, baseParams.FFTLength, ...
                    samplesPerFrame * batchFrames, 'single');
                truthBatch = zeros(baseParams.NumPayloadCarriers, ...
                    samplesPerFrame * batchFrames, 'uint16');
                actualBatchFrames = 0;

                for batchIdx = 1:batchFrames
                    frameNumber = framesSeen + batchIdx;
                    if cfg.CommonRandomNumbers
                        rng(common_frame_seed(cfg.RandomSeed, frameNumber), ...
                            'twister');
                    end

                    [txWaveform, txFrame] = frameTool.createRandomFrame();
                    rxWaveform = multipath_channel(txWaveform, params);
                    rxFrame = frameTool.receive(rxWaveform);
                    actualBatchFrames = actualBatchFrames + 1;
                    badSync = badSync + (rxFrame.SynchronizationMetric < 0.05);

                    mmse = frameTool.equalize(rxFrame, 'MMSE');
                    pointErrors(2) = pointErrors(2) + ...
                        count_symbol_bit_errors(mmse.PayloadSymbols, ...
                        txFrame.PayloadBits, baseParams.ModulationOrder);

                    sampleRange = (batchIdx - 1) * samplesPerFrame + ...
                        (1:samplesPerFrame);
                    xBatch(:, :, sampleRange) = build_frame_cnn_batch( ...
                        rxFrame, txFrame, carrierMap, cnn.NumInputChannels);
                    truthBatch(:, sampleRange) = uint16( ...
                        cnn.symbolsToClasses(txFrame.PayloadSymbols));
                end

                usedSamples = 1:(actualBatchFrames * samplesPerFrame);
                xBatch = cnn.normalizeInputBatch( ...
                    xBatch(:, :, usedSamples), carrierMap.ActiveGlobalIdx);
                scores = gather(extractdata(forward(cnn.Net, ...
                    dlarray(xBatch, 'CTB'))));
                scores = normalize_score_shape(scores, cnn, numel(usedSamples));
                predicted = cnn.oneHotToClasses(scores);
                predicted = predicted(carrierMap.PayloadGlobalIdx, :);
                trueBits = cnn.classesToBits(truthBatch(:, usedSamples));
                predictedBits = cnn.classesToBits(predicted);
                pointErrors(1) = pointErrors(1) + ...
                    sum(trueBits(:) ~= predictedBits(:));

                framesSeen = framesSeen + actualBatchFrames;
                bitsSeen = bitsSeen + actualBatchFrames * ...
                    numel(txFrame.PayloadBits);
                if framesSeen >= cfg.MinFrames && ...
                        all(pointErrors >= cfg.TargetBitErrors)
                    break;
                end
            end

            errors(kIdx, snrIdx, :) = pointErrors;
            totalBits(kIdx, snrIdx) = bitsSeen;
            numFrames(kIdx, snrIdx) = framesSeen;
            syncFailures(kIdx, snrIdx) = badSync;
            fprintf('K %2g dB | SNR %4g dB | CNN %.3e | MMSE %.3e | %d frame\n', ...
                kDb(kIdx), snrDb(snrIdx), pointErrors(1) / bitsSeen, ...
                pointErrors(2) / bitsSeen, framesSeen);
        end
    end

    results = struct();
    results.Timestamp = timestamp;
    results.SNRdB = snrDb;
    results.KFactorDb = kDb;
    results.KFactorLinear = kLinear;
    results.MethodNames = methodNames;
    results.Errors = errors;
    results.TotalBits = totalBits;
    results.NumFrames = numFrames;
    results.SyncFailures = syncFailures;
    results.BER = errors ./ totalBits;
    results.PlotBER = max(errors, 0.5) ./ totalBits;
    results.ZeroErrorConvention = "0.5 / bit elaborati";
    results.ModelPath = modelPath;
    results.CheckpointSelection = checkpointSelection;
    results.Configuration = cfg;
    results.Parameters = baseParams;
    results.TestProfile = struct( ...
        'ChannelType', 'Rician', ...
        'Scenario', 'LOS multipath', ...
        'SNRDefinition', baseParams.SNRDefinition, ...
        'SNRdB', snrDb, ...
        'KFactorDb', kDb, ...
        'KFactorLinear', kLinear, ...
        'PathDelaysSeconds', [0 30e-9], ...
        'AveragePathGainsDb', [0 -10], ...
        'MaximumDopplerShiftHz', cfg.MaximumDopplerShift, ...
        'RandomSeed', cfg.RandomSeed, ...
        'CommonRandomNumbers', logical(cfg.CommonRandomNumbers), ...
        'MinFrames', cfg.MinFrames, ...
        'MaxFrames', cfg.MaxFrames, ...
        'TargetBitErrors', cfg.TargetBitErrors, ...
        'FrameBatchSize', cfg.FrameBatchSize);
    results.ChartPath = "";
    results.LogPath = "";

    [figureHandle, chartPath, chartConfig] = plot_rician_k_ber( ...
        results, 'Export', cfg.ExportChart, 'Visible', cfg.Visible, ...
        'OutputDir', paths.Charts);
    results.FigureHandle = figureHandle;
    results.ChartPath = chartPath;
    results.ChartConfiguration = chartConfig;

    if cfg.WriteLog
        results.LogPath = write_base64_test_log(results, paths.Logs);
    end
end

function count = count_symbol_bit_errors(symbols, trueBits, modOrder)
    bits = qamdemod(symbols(:), modOrder, 'gray', ...
        'UnitAveragePower', true, 'OutputType', 'bit');
    count = sum(uint8(bits(:)) ~= trueBits(:));
end

function scores = normalize_score_shape(scores, cnn, batchCount)
    if ismatrix(scores)
        scores = reshape(scores, size(scores, 1), size(scores, 2), 1);
    end
    if size(scores, 1) ~= cnn.ModOrder
        error('Output CNN non valido: prima dimensione diversa da M.');
    end
    if size(scores, 2) == cnn.NFFT && size(scores, 3) == batchCount
        return;
    end
    if size(scores, 2) == batchCount && size(scores, 3) == cnn.NFFT
        scores = permute(scores, [1 3 2]);
        return;
    end
    error('Output CNN non valido: [%s].', num2str(size(scores)));
end

function seed = common_frame_seed(baseSeed, frameNumber)
    seed = mod(double(baseSeed) + 104729 * double(frameNumber), 2^31 - 1);
    seed = max(1, floor(seed));
end

function validate_configuration(cfg)
    validateattributes(cfg.KFactorDb, {'numeric'}, ...
        {'real', 'finite', 'vector', 'nonempty'});
    validateattributes(cfg.SNRdB, {'numeric'}, ...
        {'real', 'finite', 'vector', 'nonempty', '>=', 0});
    validateattributes(cfg.MinFrames, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    validateattributes(cfg.MaxFrames, {'numeric'}, ...
        {'scalar', 'integer', '>=', cfg.MinFrames});
    validateattributes(cfg.TargetBitErrors, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    validateattributes(cfg.FrameBatchSize, {'numeric'}, ...
        {'scalar', 'integer', 'positive', '<=', cfg.MaxFrames});
    validateattributes(cfg.MaximumDopplerShift, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'nonnegative'});
    validateattributes(cfg.RandomSeed, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative'});
end
