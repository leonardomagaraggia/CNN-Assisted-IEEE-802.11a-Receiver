function results = run_ber_benchmark(varargin)
% BER su frame identici per i metodi richiesti con arresto Monte Carlo.

    parser = inputParser();
    addParameter(parser, 'EbNoDb', -1:2:35);
    addParameter(parser, 'Scenarios', ["AWGN" "LOS" "NLOS"]);
    addParameter(parser, 'KFactorLOS', 15);
    addParameter(parser, 'MaximumDopplerShift', 0);
    addParameter(parser, 'MinFrames', 40);
    addParameter(parser, 'MaxFrames', 400);
    addParameter(parser, 'TargetBitErrors', 250);
    addParameter(parser, 'IncludeCNN', true);
    addParameter(parser, 'Methods', ["CNN" "ZF" "MMSE"]);
    addParameter(parser, 'ModelPath', "");
    addParameter(parser, 'UseCNNZFFallback', false);
    addParameter(parser, 'CNNZFGuardSNRdB', 12);
    addParameter(parser, 'CommonRandomNumbers', true);
    addParameter(parser, 'MakePlot', true);
    parse(parser, varargin{:});
    cfg = parser.Results;

    methodKeys = upper(string(cfg.Methods(:).'));
    methodKeys = unique(methodKeys, 'stable');
    allowedMethods = ["CNN" "ZF" "MMSE"];
    if any(~ismember(methodKeys, allowedMethods))
        error('Methods supportati: CNN, ZF e MMSE.');
    end
    if ~cfg.IncludeCNN
        methodKeys(methodKeys == "CNN") = [];
    end
    if isempty(methodKeys)
        error('Specificare almeno un metodo da valutare.');
    end
    if cfg.UseCNNZFFallback && ~any(methodKeys == "CNN")
        error('UseCNNZFFallback richiede il metodo CNN.');
    end

    rng(352);
    baseParams = simulation_parameters().refreshDerived();
    carrierMap = baseParams.getCarrierMap();
    frameTool = IEEE80211aFrame(baseParams);

    cnn = [];
    modelPath = "";
    checkpointSelection = struct();
    if any(methodKeys == "CNN")
        [cnn, modelPath, checkpointSelection] = ...
            load_cnn_checkpoint(baseParams, cfg.ModelPath);
    end

    scenarios = string(cfg.Scenarios);
    ebNoDb = double(cfg.EbNoDb);
    numScenarios = numel(scenarios);
    numPoints = numel(ebNoDb);

    methodNames = methodKeys;
    if cfg.UseCNNZFFallback
        methodNames(methodKeys == "CNN") = "CNN/ZF adattiva";
    end
    errors = zeros(numScenarios, numPoints, numel(methodNames));
    totalBits = zeros(numScenarios, numPoints);
    numFrames = zeros(numScenarios, numPoints);
    syncFailures = zeros(numScenarios, numPoints);
    cnnFallbackFrames = zeros(numScenarios, numPoints);

    fprintf("Frame 802.11a: L-STF + L-LTF + %d simboli QPSK uncoded dinamici\n", ...
        baseParams.NumOFDMSymbols);
    if strlength(modelPath) > 0
        fprintf("Modello CNN: %s\n", modelPath);
    end

    for scenarioIdx = 1:numScenarios
        scenario = scenarios(scenarioIdx);
        fprintf("\nScenario %s\n", scenario);

        for snrIdx = 1:numPoints
            params = configure_test_channel(baseParams, ebNoDb(snrIdx), ...
                scenario, cfg.KFactorLOS, ...
                'MaximumDopplerShift', cfg.MaximumDopplerShift);
            frameErrors = zeros(1, numel(methodNames));
            bitsSeen = 0;
            framesSeen = 0;
            badSync = 0;
            fallbackFrames = 0;

            while framesSeen < cfg.MaxFrames
                if cfg.CommonRandomNumbers
                    rng(100000 * scenarioIdx + framesSeen + 1, 'twister');
                end
                [txWaveform, txFrame] = frameTool.createRandomFrame();
                rxWaveform = multipath_channel(txWaveform, params);
                rxFrame = frameTool.receive(rxWaveform);
                framesSeen = framesSeen + 1;

                if rxFrame.SynchronizationMetric < 0.05
                    badSync = badSync + 1;
                end

                zf = [];
                zfIdx = find(methodKeys == "ZF", 1);
                if ~isempty(zfIdx) || cfg.UseCNNZFFallback
                    zf = frameTool.equalize(rxFrame, 'ZF');
                end
                if ~isempty(zfIdx)
                    frameErrors(zfIdx) = frameErrors(zfIdx) + ...
                        count_symbol_bit_errors(zf.PayloadSymbols, ...
                        txFrame.PayloadBits, baseParams.ModulationOrder);
                end

                mmseIdx = find(methodKeys == "MMSE", 1);
                if ~isempty(mmseIdx)
                    mmse = frameTool.equalize(rxFrame, 'MMSE');
                    frameErrors(mmseIdx) = frameErrors(mmseIdx) + ...
                        count_symbol_bit_errors(mmse.PayloadSymbols, ...
                        txFrame.PayloadBits, baseParams.ModulationOrder);
                end

                cnnIdx = find(methodKeys == "CNN", 1);
                if ~isempty(cnnIdx)
                    X = build_frame_cnn_batch(rxFrame, txFrame, carrierMap, ...
                        cnn.NumInputChannels);
                    X = cnn.normalizeInputBatch(X, carrierMap.ActiveGlobalIdx);
                    scores = gather(extractdata(forward(cnn.Net, ...
                        dlarray(single(X), 'CTB'))));
                    scores = normalize_score_shape(scores, cnn, ...
                        baseParams.NumOFDMSymbols);
                    predicted = cnn.oneHotToClasses(scores);
                    predicted = predicted(carrierMap.PayloadGlobalIdx, :);
                    if cfg.UseCNNZFFallback
                        estimatedFrameSNRdB = -10 * log10( ...
                            rxFrame.NoiseVariance);
                    else
                        estimatedFrameSNRdB = -Inf;
                    end
                    if cfg.UseCNNZFFallback && ...
                            estimatedFrameSNRdB >= cfg.CNNZFGuardSNRdB
                        predicted = qamdemod(zf.PayloadSymbols, ...
                            baseParams.ModulationOrder, 'gray', ...
                            'UnitAveragePower', true) + 1;
                        fallbackFrames = fallbackFrames + 1;
                    end
                    truth = cnn.symbolsToClasses(txFrame.PayloadSymbols);
                    trueBits = cnn.classesToBits(truth);
                    predictedBits = cnn.classesToBits(predicted);
                    frameErrors(cnnIdx) = frameErrors(cnnIdx) + ...
                        sum(trueBits(:) ~= predictedBits(:));
                end

                bitsSeen = bitsSeen + numel(txFrame.PayloadBits);
                if framesSeen >= cfg.MinFrames
                    if all(frameErrors >= cfg.TargetBitErrors)
                        break;
                    end
                end
            end

            errors(scenarioIdx, snrIdx, :) = frameErrors;
            totalBits(scenarioIdx, snrIdx) = bitsSeen;
            numFrames(scenarioIdx, snrIdx) = framesSeen;
            syncFailures(scenarioIdx, snrIdx) = badSync;
            cnnFallbackFrames(scenarioIdx, snrIdx) = fallbackFrames;

            berPoint = frameErrors / max(bitsSeen, 1);
            fprintf("  Eb/No %5.1f dB", ebNoDb(snrIdx));
            for methodIdx = 1:numel(methodNames)
                fprintf(" | %s %.3e", methodNames(methodIdx), berPoint(methodIdx));
            end
            fprintf(" | %d frame\n", framesSeen);
        end
    end

    ber = errors ./ max(totalBits, 1);
    plotBer = max(errors, 0.5) ./ max(totalBits, 1);

    results = struct();
    results.EbNoDb = ebNoDb;
    results.Scenarios = scenarios;
    results.MethodNames = methodNames;
    results.Errors = errors;
    results.TotalBits = totalBits;
    results.NumFrames = numFrames;
    results.SyncFailures = syncFailures;
    results.CNNFallbackFrames = cnnFallbackFrames;
    results.UseCNNZFFallback = cfg.UseCNNZFFallback;
    results.CNNZFGuardSNRdB = cfg.CNNZFGuardSNRdB;
    results.CommonRandomNumbers = cfg.CommonRandomNumbers;
    results.BER = ber;
    results.PlotBER = plotBer;
    results.KFactorLOS = cfg.KFactorLOS;
    results.MaximumDopplerShift = cfg.MaximumDopplerShift;
    results.ModelPath = modelPath;
    results.CheckpointSelection = checkpointSelection;
    results.ZeroErrorConvention = "0.5/numero_bit (limite grafico)";
    results.Parameters = baseParams;
    results.Persistence = "disabilitata: risultati solo in memoria";

    if cfg.MakePlot
        results.FigureHandle = plot_ber_results(results);
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
