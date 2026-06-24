function report = run_cnn_deep_test(varargin)
% Stress test multi-condizione della CNN su frame condivisi con ZF e MMSE.

    parser = inputParser();
    addParameter(parser, 'ModelPath', "");
    addParameter(parser, 'EbNoDb', [0 5 10 15 20 25 30 35]);
    addParameter(parser, 'Scenarios', ["AWGN" "LOS" "NLOS"]);
    addParameter(parser, 'LOSKFactors', [3 8 15 30]);
    addParameter(parser, 'NLOSKFactors', [1e-4 0.1 1]);
    addParameter(parser, 'DopplerHz', [0 10 40]);
    addParameter(parser, 'FramesPerCondition', 20);
    addParameter(parser, 'RandomSeed', 271828);
    addParameter(parser, 'WarmupIterations', 3);
    addParameter(parser, 'SynchronizationThreshold', 0.05);
    addParameter(parser, 'ConfidenceLevel', 0.95);
    addParameter(parser, 'MakePlots', true);
    addParameter(parser, 'Verbose', true);
    parse(parser, varargin{:});
    cfg = parser.Results;
    cfg.Scenarios = upper(string(cfg.Scenarios(:).'));
    validate_configuration(cfg);

    rng(cfg.RandomSeed, 'twister');
    baseParams = simulation_parameters().refreshDerived();
    carrierMap = baseParams.getCarrierMap();
    frameTool = IEEE80211aFrame(baseParams);
    [cnn, modelPath, checkpointSelection] = ...
        load_cnn_checkpoint(baseParams, cfg.ModelPath);

    conditions = build_conditions(cfg);
    numConditions = height(conditions);
    methodNames = ["CNN" "ZF" "MMSE"];
    numMethods = numel(methodNames);
    totalFrameSamples = numConditions * cfg.FramesPerCondition;

    bitErrors = zeros(numConditions, numMethods);
    symbolErrors = zeros(numConditions, numMethods);
    frameErrors = zeros(numConditions, numMethods);
    totalBits = zeros(numConditions, 1);
    totalSymbols = zeros(numConditions, 1);
    syncFailures = zeros(numConditions, 1);
    pilotTrackingFrames = zeros(numConditions, 1);

    cnnFeatureSeconds = zeros(totalFrameSamples, 1);
    cnnInferenceSeconds = zeros(totalFrameSamples, 1);
    cnnPostprocessSeconds = zeros(totalFrameSamples, 1);
    cnnEndToEndSeconds = zeros(totalFrameSamples, 1);
    zfDetectorSeconds = zeros(totalFrameSamples, 1);
    mmseDetectorSeconds = zeros(totalFrameSamples, 1);
    timingConditionIndex = zeros(totalFrameSamples, 1);

    warm_up_detector(cnn, baseParams, carrierMap, frameTool, ...
        conditions(1, :), cfg);

    bitsPerFrame = baseParams.NumPayloadCarriers * ...
        baseParams.NumOFDMSymbols * log2(baseParams.ModulationOrder);
    fprintf("=== STRESS TEST CNN MULTI-CONDIZIONE ===\n");
    fprintf("Modello: %s\n", modelPath);
    fprintf("Condizioni: %d | frame/condizione: %d | frame totali: %d\n", ...
        numConditions, cfg.FramesPerCondition, totalFrameSamples);
    fprintf("Bit valutati previsti: %.3g\n", ...
        totalFrameSamples * bitsPerFrame);

    timingIdx = 0;
    for conditionIdx = 1:numConditions
        condition = conditions(conditionIdx, :);
        params = condition_parameters(baseParams, condition);
        params.assertOFDMChannelConsistency(condition_label(condition));

        for frameIdx = 1:cfg.FramesPerCondition
            rng(frame_seed(cfg.RandomSeed, conditionIdx, frameIdx), 'twister');
            [txWaveform, txFrame] = frameTool.createRandomFrame();
            rxWaveform = multipath_channel(txWaveform, params);
            rxFrame = frameTool.receive(rxWaveform);

            truthClasses = cnn.symbolsToClasses(txFrame.PayloadSymbols);
            truthBits = txFrame.PayloadBits;

            [cnnClasses, cnnBits, cnnTiming] = detect_with_cnn( ...
                cnn, rxFrame, txFrame, carrierMap, ...
                baseParams.NumOFDMSymbols);
            [zfClasses, zfBits, zfSeconds] = detect_with_linear_equalizer( ...
                frameTool, rxFrame, 'ZF', baseParams.ModulationOrder);
            [mmseClasses, mmseBits, mmseSeconds] = ...
                detect_with_linear_equalizer(frameTool, rxFrame, ...
                'MMSE', baseParams.ModulationOrder);

            frameBitErrors = [ ...
                sum(cnnBits(:) ~= truthBits(:)), ...
                sum(zfBits(:) ~= truthBits(:)), ...
                sum(mmseBits(:) ~= truthBits(:))];
            frameSymbolErrors = [ ...
                sum(cnnClasses(:) ~= truthClasses(:)), ...
                sum(zfClasses(:) ~= truthClasses(:)), ...
                sum(mmseClasses(:) ~= truthClasses(:))];

            bitErrors(conditionIdx, :) = ...
                bitErrors(conditionIdx, :) + frameBitErrors;
            symbolErrors(conditionIdx, :) = ...
                symbolErrors(conditionIdx, :) + frameSymbolErrors;
            frameErrors(conditionIdx, :) = ...
                frameErrors(conditionIdx, :) + (frameBitErrors > 0);
            totalBits(conditionIdx) = totalBits(conditionIdx) + ...
                numel(truthBits);
            totalSymbols(conditionIdx) = totalSymbols(conditionIdx) + ...
                numel(truthClasses);
            syncFailures(conditionIdx) = syncFailures(conditionIdx) + ...
                (rxFrame.SynchronizationMetric < ...
                cfg.SynchronizationThreshold);
            pilotTrackingFrames(conditionIdx) = ...
                pilotTrackingFrames(conditionIdx) + ...
                double(rxFrame.PilotPhaseTrackingEnabled);

            timingIdx = timingIdx + 1;
            cnnFeatureSeconds(timingIdx) = cnnTiming.FeatureSeconds;
            cnnInferenceSeconds(timingIdx) = cnnTiming.InferenceSeconds;
            cnnPostprocessSeconds(timingIdx) = cnnTiming.PostprocessSeconds;
            cnnEndToEndSeconds(timingIdx) = cnnTiming.EndToEndSeconds;
            zfDetectorSeconds(timingIdx) = zfSeconds;
            mmseDetectorSeconds(timingIdx) = mmseSeconds;
            timingConditionIndex(timingIdx) = conditionIdx;
        end

        if cfg.Verbose
            conditionBER = bitErrors(conditionIdx, :) ./ ...
                totalBits(conditionIdx);
            fprintf("%3d/%3d | %-38s | BER CNN %.3e | ZF %.3e | MMSE %.3e\n", ...
                conditionIdx, numConditions, condition_label(condition), ...
                conditionBER(1), conditionBER(2), conditionBER(3));
        end
    end

    zValue = normal_quantile(0.5 + cfg.ConfidenceLevel / 2);
    ber = bitErrors ./ totalBits;
    ser = symbolErrors ./ totalSymbols;
    fer = frameErrors ./ cfg.FramesPerCondition;
    [berLower, berUpper] = wilson_interval(bitErrors, totalBits, zValue);
    [ferLower, ferUpper] = wilson_interval( ...
        frameErrors, cfg.FramesPerCondition, zValue);

    conditionTable = build_condition_results( ...
        conditions, methodNames, bitErrors, symbolErrors, frameErrors, ...
        totalBits, ber, ser, fer, berLower, berUpper, ...
        ferLower, ferUpper, syncFailures, pilotTrackingFrames, ...
        cfg.FramesPerCondition, timingConditionIndex, ...
        cnnInferenceSeconds, cnnEndToEndSeconds);
    summaryTable = aggregate_summary( ...
        methodNames, bitErrors, symbolErrors, frameErrors, ...
        totalBits, totalSymbols, cfg.FramesPerCondition, zValue);
    scenarioSummary = grouped_summary( ...
        conditions, methodNames, bitErrors, symbolErrors, frameErrors, ...
        totalBits, totalSymbols, cfg.FramesPerCondition, zValue, ...
        'Scenario');
    snrSummary = grouped_summary( ...
        conditions, methodNames, bitErrors, symbolErrors, frameErrors, ...
        totalBits, totalSymbols, cfg.FramesPerCondition, zValue, ...
        'ScenarioEbNo');

    timingSummary = build_timing_summary( ...
        cnnFeatureSeconds, cnnInferenceSeconds, cnnPostprocessSeconds, ...
        cnnEndToEndSeconds, zfDetectorSeconds, mmseDetectorSeconds, ...
        baseParams.NumOFDMSymbols);
    comparison = build_comparison( ...
        conditionTable, summaryTable, methodNames);

    report = struct();
    report.ModelPath = modelPath;
    report.CheckpointSelection = checkpointSelection;
    report.Configuration = cfg;
    report.Parameters = baseParams;
    report.MethodNames = methodNames;
    report.Conditions = conditionTable;
    report.Summary = summaryTable;
    report.ScenarioSummary = scenarioSummary;
    report.SNRSummary = snrSummary;
    report.Comparison = comparison;
    report.Timing = struct();
    report.Timing.Summary = timingSummary;
    report.Timing.CNNFeatureSeconds = cnnFeatureSeconds;
    report.Timing.CNNInferenceSeconds = cnnInferenceSeconds;
    report.Timing.CNNPostprocessSeconds = cnnPostprocessSeconds;
    report.Timing.CNNEndToEndSeconds = cnnEndToEndSeconds;
    report.Timing.ZFDetectorSeconds = zfDetectorSeconds;
    report.Timing.MMSEDetectorSeconds = mmseDetectorSeconds;
    report.Timing.ConditionIndex = timingConditionIndex;
    report.Persistence = "disabilitata: risultati solo in memoria";

    print_summary(report);
    if cfg.MakePlots
        report.FigureHandles = plot_report(report);
    end
end

function conditions = build_conditions(cfg)
    scenarioColumn = strings(0, 1);
    kFactorColumn = zeros(0, 1);
    dopplerColumn = zeros(0, 1);
    ebNoColumn = zeros(0, 1);

    for scenario = cfg.Scenarios
        switch scenario
            case "AWGN"
                kFactors = NaN;
                dopplerValues = 0;
            case "LOS"
                kFactors = double(cfg.LOSKFactors(:).');
                dopplerValues = double(cfg.DopplerHz(:).');
            case "NLOS"
                kFactors = double(cfg.NLOSKFactors(:).');
                dopplerValues = double(cfg.DopplerHz(:).');
        end

        for kFactor = kFactors
            for dopplerHz = dopplerValues
                for ebNoDb = double(cfg.EbNoDb(:).')
                    scenarioColumn(end+1, 1) = scenario; %#ok<AGROW>
                    kFactorColumn(end+1, 1) = kFactor; %#ok<AGROW>
                    dopplerColumn(end+1, 1) = dopplerHz; %#ok<AGROW>
                    ebNoColumn(end+1, 1) = ebNoDb; %#ok<AGROW>
                end
            end
        end
    end

    conditions = table(scenarioColumn, kFactorColumn, dopplerColumn, ...
        ebNoColumn, 'VariableNames', ...
        {'Scenario', 'KFactor', 'DopplerHz', 'EbNoDb'});
end

function params = condition_parameters(baseParams, condition)
    if condition.Scenario == "AWGN"
        params = configure_test_channel( ...
            baseParams, condition.EbNoDb, condition.Scenario, 15);
    else
        params = configure_test_channel( ...
            baseParams, condition.EbNoDb, condition.Scenario, 15, ...
            'KFactor', condition.KFactor, ...
            'MaximumDopplerShift', condition.DopplerHz);
    end
end

function warm_up_detector(cnn, baseParams, carrierMap, frameTool, ...
        condition, cfg)
    if cfg.WarmupIterations == 0
        return;
    end

    params = condition_parameters(baseParams, condition);
    rng(frame_seed(cfg.RandomSeed, 0, 1), 'twister');
    [txWaveform, txFrame] = frameTool.createRandomFrame();
    rxFrame = frameTool.receive(multipath_channel(txWaveform, params));
    for warmupIdx = 1:cfg.WarmupIterations
        detect_with_cnn(cnn, rxFrame, txFrame, carrierMap, ...
            baseParams.NumOFDMSymbols);
    end
end

function [classes, bits, timing] = detect_with_cnn( ...
        cnn, rxFrame, txFrame, carrierMap, batchCount)
    featureClock = tic;
    X = build_frame_cnn_batch( ...
        rxFrame, txFrame, carrierMap, cnn.NumInputChannels);
    X = cnn.normalizeInputBatch(X, carrierMap.ActiveGlobalIdx);
    featureSeconds = toc(featureClock);

    inferenceClock = tic;
    scores = gather(extractdata(forward( ...
        cnn.Net, dlarray(single(X), 'CTB'))));
    inferenceSeconds = toc(inferenceClock);

    postprocessClock = tic;
    scores = normalize_score_shape(scores, cnn, batchCount);
    classes = cnn.oneHotToClasses(scores);
    classes = classes(carrierMap.PayloadGlobalIdx, :);
    bits = cnn.classesToBits(classes);
    postprocessSeconds = toc(postprocessClock);

    timing = struct();
    timing.FeatureSeconds = featureSeconds;
    timing.InferenceSeconds = inferenceSeconds;
    timing.PostprocessSeconds = postprocessSeconds;
    timing.EndToEndSeconds = ...
        featureSeconds + inferenceSeconds + postprocessSeconds;
end

function [classes, bits, elapsedSeconds] = detect_with_linear_equalizer( ...
        frameTool, rxFrame, method, modOrder)
    detectorClock = tic;
    equalized = frameTool.equalize(rxFrame, method);
    qamIndices = qamdemod(equalized.PayloadSymbols, modOrder, 'gray', ...
        'UnitAveragePower', true, 'OutputType', 'integer');
    classes = double(qamIndices) + 1;
    bitsColumn = qamdemod(equalized.PayloadSymbols(:), modOrder, 'gray', ...
        'UnitAveragePower', true, 'OutputType', 'bit');
    bits = reshape(uint8(bitsColumn), ...
        [log2(modOrder), size(equalized.PayloadSymbols)]);
    elapsedSeconds = toc(detectorClock);
end

function conditionTable = build_condition_results( ...
        conditions, methodNames, bitErrors, symbolErrors, frameErrors, ...
        totalBits, ber, ser, fer, berLower, berUpper, ...
        ferLower, ferUpper, syncFailures, pilotTrackingFrames, ...
        framesPerCondition, timingConditionIndex, ...
        cnnInferenceSeconds, cnnEndToEndSeconds)
    conditionTable = conditions;
    conditionTable.Frames = repmat(framesPerCondition, height(conditions), 1);
    conditionTable.TotalBits = totalBits;
    conditionTable.SyncFailures = syncFailures;
    conditionTable.PilotTrackingRate = ...
        pilotTrackingFrames / framesPerCondition;

    for methodIdx = 1:numel(methodNames)
        prefix = char(methodNames(methodIdx));
        conditionTable.([prefix 'BitErrors']) = bitErrors(:, methodIdx);
        conditionTable.([prefix 'BER']) = ber(:, methodIdx);
        conditionTable.([prefix 'BERLower']) = berLower(:, methodIdx);
        conditionTable.([prefix 'BERUpper']) = berUpper(:, methodIdx);
        conditionTable.([prefix 'SymbolErrors']) = ...
            symbolErrors(:, methodIdx);
        conditionTable.([prefix 'SER']) = ser(:, methodIdx);
        conditionTable.([prefix 'FrameErrors']) = frameErrors(:, methodIdx);
        conditionTable.([prefix 'FER']) = fer(:, methodIdx);
        conditionTable.([prefix 'FERLower']) = ferLower(:, methodIdx);
        conditionTable.([prefix 'FERUpper']) = ferUpper(:, methodIdx);
    end

    winners = strings(height(conditions), 1);
    for conditionIdx = 1:height(conditions)
        bestBER = min(ber(conditionIdx, :));
        winners(conditionIdx) = strjoin( ...
            methodNames(ber(conditionIdx, :) == bestBER), "=");
    end
    conditionTable.BestBERMethod = winners;

    inferenceMeanMs = zeros(height(conditions), 1);
    inferenceP95Ms = zeros(height(conditions), 1);
    endToEndMeanMs = zeros(height(conditions), 1);
    endToEndP95Ms = zeros(height(conditions), 1);
    for conditionIdx = 1:height(conditions)
        mask = timingConditionIndex == conditionIdx;
        inferenceValues = 1000 * cnnInferenceSeconds(mask);
        endToEndValues = 1000 * cnnEndToEndSeconds(mask);
        inferenceMeanMs(conditionIdx) = mean(inferenceValues);
        inferenceP95Ms(conditionIdx) = percentile(inferenceValues, 95);
        endToEndMeanMs(conditionIdx) = mean(endToEndValues);
        endToEndP95Ms(conditionIdx) = percentile(endToEndValues, 95);
    end
    conditionTable.CNNInferenceMeanMs = inferenceMeanMs;
    conditionTable.CNNInferenceP95Ms = inferenceP95Ms;
    conditionTable.CNNEndToEndMeanMs = endToEndMeanMs;
    conditionTable.CNNEndToEndP95Ms = endToEndP95Ms;
end

function summaryTable = aggregate_summary( ...
        methodNames, bitErrors, symbolErrors, frameErrors, totalBits, ...
        totalSymbols, framesPerCondition, zValue)
    totalFrameCount = size(bitErrors, 1) * framesPerCondition;
    aggregateBitErrors = sum(bitErrors, 1).';
    aggregateSymbolErrors = sum(symbolErrors, 1).';
    aggregateFrameErrors = sum(frameErrors, 1).';
    aggregateBits = sum(totalBits);
    aggregateSymbols = sum(totalSymbols);
    [berLower, berUpper] = wilson_interval( ...
        aggregateBitErrors, aggregateBits, zValue);
    [ferLower, ferUpper] = wilson_interval( ...
        aggregateFrameErrors, totalFrameCount, zValue);

    summaryTable = table(methodNames(:), aggregateBitErrors, ...
        repmat(aggregateBits, numel(methodNames), 1), ...
        aggregateBitErrors / aggregateBits, berLower, berUpper, ...
        aggregateSymbolErrors, ...
        aggregateSymbolErrors / aggregateSymbols, ...
        aggregateFrameErrors, ...
        aggregateFrameErrors / totalFrameCount, ferLower, ferUpper, ...
        'VariableNames', {'Method', 'BitErrors', 'TotalBits', 'BER', ...
        'BERLower', 'BERUpper', 'SymbolErrors', 'SER', 'FrameErrors', ...
        'FER', 'FERLower', 'FERUpper'});
end

function result = grouped_summary( ...
        conditions, methodNames, bitErrors, symbolErrors, frameErrors, ...
        totalBits, totalSymbols, framesPerCondition, zValue, groupingMode)
    scenarioValues = unique(conditions.Scenario, 'stable');
    result = table();

    for scenario = scenarioValues.'
        if strcmp(groupingMode, 'ScenarioEbNo')
            ebNoValues = unique(conditions.EbNoDb( ...
                conditions.Scenario == scenario), 'stable').';
        else
            ebNoValues = NaN;
        end

        for ebNoDb = ebNoValues
            mask = conditions.Scenario == scenario;
            if strcmp(groupingMode, 'ScenarioEbNo')
                mask = mask & conditions.EbNoDb == ebNoDb;
            end
            numFrames = sum(mask) * framesPerCondition;
            bits = sum(totalBits(mask));
            symbols = sum(totalSymbols(mask));

            for methodIdx = 1:numel(methodNames)
                errors = sum(bitErrors(mask, methodIdx));
                symbolErrorCount = sum(symbolErrors(mask, methodIdx));
                frameErrorCount = sum(frameErrors(mask, methodIdx));
                [berLower, berUpper] = wilson_interval( ...
                    errors, bits, zValue);
                [ferLower, ferUpper] = wilson_interval( ...
                    frameErrorCount, numFrames, zValue);
                plotBER = max(errors, 0.5) / bits;

                row = table(scenario, ebNoDb, methodNames(methodIdx), ...
                    errors, bits, errors / bits, plotBER, ...
                    berLower, berUpper, symbolErrorCount / symbols, ...
                    frameErrorCount / numFrames, ferLower, ferUpper, ...
                    'VariableNames', {'Scenario', 'EbNoDb', 'Method', ...
                    'BitErrors', 'TotalBits', 'BER', 'PlotBER', ...
                    'BERLower', 'BERUpper', 'SER', 'FER', ...
                    'FERLower', 'FERUpper'});
                if isempty(result)
                    result = row;
                else
                    result = [result; row]; %#ok<AGROW>
                end
            end
        end
    end
end

function timingTable = build_timing_summary( ...
        featureSeconds, inferenceSeconds, postprocessSeconds, ...
        endToEndSeconds, zfSeconds, mmseSeconds, numSymbolsPerFrame)
    stageNames = [ ...
        "CNN feature extraction"
        "CNN inference"
        "CNN post-processing"
        "CNN end-to-end detector"
        "ZF detector"
        "MMSE detector"];
    values = {featureSeconds, inferenceSeconds, postprocessSeconds, ...
        endToEndSeconds, zfSeconds, mmseSeconds};
    numStages = numel(stageNames);

    meanMs = zeros(numStages, 1);
    medianMs = zeros(numStages, 1);
    p95Ms = zeros(numStages, 1);
    p99Ms = zeros(numStages, 1);
    minMs = zeros(numStages, 1);
    maxMs = zeros(numStages, 1);
    microsecondsPerOFDMSymbol = zeros(numStages, 1);
    framesPerSecond = zeros(numStages, 1);

    for stageIdx = 1:numStages
        milliseconds = 1000 * values{stageIdx}(:);
        meanMs(stageIdx) = mean(milliseconds);
        medianMs(stageIdx) = median(milliseconds);
        p95Ms(stageIdx) = percentile(milliseconds, 95);
        p99Ms(stageIdx) = percentile(milliseconds, 99);
        minMs(stageIdx) = min(milliseconds);
        maxMs(stageIdx) = max(milliseconds);
        microsecondsPerOFDMSymbol(stageIdx) = ...
            1000 * meanMs(stageIdx) / numSymbolsPerFrame;
        framesPerSecond(stageIdx) = 1000 / meanMs(stageIdx);
    end

    timingTable = table(stageNames, meanMs, medianMs, p95Ms, p99Ms, ...
        minMs, maxMs, microsecondsPerOFDMSymbol, framesPerSecond, ...
        'VariableNames', {'Stage', 'MeanMsPerFrame', ...
        'MedianMsPerFrame', 'P95MsPerFrame', 'P99MsPerFrame', ...
        'MinMsPerFrame', 'MaxMsPerFrame', ...
        'MeanMicrosecondsPerOFDMSymbol', 'FramesPerSecond'});
end

function comparison = build_comparison( ...
        conditionTable, summaryTable, methodNames)
    cnnIdx = find(summaryTable.Method == "CNN", 1);
    zfIdx = find(summaryTable.Method == "ZF", 1);
    mmseIdx = find(summaryTable.Method == "MMSE", 1);
    totalBits = summaryTable.TotalBits(cnnIdx);

    cnnComparableBER = max(summaryTable.BitErrors(cnnIdx), 0.5) / totalBits;
    zfComparableBER = max(summaryTable.BitErrors(zfIdx), 0.5) / totalBits;
    mmseComparableBER = ...
        max(summaryTable.BitErrors(mmseIdx), 0.5) / totalBits;

    comparison = struct();
    comparison.BERGainVsZFDb = ...
        10 * log10(zfComparableBER / cnnComparableBER);
    comparison.BERGainVsMMSEDb = ...
        10 * log10(mmseComparableBER / cnnComparableBER);
    comparison.ConditionsCNNBetterThanZF = sum( ...
        conditionTable.CNNBER < conditionTable.ZFBER);
    comparison.ConditionsCNNBetterThanMMSE = sum( ...
        conditionTable.CNNBER < conditionTable.MMSEBER);
    comparison.ConditionsCNNBestOrTied = sum(contains( ...
        conditionTable.BestBERMethod, methodNames(1)));
    comparison.TotalConditions = height(conditionTable);
    comparison.ZeroErrorComparisonConvention = ...
        "0.5 errori come limite comparativo quando gli errori osservati sono zero";
end

function print_summary(report)
    fprintf("\n=== RISULTATI AGGREGATI ===\n");
    disp(report.Summary);

    fprintf("CNN vs ZF: guadagno BER %.2f dB | migliore in %d/%d condizioni\n", ...
        report.Comparison.BERGainVsZFDb, ...
        report.Comparison.ConditionsCNNBetterThanZF, ...
        report.Comparison.TotalConditions);
    fprintf("CNN vs MMSE: guadagno BER %.2f dB | migliore in %d/%d condizioni\n", ...
        report.Comparison.BERGainVsMMSEDb, ...
        report.Comparison.ConditionsCNNBetterThanMMSE, ...
        report.Comparison.TotalConditions);
    fprintf("CNN migliore o a pari merito in %d/%d condizioni\n", ...
        report.Comparison.ConditionsCNNBestOrTied, ...
        report.Comparison.TotalConditions);

    fprintf("\n=== TEMPI DI RISPOSTA ===\n");
    disp(report.Timing.Summary);

    hardest = sortrows(report.Conditions, 'CNNBER', 'descend');
    hardest = hardest(1:min(10, height(hardest)), ...
        {'Scenario', 'KFactor', 'DopplerHz', 'EbNoDb', ...
        'CNNBER', 'ZFBER', 'MMSEBER', 'CNNFER', ...
        'CNNInferenceP95Ms', 'BestBERMethod'});
    fprintf("\n=== 10 CONDIZIONI PIU DIFFICILI PER LA CNN ===\n");
    disp(hardest);
end

function handles = plot_report(report)
    scenarios = unique(report.SNRSummary.Scenario, 'stable');
    colors = lines(numel(report.MethodNames));
    handles = gobjects(2, 1);

    handles(1) = figure('Name', 'CNN deep test - BER', ...
        'Position', [80 80 420 * numel(scenarios) 430]);
    tiledlayout(1, numel(scenarios), 'TileSpacing', 'compact');
    for scenarioIdx = 1:numel(scenarios)
        scenario = scenarios(scenarioIdx);
        ax = nexttile;
        hold(ax, 'on');
        for methodIdx = 1:numel(report.MethodNames)
            mask = report.SNRSummary.Scenario == scenario & ...
                report.SNRSummary.Method == report.MethodNames(methodIdx);
            rows = report.SNRSummary(mask, :);
            semilogy(ax, rows.EbNoDb, rows.PlotBER, 'o-', ...
                'LineWidth', 1.5, 'MarkerSize', 5, ...
                'Color', colors(methodIdx, :), ...
                'DisplayName', report.MethodNames(methodIdx));
        end
        title(ax, scenario);
        xlabel(ax, 'E_b/N_0 (dB)');
        ylabel(ax, 'BER aggregato');
        grid(ax, 'on');
        ylim(ax, [1e-7 1]);
        legend(ax, 'Location', 'southwest');
        hold(ax, 'off');
    end
    sgtitle('CNN vs ZF vs MMSE sugli stessi frame');

    handles(2) = figure('Name', 'CNN deep test - latency', ...
        'Position', [120 120 980 420]);
    tiledlayout(1, 2, 'TileSpacing', 'compact');
    ax = nexttile;
    plot_latency_cdf(ax, 1000 * report.Timing.CNNInferenceSeconds, ...
        'Inferenza CNN');
    ax = nexttile;
    plot_latency_cdf(ax, 1000 * report.Timing.CNNEndToEndSeconds, ...
        'Detector CNN end-to-end');
end

function plot_latency_cdf(ax, valuesMs, plotTitle)
    sortedValues = sort(valuesMs(:));
    probabilities = (1:numel(sortedValues)).' / numel(sortedValues);
    plot(ax, sortedValues, probabilities, 'LineWidth', 1.5);
    xlabel(ax, 'Tempo per frame (ms)');
    ylabel(ax, 'Probabilita cumulativa');
    title(ax, plotTitle);
    grid(ax, 'on');
    ylim(ax, [0 1]);
end

function [lower, upper] = wilson_interval(errors, total, zValue)
    totalArray = total;
    if isscalar(totalArray)
        totalArray = repmat(totalArray, size(errors));
    else
        totalArray = totalArray + zeros(size(errors));
    end
    proportion = errors ./ totalArray;
    denominator = 1 + zValue^2 ./ totalArray;
    center = (proportion + zValue^2 ./ (2 * totalArray)) ./ denominator;
    margin = zValue .* sqrt( ...
        proportion .* (1 - proportion) ./ totalArray + ...
        zValue^2 ./ (4 * totalArray.^2)) ./ denominator;
    lower = max(0, center - margin);
    upper = min(1, center + margin);
end

function value = percentile(samples, percentage)
    samples = sort(samples(:));
    if isempty(samples)
        value = NaN;
        return;
    end
    if isscalar(samples)
        value = samples;
        return;
    end

    position = 1 + (numel(samples) - 1) * percentage / 100;
    lowerIdx = floor(position);
    upperIdx = ceil(position);
    interpolation = position - lowerIdx;
    value = samples(lowerIdx) + interpolation * ...
        (samples(upperIdx) - samples(lowerIdx));
end

function value = normal_quantile(probability)
    value = -sqrt(2) * erfcinv(2 * probability);
end

function label = condition_label(condition)
    if condition.Scenario == "AWGN"
        label = sprintf('AWGN | Eb/No=%g dB', condition.EbNoDb);
    else
        label = sprintf('%s | K=%g | fd=%g Hz | Eb/No=%g dB', ...
            condition.Scenario, condition.KFactor, ...
            condition.DopplerHz, condition.EbNoDb);
    end
end

function seed = frame_seed(baseSeed, conditionIdx, frameIdx)
    seed = double(baseSeed) + 100000 * double(conditionIdx) + ...
        double(frameIdx);
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

function validate_configuration(cfg)
    allowedScenarios = ["AWGN" "LOS" "NLOS"];
    if isempty(cfg.Scenarios) || any(~ismember(cfg.Scenarios, allowedScenarios))
        error('Scenarios supportati: AWGN, LOS e NLOS.');
    end
    if numel(unique(cfg.Scenarios)) ~= numel(cfg.Scenarios)
        error('Scenarios non deve contenere duplicati.');
    end

    validateattributes(cfg.EbNoDb, {'numeric'}, ...
        {'vector', 'nonempty', 'real', 'finite'});
    validateattributes(cfg.LOSKFactors, {'numeric'}, ...
        {'vector', 'real', 'finite', 'nonnegative'});
    validateattributes(cfg.NLOSKFactors, {'numeric'}, ...
        {'vector', 'real', 'finite', 'nonnegative'});
    validateattributes(cfg.DopplerHz, {'numeric'}, ...
        {'vector', 'nonempty', 'real', 'finite', 'nonnegative'});
    validateattributes(cfg.FramesPerCondition, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    validateattributes(cfg.RandomSeed, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative'});
    validateattributes(cfg.WarmupIterations, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative'});
    validateattributes(cfg.SynchronizationThreshold, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'nonnegative'});
    validateattributes(cfg.ConfidenceLevel, {'numeric'}, ...
        {'scalar', 'real', '>', 0, '<', 1});
    if any(cfg.Scenarios == "LOS") && isempty(cfg.LOSKFactors)
        error('LOSKFactors non puo essere vuoto quando si valuta LOS.');
    end
    if any(cfg.Scenarios == "NLOS") && isempty(cfg.NLOSKFactors)
        error('NLOSKFactors non puo essere vuoto quando si valuta NLOS.');
    end
    if ~(ischar(cfg.ModelPath) || ...
            (isstring(cfg.ModelPath) && isscalar(cfg.ModelPath)))
        error('ModelPath deve essere una stringa scalare.');
    end
    if ~(islogical(cfg.MakePlots) && isscalar(cfg.MakePlots))
        error('MakePlots deve essere un valore logico scalare.');
    end
    if ~(islogical(cfg.Verbose) && isscalar(cfg.Verbose))
        error('Verbose deve essere un valore logico scalare.');
    end
end
