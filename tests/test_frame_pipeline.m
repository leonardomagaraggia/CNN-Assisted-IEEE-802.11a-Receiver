function test_frame_pipeline()
% Smoke test end-to-end della nuova logica a frame.

    rootDir = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(rootDir, 'src')));

    rng(901);
    params = simulation_parameters().refreshDerived();
    carrierMap = params.getCarrierMap();
    frameTool = IEEE80211aFrame(params);

    [tx, frame] = frameTool.createRandomFrame();
    payloadBySymbol = reshape(frame.PayloadBits, [], params.NumOFDMSymbols);
    assert(any(diff(payloadBySymbol, 1, 2) ~= 0, 'all'), ...
        'Il payload non cambia tra simboli OFDM.');

    awgnParams = configure_test_channel(params, 60, 'AWGN', 15);
    rx = frameTool.receive(multipath_channel(tx, awgnParams));
    zf = frameTool.equalize(rx, 'ZF');
    zfBits = qamdemod(zf.PayloadSymbols(:), params.ModulationOrder, ...
        'gray', 'UnitAveragePower', true, 'OutputType', 'bit');
    assert(sum(uint8(zfBits(:)) ~= frame.PayloadBits(:)) == 0, ...
        'ZF AWGN quasi-noiseless non ha BER zero.');
    assert(rx.SynchronizationMetric > 0.5, ...
        'Metrica di sincronizzazione AWGN troppo bassa.');

    X = build_frame_cnn_batch(rx, frame, carrierMap, ...
        params.NumCNNInputChannels);
    assert(isequal(size(X), [params.NumCNNInputChannels, ...
        params.FFTLength, params.NumOFDMSymbols]), ...
        'Dimensione feature CNN non valida.');

    nlosParams = configure_test_channel(params, 80, 'NLOS', 15);
    bitErrors = 0;
    totalBits = 0;
    for frameIdx = 1:100
        [tx, frame] = frameTool.createRandomFrame();
        rx = frameTool.receive(multipath_channel(tx, nlosParams));
        zf = frameTool.equalize(rx, 'ZF');
        bits = qamdemod(zf.PayloadSymbols(:), params.ModulationOrder, ...
            'gray', 'UnitAveragePower', true, 'OutputType', 'bit');
        bitErrors = bitErrors + sum(uint8(bits(:)) ~= frame.PayloadBits(:));
        totalBits = totalBits + numel(frame.PayloadBits);
    end
    assert(bitErrors / totalBits < 1e-5, ...
        'Persistenza di un floor quasi-noiseless nella pipeline NLOS.');

    fprintf("test_frame_pipeline: PASS | BER NLOS quasi-noiseless %.3e\n", ...
        bitErrors / totalBits);
end
