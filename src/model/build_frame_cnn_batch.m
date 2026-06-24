function X = build_frame_cnn_batch(rxFrame, txFrame, carrierMap, numInputChannels)
% Costruisce un esempio CNN per ogni simbolo payload dello stesso frame.

    numSymbols = size(rxFrame.ActiveGrid, 2);
    X = zeros(numInputChannels, carrierMap.NumSubcarriers, numSymbols, 'single');
    cnnChannelEstimate = regularize_ltf_channel_estimate( ...
        rxFrame.ChannelEstimate, carrierMap, carrierMap.NumSubcarriers, 8);

    for symbolIdx = 1:numSymbols
        X(:, :, symbolIdx) = build_ofdm_cnn_features( ...
            rxFrame.ActiveGrid(:, symbolIdx), carrierMap, ...
            txFrame.PilotSymbols(:, symbolIdx), numInputChannels, ...
            cnnChannelEstimate);
    end
end
