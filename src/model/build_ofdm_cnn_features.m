function XFrame = build_ofdm_cnn_features(rxGrid, carrierMap, pilotSymbols, ...
        numInputChannels, channelEstimate)
% rxGrid: [numCarrierAttivi x numSimboliOFDM].
% pilotSymbols: [numPiloti x numSimboliOFDM].
% XFrame: [numInputChannels x NFFT].
%
% Canali: raw I/Q, ZF I/Q, stima L-LTF I/Q, modulo del canale,
% maschere piloti/attivi e coordinata normalizzata della sottoportante.

    if nargin < 4 || isempty(numInputChannels)
        numInputChannels = 10;
    end
    if numInputChannels < 2 || numInputChannels > 10
        error('numInputChannels deve essere compreso tra 2 e 10.');
    end

    validate_carrier_map(carrierMap);
    if size(rxGrid, 1) ~= carrierMap.NumActiveCarriers
        error('rxGrid ha %d righe, attese %d carrier attivi.', ...
            size(rxGrid, 1), carrierMap.NumActiveCarriers);
    end
    if size(pilotSymbols, 1) ~= numel(carrierMap.PilotActiveIdx) || ...
            size(pilotSymbols, 2) ~= size(rxGrid, 2)
        error('pilotSymbols deve essere [numPiloti x numSimboli].');
    end

    XFrame = zeros(numInputChannels, carrierMap.NumSubcarriers, 'single');

    pilotActiveIdx = carrierMap.PilotActiveIdx;
    activeGlobalIdx = carrierMap.ActiveGlobalIdx;

    rxGridForRaw = rxGrid;
    rxGridForRaw(pilotActiveIdx, :) = ...
        rxGridForRaw(pilotActiveIdx, :) ./ pilotSymbols;
    avgRaw = mean(rxGridForRaw, 2);
    XFrame(1, activeGlobalIdx) = single(real(avgRaw)).';
    XFrame(2, activeGlobalIdx) = single(imag(avgRaw)).';

    if nargin < 5
        channelEstimate = [];
    end
    if numInputChannels >= 4
        if isempty(channelEstimate)
            channelEstimate = estimate_channel_from_80211a_pilots( ...
                rxGrid, carrierMap, pilotSymbols);
        end
        if isvector(channelEstimate)
            channelEstimate = repmat(channelEstimate(:), 1, size(rxGrid, 2));
        end
        if ~isequal(size(channelEstimate), size(rxGrid))
            error('channelEstimate deve essere [numCarrierAttivi x numSimboli].');
        end
        eqGrid = rxGrid ./ channelEstimate;
        eqGrid(pilotActiveIdx, :) = eqGrid(pilotActiveIdx, :) ./ pilotSymbols;
        avgEq = mean(eqGrid, 2);
        XFrame(3, activeGlobalIdx) = single(real(avgEq)).';
        XFrame(4, activeGlobalIdx) = single(imag(avgEq)).';
    end

    if numInputChannels >= 5
        if isempty(channelEstimate)
            channelEstimate = estimate_channel_from_80211a_pilots( ...
                rxGrid, carrierMap, pilotSymbols);
        end
        avgChannel = mean(channelEstimate, 2);
        XFrame(5, activeGlobalIdx) = single(real(avgChannel)).';
    end

    if numInputChannels >= 6
        avgChannel = mean(channelEstimate, 2);
        XFrame(6, activeGlobalIdx) = single(imag(avgChannel)).';
    end

    if numInputChannels >= 7
        avgMagnitude = mean(abs(channelEstimate), 2);
        XFrame(7, activeGlobalIdx) = single(avgMagnitude).';
    end

    if numInputChannels >= 8
        XFrame(8, carrierMap.PilotGlobalIdx) = 1;
    end

    if numInputChannels >= 9
        XFrame(9, activeGlobalIdx) = 1;
    end

    if numInputChannels >= 10
        carrierCoord = double(carrierMap.GlobalSubcarrierIdx);
        maxAbsCoord = max(abs(carrierCoord));
        if maxAbsCoord <= 0
            maxAbsCoord = 1;
        end
        XFrame(10, :) = single(carrierCoord ./ maxAbsCoord);
    end
end

function channelEstimate = estimate_channel_from_80211a_pilots(rxGrid, carrierMap, pilotSymbols)
    pilotActiveIdx = carrierMap.PilotActiveIdx(:);
    pilotAxis = double(carrierMap.PilotSubcarrierIdx(:));
    activeAxis = double(carrierMap.ActiveSubcarrierIdx(:));

    pilotEstimate = rxGrid(pilotActiveIdx, :) ./ pilotSymbols;
    channelEstimate = interp1(pilotAxis, pilotEstimate, activeAxis, 'linear', 'extrap');

    minMagnitude = 1e-4;
    tooSmall = abs(channelEstimate) < minMagnitude;
    if any(tooSmall(:))
        phases = exp(1i * angle(channelEstimate(tooSmall)));
        phases(abs(phases) == 0) = 1;
        channelEstimate(tooSmall) = minMagnitude .* phases;
    end
end

function validate_carrier_map(carrierMap)
    requiredFields = {'NumSubcarriers', 'NumActiveCarriers', 'ActiveGlobalIdx', ...
        'ActiveSubcarrierIdx', 'PilotActiveIdx', 'PilotGlobalIdx', ...
        'PilotSubcarrierIdx'};
    for k = 1:numel(requiredFields)
        if ~isfield(carrierMap, requiredFields{k})
            error('carrierMap manca il campo %s.', requiredFields{k});
        end
    end
end
