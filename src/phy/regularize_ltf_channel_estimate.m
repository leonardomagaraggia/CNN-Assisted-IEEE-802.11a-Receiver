function channelEstimate = regularize_ltf_channel_estimate( ...
        rawEstimate, carrierMap, nfft, maxChannelTaps)
% Proietta la stima L-LTF su un canale corto compatibile con il CP.

    if nargin < 4 || isempty(maxChannelTaps)
        maxChannelTaps = 8;
    end
    rawEstimate = rawEstimate(:);
    if numel(rawEstimate) ~= carrierMap.NumActiveCarriers
        error('rawEstimate deve contenere una stima per ogni carrier attivo.');
    end
    if maxChannelTaps < 1 || maxChannelTaps > nfft
        error('maxChannelTaps fuori range.');
    end

    subcarriers = double(carrierMap.ActiveSubcarrierIdx(:));
    delayAxis = 0:(maxChannelTaps - 1);
    dictionary = exp(-1i * 2 * pi / nfft .* ...
        (subcarriers * delayAxis));

    ridge = 1e-5;
    impulseResponse = (dictionary' * dictionary + ...
        ridge * eye(maxChannelTaps)) \ (dictionary' * rawEstimate);
    channelEstimate = dictionary * impulseResponse;
end
