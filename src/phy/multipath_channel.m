function [rxSignal, channelInfo] = multipath_channel(txSignal, p)
    % txSignal, rxSignal: [numCampioni x 1] nel dominio del tempo.

    if ismethod(p, 'refreshDerived')
        p = p.refreshDerived();
    end

    switch upper(string(p.ChannelType))
        case "AWGN"
            fadedSignal = txSignal;

        case "RICIAN"
            channel = comm.RicianChannel( ...
                'SampleRate',          p.SampleRate, ...
                'KFactor',             p.KFactor, ...
                'PathDelays',          p.PathDelays, ...
                'AveragePathGains',    p.AveragePathGains, ...
                'MaximumDopplerShift', p.MaximumDopplerShift, ...
                'PathGainsOutputPort', false, ...
                'Visualization',       'Off');
            fadedSignal = channel(txSignal);

        otherwise
            error('Tipo di canale non supportato: %s.', char(p.ChannelType));
    end

    signalMask = abs(txSignal(:)) > sqrt(eps);
    if any(signalMask)
        referencePower = mean(abs(txSignal(signalMask)).^2);
    else
        referencePower = mean(abs(txSignal(:)).^2);
    end
    if strcmpi(char(p.SNRDefinition), 'EbNo')
        % Eb/No della costellazione QPSK sulle sottoportanti. La conversione
        % include la scala IFFT dovuta ai 52 toni attivi; CP e preambolo sono
        % overhead di frame e non cambiano l'Eb della costellazione.
        constellationBitsPerSample = log2(p.ModulationOrder) * ...
            p.NumDataCarriers / p.FFTLength;
        sampleSnrDb = p.SNR_dB + 10 * log10(constellationBitsPerSample);
    else
        sampleSnrDb = p.SNR_dB;
    end
    noiseVariance = referencePower / (10^(sampleSnrDb / 10));
    noise = sqrt(noiseVariance / 2) .* ...
        (randn(size(fadedSignal)) + 1i * randn(size(fadedSignal)));
    rxSignal = fadedSignal + noise;

    channelInfo = struct();
    channelInfo.ReferencePower = referencePower;
    channelInfo.NoiseVarianceTime = noiseVariance;
    channelInfo.SampleSNRdB = sampleSnrDb;
end
