function params = configure_test_channel(baseParams, ebNoDb, channelMode, ...
        kFactorLOS, varargin)
% Configurazioni riproducibili usate da tutti i benchmark BER.

    if nargin < 4 || isempty(kFactorLOS)
        kFactorLOS = 15;
    end
    parser = inputParser();
    addParameter(parser, 'KFactor', []);
    addParameter(parser, 'MaximumDopplerShift', []);
    parse(parser, varargin{:});
    cfg = parser.Results;

    params = baseParams;
    params.SNR_dB = ebNoDb;

    switch upper(string(channelMode))
        case "AWGN"
            params.ChannelType = 'AWGN';
            params.KFactor = 0;
            params.MaximumDopplerShift = 0;
            params.PathDelays = 0;
            params.AveragePathGains = 0;

        case "LOS"
            params.ChannelType = 'Rician';
            params.KFactor = kFactorLOS;
            % Le curve BER teoriche assumono block fading sul frame.
            params.MaximumDopplerShift = 0;
            params.PathDelays = [0, 30e-9];
            params.AveragePathGains = [0, -10];

        case "NLOS"
            params.ChannelType = 'Rician';
            params.KFactor = 1e-4;
            params.MaximumDopplerShift = 0;
            params.PathDelays = [0, 60e-9, 140e-9];
            params.AveragePathGains = [0, -2, -6];

        otherwise
            error('Scenario canale non supportato: %s.', char(channelMode));
    end

    if ~strcmpi(params.ChannelType, 'AWGN')
        if ~isempty(cfg.KFactor)
            if ~isscalar(cfg.KFactor) || cfg.KFactor < 0
                error('KFactor deve essere uno scalare non negativo.');
            end
            params.KFactor = double(cfg.KFactor);
        end
        if ~isempty(cfg.MaximumDopplerShift)
            if ~isscalar(cfg.MaximumDopplerShift) || ...
                    cfg.MaximumDopplerShift < 0
                error('MaximumDopplerShift deve essere uno scalare non negativo.');
            end
            params.MaximumDopplerShift = double(cfg.MaximumDopplerShift);
        end
    end

    params = params.refreshDerived();
end
