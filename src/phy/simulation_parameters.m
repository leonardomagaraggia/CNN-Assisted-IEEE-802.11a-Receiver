classdef simulation_parameters

    properties

        %% Modulazione

        ModulationOrder = 4;          % 4-QAM

        %% OFDM IEEE 802.11a

        FFTLength = 64;
        CPLength = 16;

        NumOFDMSymbols = 50;

        % Modalita legacy 802.11a a 12 Mb/s: QPSK. Il payload e lasciato
        % uncoded come richiesto, quindi il bit-rate netto simulato e 24 Mb/s.
        DataRateMbps = 12;
        UseFEC = false;
        NumTrainingSymbolsPerFrame = 12;

        NumGuardBandCarriers = [6; 5]; % [sinistra; destra]

        InsertDCNull = true;

        % Layout IEEE 802.11a: 48 dati + 4 piloti sulle 52 portanti attive.
        NumPilots = 4;
        PilotSubcarrierIndices = [-21 -7 7 21];
        PilotBasePattern = [1; 1; 1; -1];
        PilotClass = 1;

        %% Banda sottoportante e coerenza canale

        % Larghezza/spaziatura di ciascuna sottoportante OFDM.
        % Modificare SubcarrierBandwidthHz per cambiare la griglia OFDM.
        % La SampleRate viene riallineata a FFTLength * SubcarrierBandwidthHz.
        SubcarrierBandwidthHz = 312.5e3;
        SubcarrierSpacingHz = 312.5e3; % Alias derivato da SubcarrierBandwidthHz.

        SampleRate = 20e6;
        UsefulSymbolDuration = 3.2e-6; % 1 / SubcarrierSpacingHz.
        CPDuration = 0.8e-6;
        OFDMSymbolDuration = 4.0e-6;   % UsefulSymbolDuration + CPDuration.

        % Controlli di progetto:
        % - il simbolo OFDM deve essere molto piu corto del coherence time;
        % - la sottoportante deve stare sotto la coherence bandwidth stimata.
        MinOFDMSymbolsPerCoherenceTime = 10;
        MaxSubcarrierSpacingOverCoherenceBandwidth = 0.6;
        MaxExcessDelayOverCPLimit = 0.90;

        CoherenceTime = Inf;
        CoherenceBandwidthHz = Inf;
        RMSDelaySpread = 0;
        MaxExcessDelay = 0;
        OFDMSymbolsPerCoherenceTime = Inf;
        SubcarrierSpacingOverCoherenceBandwidth = 0;
        MaxExcessDelayOverCP = 0;

        %% Canale

        SNR_dB = 5;
        SNRDefinition = 'EbNo';

        ChannelType = 'Rician';

        PathDelays = [0 50e-9 110e-9]; % [1 x numPath] secondi

        AveragePathGains = [0 -8 -16]; % [1 x numPath] dB

        MaximumDopplerShift = 10;

        KFactor = 8;

        %% Dataset / esecuzione

        DatasetSize = 1e9;
        TrainingMiniBatchSize = 512;
        TestMiniBatchSize = 1024;
        UseGPUIfAvailable = false; % Training CPU/RAM.
        NumCNNInputChannels = 10;

        %% Derivati

        NumDataCarriers % Portanti OFDM utili escluse guard band e DC.
        NumPayloadCarriers
        BitsPerPayloadOFDMSymbol

    end

    methods

        function obj = simulation_parameters()
            obj = obj.refreshDerived();
        end

        function obj = refreshDerived(obj)
            if ~isempty(obj.PilotSubcarrierIndices)
                obj.NumPilots = numel(obj.PilotSubcarrierIndices);
            end
            obj.validateBaseParameters();

            obj.SubcarrierSpacingHz = obj.SubcarrierBandwidthHz;
            obj.SampleRate = obj.FFTLength * obj.SubcarrierBandwidthHz;
            obj.UsefulSymbolDuration = 1 / obj.SubcarrierSpacingHz;
            obj.CPDuration = obj.CPLength / obj.SampleRate;
            obj.OFDMSymbolDuration = obj.UsefulSymbolDuration + obj.CPDuration;

            obj.NumDataCarriers = ...
                obj.FFTLength ...
                - sum(obj.NumGuardBandCarriers) ...
                - double(obj.InsertDCNull);
            obj.NumPayloadCarriers = obj.NumDataCarriers - obj.NumPilots;
            obj.BitsPerPayloadOFDMSymbol = ...
                obj.NumPayloadCarriers * log2(obj.ModulationOrder);

            obj.CoherenceTime = obj.estimateCoherenceTime(obj.MaximumDopplerShift);
            [obj.CoherenceBandwidthHz, obj.RMSDelaySpread, obj.MaxExcessDelay] = ...
                obj.estimateDelayMetrics( ...
                obj.PathDelays, obj.AveragePathGains);

            if isfinite(obj.CoherenceTime)
                obj.OFDMSymbolsPerCoherenceTime = ...
                    obj.CoherenceTime / obj.OFDMSymbolDuration;
            else
                obj.OFDMSymbolsPerCoherenceTime = Inf;
            end

            if isfinite(obj.CoherenceBandwidthHz)
                obj.SubcarrierSpacingOverCoherenceBandwidth = ...
                    obj.SubcarrierSpacingHz / obj.CoherenceBandwidthHz;
            else
                obj.SubcarrierSpacingOverCoherenceBandwidth = 0;
            end

            if obj.CPDuration > 0
                obj.MaxExcessDelayOverCP = obj.MaxExcessDelay / obj.CPDuration;
            else
                obj.MaxExcessDelayOverCP = Inf;
            end
        end

        function pilotIdx = getPilotIndices(obj, numCarriers)
            if nargin < 2 || isempty(numCarriers)
                numCarriers = obj.NumDataCarriers;
            end

            p = obj.refreshDerived();

            if p.NumPilots < 1
                error('NumPilots deve essere almeno 1.');
            end
            if p.NumPilots >= numCarriers
                error('NumPilots (%d) deve essere minore delle portanti attive (%d).', ...
                    p.NumPilots, numCarriers);
            end

            if numCarriers == p.NumDataCarriers && ~isempty(p.PilotSubcarrierIndices)
                activeSubcarrierIdx = p.getActiveSubcarrierIndices();
                [isPilot, pilotIdx] = ismember(double(p.PilotSubcarrierIndices(:)).', ...
                    activeSubcarrierIdx);
                if ~all(isPilot)
                    error('PilotSubcarrierIndices contiene toni non attivi: %s.', ...
                        mat2str(p.PilotSubcarrierIndices(~isPilot)));
                end
                if numel(pilotIdx) ~= p.NumPilots
                    error('Layout piloti non valido: attesi %d piloti, trovati %d.', ...
                        p.NumPilots, numel(pilotIdx));
                end
                return;
            end

            pilotIdx = round(linspace(1, numCarriers, p.NumPilots + 2));
            pilotIdx = unique(pilotIdx(2:end-1), 'stable');

            if numel(pilotIdx) ~= p.NumPilots
                pilotIdx = round(linspace(2, numCarriers - 1, p.NumPilots));
                pilotIdx = unique(pilotIdx, 'stable');
            end

            if numel(pilotIdx) ~= p.NumPilots
                error('Impossibile costruire %d piloti distinti su %d portanti.', ...
                    p.NumPilots, numCarriers);
            end
        end

        function subcarrierIdx = getSubcarrierIndices(obj)
            p = obj.refreshDerived();
            if mod(p.FFTLength, 2) ~= 0
                error('La mappa subcarrier IEEE 802.11a richiede FFTLength pari.');
            end
            subcarrierIdx = (-p.FFTLength/2):(p.FFTLength/2 - 1);
        end

        function activeSubcarrierIdx = getActiveSubcarrierIndices(obj)
            p = obj.refreshDerived();
            subcarrierIdx = p.getSubcarrierIndices();
            activeSubcarrierIdx = subcarrierIdx(p.getActiveCarrierIndices());
        end

        function pilotSymbols = getPilotSymbols(obj, numSymbols, symbolOffset)
            if nargin < 2 || isempty(numSymbols)
                numSymbols = obj.NumOFDMSymbols;
            end
            if nargin < 3 || isempty(symbolOffset)
                symbolOffset = 0;
            end
            if numSymbols < 1 || abs(numSymbols - round(numSymbols)) > eps
                error('numSymbols deve essere un intero positivo.');
            end

            p = obj.refreshDerived();
            basePattern = double(p.PilotBasePattern(:));
            if numel(basePattern) ~= p.NumPilots
                error('PilotBasePattern deve avere %d elementi.', p.NumPilots);
            end

            polarity = p.ieee80211aPilotPolaritySequence();
            seqIdx = mod(double(symbolOffset) + (0:numSymbols-1), numel(polarity)) + 1;
            pilotSymbols = complex(basePattern * polarity(seqIdx));
        end

        function activeIdx = getActiveCarrierIndices(obj)
            p = obj.refreshDerived();
            numSubcarriers = p.FFTLength;
            guardBands = double(p.NumGuardBandCarriers(:));

            activeMask = true(numSubcarriers, 1);
            activeMask(1:guardBands(1)) = false;
            if guardBands(2) > 0
                activeMask(numSubcarriers-guardBands(2)+1:numSubcarriers) = false;
            end

            if p.InsertDCNull
                dcIdx = floor(numSubcarriers / 2) + 1;
                activeMask(dcIdx) = false;
            end

            activeIdx = find(activeMask).';
            if numel(activeIdx) ~= p.NumDataCarriers
                error('Mappa carrier non valida: attesi %d carrier utili, trovati %d.', ...
                    p.NumDataCarriers, numel(activeIdx));
            end
        end

        function carrierMap = getCarrierMap(obj)
            p = obj.refreshDerived();
            subcarrierIdx = p.getSubcarrierIndices();
            activeIdx = p.getActiveCarrierIndices();
            pilotActiveIdx = p.getPilotIndices(p.NumDataCarriers);
            payloadActiveIdx = setdiff(1:p.NumDataCarriers, pilotActiveIdx);

            carrierMap = struct();
            carrierMap.NumSubcarriers = p.FFTLength;
            carrierMap.NumActiveCarriers = p.NumDataCarriers;
            carrierMap.GlobalSubcarrierIdx = subcarrierIdx;
            carrierMap.ActiveGlobalIdx = activeIdx;
            carrierMap.ActiveSubcarrierIdx = subcarrierIdx(activeIdx);
            carrierMap.InactiveGlobalIdx = setdiff(1:p.FFTLength, activeIdx);
            carrierMap.PilotActiveIdx = pilotActiveIdx;
            carrierMap.PilotGlobalIdx = activeIdx(pilotActiveIdx);
            carrierMap.PilotSubcarrierIdx = subcarrierIdx(activeIdx(pilotActiveIdx));
            carrierMap.PayloadActiveIdx = payloadActiveIdx;
            carrierMap.PayloadGlobalIdx = activeIdx(payloadActiveIdx);
            carrierMap.PayloadSubcarrierIdx = subcarrierIdx(activeIdx(payloadActiveIdx));
        end

        function report = ofdmChannelReport(obj)
            p = obj.refreshDerived();
            issues = strings(0, 1);

            isTimeCoherent = true;
            if isfinite(p.OFDMSymbolsPerCoherenceTime)
                isTimeCoherent = ...
                    p.OFDMSymbolsPerCoherenceTime >= p.MinOFDMSymbolsPerCoherenceTime;
                if ~isTimeCoherent
                    issues(end+1, 1) = sprintf( ...
                        'Simboli/coherence-time %.2f < minimo %.2f.', ...
                        p.OFDMSymbolsPerCoherenceTime, ...
                        p.MinOFDMSymbolsPerCoherenceTime);
                end
            end

            isFrequencyCoherent = true;
            if isfinite(p.CoherenceBandwidthHz)
                isFrequencyCoherent = ...
                    p.SubcarrierSpacingOverCoherenceBandwidth <= ...
                    p.MaxSubcarrierSpacingOverCoherenceBandwidth;
                if ~isFrequencyCoherent
                    issues(end+1, 1) = sprintf( ...
                        'SubcarrierSpacing/CoherenceBandwidth %.3f > massimo %.3f.', ...
                        p.SubcarrierSpacingOverCoherenceBandwidth, ...
                        p.MaxSubcarrierSpacingOverCoherenceBandwidth);
                end
            end

            isGuardIntervalSafe = true;
            if isfinite(p.MaxExcessDelayOverCP)
                isGuardIntervalSafe = ...
                    p.MaxExcessDelayOverCP <= p.MaxExcessDelayOverCPLimit;
                if ~isGuardIntervalSafe
                    issues(end+1, 1) = sprintf( ...
                        'MaxExcessDelay/CP %.3f > massimo %.3f.', ...
                        p.MaxExcessDelayOverCP, p.MaxExcessDelayOverCPLimit);
                end
            end

            report = struct();
            report.SubcarrierBandwidthHz = p.SubcarrierBandwidthHz;
            report.SampleRate = p.SampleRate;
            report.UsefulSymbolDuration = p.UsefulSymbolDuration;
            report.CPDuration = p.CPDuration;
            report.OFDMSymbolDuration = p.OFDMSymbolDuration;
            report.CoherenceTime = p.CoherenceTime;
            report.CoherenceBandwidthHz = p.CoherenceBandwidthHz;
            report.RMSDelaySpread = p.RMSDelaySpread;
            report.MaxExcessDelay = p.MaxExcessDelay;
            report.OFDMSymbolsPerCoherenceTime = p.OFDMSymbolsPerCoherenceTime;
            report.SubcarrierSpacingOverCoherenceBandwidth = ...
                p.SubcarrierSpacingOverCoherenceBandwidth;
            report.MaxExcessDelayOverCP = p.MaxExcessDelayOverCP;
            report.IsTimeCoherent = isTimeCoherent;
            report.IsFrequencyCoherent = isFrequencyCoherent;
            report.IsGuardIntervalSafe = isGuardIntervalSafe;
            report.IsValid = isTimeCoherent && isFrequencyCoherent && isGuardIntervalSafe;
            report.Issues = issues;
        end

        function assertOFDMChannelConsistency(obj, contextLabel)
            if nargin < 2
                contextLabel = "";
            end

            report = obj.ofdmChannelReport();
            if report.IsValid
                return;
            end

            error('OFDM non coerente per %s: %s', char(contextLabel), ...
                strjoin(report.Issues, ' | '));
        end

    end

    methods (Access = private)

        function validateBaseParameters(obj)
            if obj.ModulationOrder ~= 4
                error('Questo progetto e configurato solo per 4-QAM.');
            end
            if obj.DataRateMbps ~= 12
                error('Il frame e configurato per la modalita 802.11a a 12 Mb/s.');
            end
            if obj.UseFEC
                error('Questa pipeline implementa il payload senza FEC.');
            end
            if obj.FFTLength ~= 64
                error('IEEE 802.11a richiede FFTLength=64.');
            end
            if obj.CPLength ~= 16
                error('IEEE 802.11a richiede CPLength=16.');
            end
            if ~isequal(double(obj.NumGuardBandCarriers(:)), [6; 5])
                error('IEEE 802.11a richiede NumGuardBandCarriers=[6;5].');
            end
            if ~obj.InsertDCNull
                error('IEEE 802.11a richiede InsertDCNull=true.');
            end
            if obj.SubcarrierBandwidthHz <= 0
                error('SubcarrierBandwidthHz deve essere positivo.');
            end
            if numel(obj.NumGuardBandCarriers) ~= 2
                error('NumGuardBandCarriers deve avere due elementi.');
            end
            if numel(obj.PilotBasePattern) ~= obj.NumPilots
                error('PilotBasePattern deve avere lo stesso numero di elementi di NumPilots.');
            end
            if any(obj.PilotSubcarrierIndices == 0)
                error('I piloti IEEE 802.11a non possono occupare la DC.');
            end
            if obj.NumTrainingSymbolsPerFrame < 1 || ...
                    abs(obj.NumTrainingSymbolsPerFrame - round(obj.NumTrainingSymbolsPerFrame)) > eps
                error('NumTrainingSymbolsPerFrame deve essere un intero positivo.');
            end
        end

        function coherenceTime = estimateCoherenceTime(~, maxDopplerHz)
            if isempty(maxDopplerHz)
                coherenceTime = Inf;
            else
                maxDopplerHz = max(double(maxDopplerHz(:)));
                if maxDopplerHz <= 0
                    coherenceTime = Inf;
                else
                    coherenceTime = 0.423 / maxDopplerHz;
                end
            end
        end

        function [coherenceBandwidth, rmsDelaySpread, maxExcessDelay] = ...
                estimateDelayMetrics(~, pathDelays, pathGainsDb)
            if isempty(pathDelays) || numel(pathDelays) < 2
                coherenceBandwidth = Inf;
                rmsDelaySpread = 0;
                maxExcessDelay = 0;
                return;
            end

            delays = double(pathDelays(:));
            gains = double(pathGainsDb(:));
            if numel(gains) ~= numel(delays)
                error('PathDelays e AveragePathGains devono avere la stessa lunghezza.');
            end

            powers = 10.^(gains / 10);
            powers = powers / sum(powers);
            meanDelay = sum(powers .* delays);
            rmsDelaySpread = sqrt(sum(powers .* (delays - meanDelay).^2));
            maxExcessDelay = max(delays) - min(delays);

            if rmsDelaySpread <= eps
                coherenceBandwidth = Inf;
            else
                coherenceBandwidth = 1 / (5 * rmsDelaySpread);
            end
        end

        function polarity = ieee80211aPilotPolaritySequence(~)
            polarity = [ ...
                1 1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 -1 1 ...
                -1 -1 1 1 -1 1 1 -1 1 1 1 1 1 1 -1 1 ...
                1 1 -1 1 1 -1 -1 1 1 1 -1 1 -1 -1 -1 1 ...
                -1 1 -1 -1 1 -1 -1 1 1 1 1 1 -1 -1 1 1 ...
                -1 -1 1 -1 1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 ...
                -1 1 -1 -1 1 -1 1 1 1 1 -1 1 -1 1 -1 1 ...
                -1 -1 -1 -1 -1 1 -1 1 1 -1 1 -1 1 1 1 -1 ...
                -1 1 -1 -1 -1 1 1 1 -1 -1 -1 -1 -1 -1 -1];
            if numel(polarity) ~= 127
                error('Sequenza piloti IEEE 802.11a non valida.');
            end
        end

    end

end
