function results = tester(varargin)
%TESTER Entry point BER vs SNR per CNN e MMSE su canale Rician.
%   I valori predefiniti di K sono espressi in dB e sono modificabili con
%   TESTER('KFactorDb', [...]). Grafico PNG e log Base64 sono sempre
%   timestampati, quindi non sovrascrivono esecuzioni precedenti.

    rootDir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(rootDir, 'src')));

    parser = inputParser();
    parser.FunctionName = 'tester';
    addParameter(parser, 'KFactorDb', [0 10 20 33]);
    addParameter(parser, 'SNRdB', 0:2:36);
    addParameter(parser, 'MinFrames', 600);
    addParameter(parser, 'MaxFrames', 900);
    addParameter(parser, 'TargetBitErrors', 400);
    addParameter(parser, 'FrameBatchSize', 4);
    addParameter(parser, 'ModelPath', "");
    addParameter(parser, 'MaximumDopplerShift', 0);
    addParameter(parser, 'RandomSeed', 352);
    addParameter(parser, 'Visible', true);
    parse(parser, varargin{:});
    cfg = parser.Results;

    fprintf('=== BER vs SNR | CNN vs MMSE | Rician multi-K ===\n');
    results = run_rician_k_benchmark( ...
        'KFactorDb', cfg.KFactorDb, ...
        'SNRdB', cfg.SNRdB, ...
        'MinFrames', cfg.MinFrames, ...
        'MaxFrames', cfg.MaxFrames, ...
        'TargetBitErrors', cfg.TargetBitErrors, ...
        'FrameBatchSize', cfg.FrameBatchSize, ...
        'ModelPath', cfg.ModelPath, ...
        'MaximumDopplerShift', cfg.MaximumDopplerShift, ...
        'RandomSeed', cfg.RandomSeed, ...
        'Visible', cfg.Visible, ...
        'ExportChart', true, ...
        'WriteLog', true);

    fprintf('Grafico: %s\n', results.ChartPath);
    fprintf('Log Base64: %s\n', results.LogPath);
end
