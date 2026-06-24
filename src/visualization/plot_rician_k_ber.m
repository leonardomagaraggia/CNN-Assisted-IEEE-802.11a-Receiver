function [figureHandle, outputPath, chartConfig] = plot_rician_k_ber(results, varargin)
%PLOT_RICIAN_K_BER Grafico BER raffinato in stile IEEE per coppie CNN/MMSE.

    parser = inputParser();
    addParameter(parser, 'Export', true);
    addParameter(parser, 'Visible', true);
    addParameter(parser, 'OutputDir', project_paths(true).Charts);
    parse(parser, varargin{:});
    cfg = parser.Results;

    visibility = "off";
    if cfg.Visible
        visibility = "on";
    end

    colors = lines(numel(results.KFactorDb));
    figureHandle = figure('Color', 'white', 'Visible', visibility, ...
        'Name', 'BER vs SNR - CNN/MMSE multi-K', ...
        'Units', 'inches', 'Position', [1 1 7.16 4.65], ...
        'PaperPositionMode', 'auto');
    ax = axes(figureHandle);
    ax.YScale = 'log';
    hold(ax, 'on');

    markerStep = max(1, ceil(numel(results.SNRdB) / 9));
    markerIndices = 1:markerStep:numel(results.SNRdB);
    for kIdx = 1:numel(results.KFactorDb)
        color = colors(kIdx, :);
        semilogy(ax, results.SNRdB, ...
            squeeze(results.PlotBER(kIdx, :, 1)), '-', ...
            'Color', color, 'LineWidth', 1, 'Marker', 'o', ...
            'MarkerIndices', markerIndices, 'MarkerSize', 4, ...
            'MarkerFaceColor', 'white', ...
            'DisplayName', sprintf('CNN, K = %g dB', ...
            results.KFactorDb(kIdx)));
        semilogy(ax, results.SNRdB, ...
            squeeze(results.PlotBER(kIdx, :, 2)), '--', ...
            'Color', color, 'LineWidth', 1, 'Marker', 's', ...
            'MarkerIndices', markerIndices, 'MarkerSize', 3.5, ...
            'MarkerFaceColor', 'white', ...
            'DisplayName', sprintf('MMSE, K = %g dB', ...
            results.KFactorDb(kIdx)));
    end

    xlim(ax, [0 35]);
    ylim(ax, [5e-7 1]);
    xticks(ax, 0:5:35);
    ax.XAxis.MinorTickValues = setdiff(0:2:34, 0:5:35);
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    grid(ax, 'on');
    grid(ax, 'minor');
    ax.GridAlpha = 0.18;
    ax.MinorGridAlpha = 0.09;
    ax.GridLineStyle = '-';
    ax.MinorGridLineStyle = ':';
    ax.Layer = 'top';
    ax.Box = 'on';
    ax.LineWidth = 0.75;
    ax.FontName = 'Times New Roman';
    ax.FontSize = 9;
    ax.TickDir = 'in';
    xlabel(ax, 'SNR (E_b/N_0) [dB]', 'FontName', 'Times New Roman');
    ylabel(ax, 'Bit error rate (BER)', 'FontName', 'Times New Roman');
    title(ax, 'Rician channel: CNN and MMSE equalization', ...
        'FontName', 'Times New Roman', 'FontWeight', 'normal');
    legend(ax, 'Location', 'northeast', 'NumColumns', 2, ...
        'FontName', 'Times New Roman', 'FontSize', 8, ...
        'Box', 'on', 'Color', 'white');
    hold(ax, 'off');

    chartConfig = struct();
    chartConfig.XLimitsDb = [0 35];
    chartConfig.MajorXTicksDb = 0:5:35;
    chartConfig.MinorXTicksDb = setdiff(0:2:34, 0:5:35);
    chartConfig.YLimits = [5e-7 1];
    chartConfig.LineWidthPt = 1;
    chartConfig.CNNLineStyle = '-';
    chartConfig.MMSELineStyle = '--';
    chartConfig.CNNMarker = 'o';
    chartConfig.MMSEMarker = 's';
    chartConfig.MarkerStep = markerStep;
    chartConfig.ColorOrderRGB = colors;
    chartConfig.LegendLocation = 'northeast';
    chartConfig.BackgroundColor = 'white';
    chartConfig.FontName = 'Times New Roman';
    chartConfig.AxisFontSizePt = 9;
    chartConfig.FigureSizeInches = [7.16 4.65];
    chartConfig.Caption = 'Rician channel: CNN and MMSE equalization';
    chartConfig.MajorGrid = true;
    chartConfig.MinorGrid = true;
    chartConfig.ResolutionDpi = 300;

    outputPath = "";
    if cfg.Export
        if ~isfolder(cfg.OutputDir)
            mkdir(cfg.OutputDir);
        end
        outputPath = string(fullfile(cfg.OutputDir, ...
            sprintf('ber_K_%s.png', results.Timestamp)));
        exportgraphics(figureHandle, outputPath, 'Resolution', 300, ...
            'BackgroundColor', 'white');
        figurePath = string(fullfile(cfg.OutputDir, ...
            sprintf('ber_K_%s.fig', results.Timestamp)));
        savefig(figureHandle, figurePath);
    end
end
