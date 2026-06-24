function figureHandle = plot_ber_results(results)
% Grafico BER a schermo con riferimenti teorici, senza esportazione.

    colors = lines(max(numel(results.MethodNames), 1));
    figureHandle = figure('Position', [80 80 1180 420]);
    tiledlayout(1, numel(results.Scenarios), 'TileSpacing', 'compact');

    for scenarioIdx = 1:numel(results.Scenarios)
        ax = nexttile;
        set(ax, 'YScale', 'log');
        hold(ax, 'on');

        for methodIdx = 1:numel(results.MethodNames)
            semilogy(ax, results.EbNoDb, ...
                squeeze(results.PlotBER(scenarioIdx, :, methodIdx)), ...
                'o-', 'LineWidth', 1.5, 'MarkerSize', 5, ...
                'Color', colors(methodIdx, :), ...
                'DisplayName', results.MethodNames(methodIdx));
        end

        switch results.Scenarios(scenarioIdx)
            case "AWGN"
                theory = berawgn(results.EbNoDb, 'qam', 4);
                theoryLabel = "Teoria AWGN";
            case "LOS"
                theory = berfading(results.EbNoDb, 'qam', 4, 1, ...
                    results.KFactorLOS);
                theoryLabel = "Rician flat di riferimento";
            otherwise
                theory = berfading(results.EbNoDb, 'qam', 4, 1);
                theoryLabel = "Rayleigh flat di riferimento";
        end
        semilogy(ax, results.EbNoDb, max(theory, 1e-8), ...
            'k--', 'LineWidth', 1.2, 'DisplayName', theoryLabel);

        title(ax, results.Scenarios(scenarioIdx));
        xlabel(ax, 'E_b/N_0 (dB)');
        ylabel(ax, 'BER');
        grid(ax, 'on');
        ylim(ax, [1e-7 1]);
        legend(ax, 'Location', 'southwest');
        hold(ax, 'off');
    end

    sgtitle("Frame 802.11a QPSK uncoded: " + ...
        strjoin(results.MethodNames, " vs ") + " e riferimenti teorici");
end
