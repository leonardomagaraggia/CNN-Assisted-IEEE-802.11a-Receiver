function paths = project_paths(createOutputDirs)
%PROJECT_PATHS Percorsi assoluti indipendenti dalla current folder MATLAB.

    if nargin < 1
        createOutputDirs = false;
    end

    utilsDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(fileparts(utilsDir));
    paths = struct();
    paths.Root = string(rootDir);
    paths.Source = string(fullfile(rootDir, 'src'));
    paths.Checkpoints = string(fullfile(rootDir, 'CHECKPOINTS'));
    paths.Charts = string(fullfile(rootDir, 'CHARTS'));
    paths.Logs = string(fullfile(rootDir, 'LOGS'));
    paths.Tests = string(fullfile(rootDir, 'tests'));

    if createOutputDirs
        outputDirs = [paths.Checkpoints paths.Charts paths.Logs];
        for outputDir = outputDirs
            if ~isfolder(outputDir)
                [ok, message] = mkdir(outputDir);
                if ~ok
                    error('Impossibile creare %s: %s', outputDir, message);
                end
            end
        end
    end
end
