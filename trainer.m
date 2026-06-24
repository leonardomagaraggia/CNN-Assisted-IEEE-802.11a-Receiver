function modelPath = trainer(varargin)
%TRAINER Entry point principale per l'addestramento CNN.
%   MODEL_PATH = TRAINER(Name, Value, ...) inoltra le opzioni a
%   train_frame_cnn. Eseguire dalla root del progetto.

    rootDir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(rootDir, 'src')));

    fprintf('=== CNN TRAINER | IEEE 802.11a QPSK ===\n');
    modelPath = train_frame_cnn(varargin{:});
end
