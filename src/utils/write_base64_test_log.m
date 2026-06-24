function outputPath = write_base64_test_log(results, outputDir)
%WRITE_BASE64_TEST_LOG Serializza in JSON UTF-8 e codifica in Base64.

    if nargin < 2 || strlength(string(outputDir)) == 0
        outputDir = project_paths(true).Logs;
    end
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    record = struct();
    record.schema = 'cnn-ofdm-ber-log/v1';
    record.encoding = 'base64(json/utf-8)';
    record.timestamp = char(results.Timestamp);
    record.checkpoint = char(results.ModelPath);
    record.checkpointSelection = make_json_safe(results.CheckpointSelection);
    record.test = make_json_safe(results.Configuration);
    record.test.SNRdB = results.SNRdB;
    record.test.KFactorDb = results.KFactorDb;
    record.test.KFactorLinear = results.KFactorLinear;
    record.testProfile = make_json_safe(results.TestProfile);
    record.simulation = object_properties(results.Parameters);
    record.plot = make_json_safe(results.ChartConfiguration);
    record.plot.outputPng = char(results.ChartPath);
    record.results = struct();
    record.results.methods = cellstr(results.MethodNames);
    record.results.errors = results.Errors;
    record.results.totalBits = results.TotalBits;
    record.results.numFrames = results.NumFrames;
    record.results.syncFailures = results.SyncFailures;
    record.results.BER = results.BER;
    record.results.zeroErrorConvention = char(results.ZeroErrorConvention);

    jsonText = jsonencode(make_json_safe(record));
    payload = matlab.net.base64encode(unicode2native(jsonText, 'UTF-8'));
    outputPath = string(fullfile(outputDir, ...
        sprintf('berlog_%s.txt', results.Timestamp)));

    [fileId, message] = fopen(outputPath, 'w');
    if fileId < 0
        error('Impossibile creare il log %s: %s', outputPath, message);
    end
    cleanup = onCleanup(@() fclose(fileId));
    written = fwrite(fileId, payload, 'char');
    if written ~= numel(payload)
        error('Scrittura incompleta del log Base64: %s', outputPath);
    end
end

function values = object_properties(object)
    names = properties(object);
    values = struct();
    for idx = 1:numel(names)
        values.(names{idx}) = object.(names{idx});
    end
    values = make_json_safe(values);
end

function value = make_json_safe(value)
    if isstruct(value)
        fields = fieldnames(value);
        for elementIdx = 1:numel(value)
            for fieldIdx = 1:numel(fields)
                name = fields{fieldIdx};
                value(elementIdx).(name) = ...
                    make_json_safe(value(elementIdx).(name));
            end
        end
    elseif iscell(value)
        for idx = 1:numel(value)
            value{idx} = make_json_safe(value{idx});
        end
    elseif isstring(value)
        value = cellstr(value);
        if isscalar(value)
            value = value{1};
        end
    elseif isnumeric(value)
        if any(~isfinite(value), 'all')
            value = string(value);
            value = cellstr(value);
            if isscalar(value)
                value = value{1};
            end
        end
    elseif islogical(value) || ischar(value)
        return;
    elseif isa(value, 'function_handle')
        value = func2str(value);
    else
        value = char(string(value));
    end
end
