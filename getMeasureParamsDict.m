function measureParamsDict = getMeasureParamsDict()
scriptPath = mfilename('fullpath');
[scriptPath, scriptName, scriptExt] = fileparts(scriptPath);
imgPropFilePath = fullfile(scriptPath, "config_files", "measurement_parameters.txt");

measureParamsDict = containers.Map;

imgPropFile = fopen(imgPropFilePath,'r');
while true
    line = fgetl(imgPropFile);
    if line == -1
        break
    end
    if startsWith(line, '#')
        continue
    end
    data_info = split(line, ' % ');
    dataArray = split(data_info(1),',');
    valuesArray = 2:length(dataArray);
    currentArray = zeros([1, length(valuesArray)]);
    for idx=valuesArray
        currentArray(idx-1) = str2double(dataArray(idx));
    end
    measureParamsDict(char(dataArray(1))) = currentArray;
end
end