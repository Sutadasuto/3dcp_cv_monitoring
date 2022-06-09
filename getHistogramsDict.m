function histogramsDict = getHistogramsDict()

scriptPath = mfilename('fullpath');
[scriptPath, scriptName, scriptExt] = fileparts(scriptPath);
histogramsFilePath = fullfile(scriptPath, "config_files", "histogram_parameters.txt");

histogramsDict = containers.Map;

histogramsFile = fopen(histogramsFilePath,'r');
while true
    line = fgetl(histogramsFile);
    if line == -1
        break
    end
    if startsWith(line, '#')
        continue
    end
    data_unit = split(line, ' % ');
    dataArray = split(data_unit(1),',');
    histogramsDict(strcat(dataArray(1), "Edges")) = str2double(dataArray(2)):str2double(dataArray(3)):str2double(dataArray(4));
end
end