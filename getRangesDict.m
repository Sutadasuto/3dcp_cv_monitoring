function rangesDict = getRangesDict()

scriptPath = mfilename('fullpath');
[scriptPath, scriptName, scriptExt] = fileparts(scriptPath);
rangesFilePath = fullfile(scriptPath, "config_files", "ranges.txt");

rangesDict = containers.Map;

rangesFile = fopen(rangesFilePath,'r');
while true
    line = fgetl(rangesFile);
    if line == -1
        break
    end
    if startsWith(line, '#')
        continue
    end
    data_unit = split(line, ' % ');
    dataArray = split(data_unit(1),',');
    rangesDict(strcat(dataArray(1), "Thresholds")) = [str2double(dataArray(2)), str2double(dataArray(3))];
end
end