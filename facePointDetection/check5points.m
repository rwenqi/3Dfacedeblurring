clc; clear;

path = dir('./dataset/');
for i = 3:length(path)
    text_path = strcat('./dataset/', path(i).name, '/face/');
    text_files = dir(fullfile(text_path, '*.txt'));
    for j = 1:length(text_files)
        disp(['video ', num2str(i-2, '%03d'), ' of ', num2str(length(path)-2), ' frame ', num2str(j), ' of ', num2str(length(text_files))])
        aa = load(strcat(text_path, text_files(j).name));
        [mm, nn] = size(aa);
        if mm~=5||nn~=2
            error('must be 5 points');
        end
    end
end