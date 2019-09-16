function generateList(aimDir)
%% Parameters
frames = dir(fullfile(aimDir,'*.png'));
if isempty(frames)
    return
end

fp = fopen('./imagelist.txt','wt');
%% Align
nFrames = length(frames);
fprintf(fp,'%d\n', nFrames);
frameDirs = @(iFrame) frames(iFrame).name;
for iFrame = 1:nFrames
%     GTpath = sprintf('%s/%s/%04d.png',videoFolder,'sharp',iFrame);
%    coeffGT = sprintf('%s/%s/%06d.txt',videoFolder,'faceGT',iFrame-1);
    fprintf(fp,'image/%s\n', frameDirs(iFrame));
%     disp(sprintf('%s/%04d.png save done!',videoFolder, iFrame));
    
end

fclose(fp);



