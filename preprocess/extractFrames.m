function extractFrames
%% Clean
clear
close all

%% Read Videos
videoExt = '.mp4';
inOutFolder = '..\dataset\qualitative_datasets';
list = dir(fullfile(inOutFolder,['*',videoExt]));
videoNames = {list.name};
nVideos = length(videoNames);
nDone = 0;
processBar = waitbar(0,'Reading ... ');
qUpdateWaitbar = parallel.pool.DataQueue;
lUpdateWaitbar = qUpdateWaitbar.afterEach(@(progress) updateWaitbar(progress));

tic
for iVideo = 1:nVideos
    videoName = videoNames{iVideo};
    message = ['Reading video ',videoName,' (',num2str(iVideo),'/',num2str(nVideos),')...'];
    disp(message);
    
    video = VideoReader(videoName);
    nFrames = video.NumberOfFrames;
    video = VideoReader(videoName);
    [~,name,~] = fileparts(videoName);
    outFolder = fullfile(inOutFolder,name,'input');
    if ~exist(outFolder,'dir')
        mkdir(outFolder)
    end
    
    iFrame = 0;
    while hasFrame(video)
        frame = readFrame(video);
        imwrite(frame,fullfile(outFolder,[sprintf('%05d',iFrame),'.jpg']));
        iFrame = iFrame+1;
        qUpdateWaitbar.send(1/nFrames);
    end

end
%% Clean
delete(processBar);
delete(lUpdateWaitbar);

    function updateWaitbar(progress)
        nDone = nDone+progress;
        x = nDone/nVideos;
        waitbar(x,processBar,sprintf('Generating... %.2f%%, %.2f minutes left.',x*100,toc/x*(1-x)/60));
    end
end
