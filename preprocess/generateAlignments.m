%% Clean
clear
close all;
warning off

%% Parameters
nFramesMax = 1e6;
% {'_nowarp= 0','_OF=1','_homography=2'}
alignmentType = 0;
inputdatasetDir = '../dataset';
outputdatasetDir = '../training_set';
% datasetDir = '..\dataset\quantitative_datasets';

%% Scan videos
videoFolders = dir(inputdatasetDir);
maskFolders = [videoFolders.isdir];
videoFolders = videoFolders(maskFolders);
videoFolders = videoFolders(3:end);
videoFolders = {videoFolders.name};

%% Generation Prepare
nVideos = length(videoFolders);
% nDone = 0;
% processBar = waitbar(0,'Generating... ');
% qUpdateWaitbar = parallel.pool.DataQueue;
% lUpdateWaitbar = qUpdateWaitbar.afterEach(@(progress) updateWaitbar(progress));
% qWriteBreakpoint = parallel.pool.DataQueue;
% fileID = fopen(breakpointFile,'a');
% lBreakpoint = qWriteBreakpoint.afterEach(@(f) fprintf(fileID,'%s\n',f));

if ~exist('../datalist_gopro.txt','file')==0
    delete('../datalist_gopro.txt');
end
%% Start Generating
tic
for iVideo = 1:nVideos
    warning off
    videoFolder = videoFolders{iVideo};
    fprintf('Generating Alignment of video %s (%d/%d)\n',videoFolder,iVideo,nVideos);
    from = fullfile(inputdatasetDir,videoFolder,'blurry');
    to = fullfile(outputdatasetDir,videoFolder);
    alignVideo(alignmentType,from,to,nFramesMax,videoFolder);
    
    disp([videoFolder,' done!']);
end

