%% Clean
clear
close all

%% Parameters
alignments = {'_nowarp'};%,'_OF','_homography'};
nAlignments = length(alignments);
datasetDir = '../data';
datasetFolderPrefix = 'training_augumented_all_nostab';

%% view dataset samples
for iAlignment = 1:nAlignments
    alignFolder = fullfile(datasetDir,[datasetFolderPrefix,alignments{iAlignment}]);
    videoName = dir(alignFolder);
    maskFolders = [videoName.isdir];
    videoName = videoName(maskFolders);
    videoName = videoName(3).name;
    disp(['viewing ' videoName]);
    
    % load sample batch
    batchName = dir(fullfile(alignFolder,videoName,'*.mat'));
    batchName = batchName(1).name;
    disp(['viewing ' batchName]);
    sampleBatchDir = fullfile(alignFolder,videoName,batchName);
    batch = load(sampleBatchDir);
    batchInput = permute(batch.batchInputTorch,[1,3,4,2]);
    batchGT = permute(batch.batchGTTorch,[1,3,4,2]);
    
    % view sample batch
    nNeighbor = size(batchInput,4)/3;
    nBatch = size(batchInput,1);
    for iPatch = 1:nBatch
        patchInput = squeeze(batchInput(iPatch,:,:,:));
        patchGT = squeeze(batchGT(iPatch,:,:,:));
       
        
        for iNeighbor1i = 1:nNeighbor
            subplot(nNeighbor,2,2*(iNeighbor1i-1)+1);
            imshow(patchInput(:,:,3*(iNeighbor1i-1)+1:3*(iNeighbor1i-1)+3));
        end
        subplot(nNeighbor,2,nNeighbor+1);
        imshow(patchGT);
        waitforbuttonpress;
    end
end
