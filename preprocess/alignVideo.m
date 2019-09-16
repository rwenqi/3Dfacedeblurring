function alignVideo(alignmentType,from,to,nFramesMax,videoFolder)
% if nargin == 4
%     qUpdateWaitbar = [];
% end

% alignmentType: 0 for nowarp, 1 for optical flow, 2 for homography, 3 for similarity
addpath(genpath('thirdparty'));

%% Parameters
frames = dir(fullfile(from,'*.png'));
if isempty(frames)
    return
end
for l = -2:2
    checkDir(fullfile(to,['blur_',num2str(l)]));
end

fp = fopen('../datalist_gopro.txt','a');
%% Align
nFrames = min(length(frames),nFramesMax);
frameDirs = @(iFrame) fullfile(from,frames(iFrame).name);
for iFrame = 1:min(nFrames,nFramesMax)
    % save image_1 to image_5
    v0 = im2double(imread(frameDirs(iFrame)));
    v0g = single(rgb2gray(v0));
    [h,w,~] = size(v0);
    
    for l = -2:2
        if l ~= 0
            vi = im2double(imread(frameDirs(max(min(iFrame+l,length(frames)),1))));
            vig = single(rgb2gray(vi));
            if alignmentType == 0
                v_i0 = vi;
            elseif alignmentType == 1
                flo_i0 = genFlow(v0g, vig);
                [v_i0, ~] = warpToRef(v0, vi, flo_i0);
            elseif alignmentType == 2
                v_i0 = homographyAlignment(v0,vi,0);
            elseif alignmentType == 3
                v_i0 = similarityAlignment(v0,vi,0);
            end
        else
            v_i0 = v0;
        end
        imwrite(v_i0, fullfile(to,['blur_',num2str(l)],frames(iFrame).name));
    end
%     fprintf(fp,'%s %s\n',GTpath, BUpath);
    GTpath = sprintf('%s/%s/%s',videoFolder,'sharp',frames(iFrame).name);
    llBlur = sprintf('%s/%s/%s',videoFolder,'blur_-2',frames(iFrame).name);
    lBlur = sprintf('%s/%s/%s',videoFolder,'blur_-1',frames(iFrame).name);
    mBlur = sprintf('%s/%s/%s',videoFolder,'blur_0',frames(iFrame).name);
    rBlur = sprintf('%s/%s/%s',videoFolder,'blur_1',frames(iFrame).name);
    rrBlur = sprintf('%s/%s/%s',videoFolder,'blur_2',frames(iFrame).name);
    imageTxtName = strrep(frames(iFrame).name, 'png', 'txt');
    coeff = sprintf('%s/%s/%s',videoFolder,'face',imageTxtName);
    render = sprintf('%s/%s/%s',videoFolder,'render',frames(iFrame).name);
%    coeffGT = sprintf('%s/%s/%06d.txt',videoFolder,'faceGT',iFrame-1);
    fprintf(fp,'%s %s %s %s %s %s %s %s\n',GTpath, llBlur, lBlur, mBlur, rBlur, rrBlur, coeff, render);
    disp(sprintf('%s/%s save done!',videoFolder, frames(iFrame).name));
%     if isa(qUpdateWaitbar,'parallel.pool.DataQueue')
%         qUpdateWaitbar.send(1/nFrames);
%     end
    
end

fclose(fp);



