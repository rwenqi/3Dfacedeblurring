clear; close all; clc

files = dir('D:\MATLAB\facedata\crop_face/');
size0 = size(files); length = size0(1);
for i =  [103:length] % vd002,
    disp(['video...', num2str(i-2)])
    % step 1
    floder_name = files(i).name;
    blur_path = sprintf('%s%s%s', '../../facedata/crop_face/', floder_name, '/blurry/');
    generateList(blur_path);
    
    % step 2
    target_path1 = './image/';
    target_path2 = './image/image/';
    dirOutput=dir(fullfile(blur_path,'*.png'));
    size_img = size(dirOutput);
    img_num = size_img(1);
    for j=1:img_num  
        image_name=[blur_path, dirOutput(j).name];  
        copyfile(image_name,target_path1);
        copyfile(image_name,target_path2);
    end
    
    % step 3, 
    !FacePartDetect.exe data imagelist.txt bbox.txt
    
    pause(1)
    data = importdata('bbox.txt');
    
    %%
    if size(data.data,1)~=img_num
        fid=fopen(['./','invalidvideo.txt'],'a+');%写入文件路径
        fprintf(fid,['video...', num2str(i-2),'\r\n']);   %按列输出，若要按行输出：fprintf(fid,'%.4\t',A(jj)); 
        fclose(fid);
        continue
%         error('must be 4 points, cannot be larger than 4')
    else
        points4 = data.data;
        image_names = cell2mat(data.textdata);

        if size(points4)>4
            error('must be 4 points, cannot be larger than 4')
        end
        [m,n] = find(isnan(points4));
        NAN_col = unique(m);
        num_NAN = size(NAN_col, 1);
        if isempty(m)||img_num>m(num_NAN)
            batch = num_NAN + 1;
        elseif img_num>m(num_NAN)
            batch = num_NAN;
        else
            error('line 40')
        end
        batch_ = cell(batch, 1);
        if batch == 1
            batch_{1} = (1:img_num);
        elseif batch>1
            batch_{1} = (1:m(1)-1);
            for j = 2:batch-1
                batch_{j} = (m(j-1)+1: m(j)-1);
            end
            batch_{batch} = (m(batch-1)+1: img_num);
        end

        if batch == 1
            selected = batch_{1};
            path_blur_selected = strcat('./batch/', floder_name, '/blurry/');
            path_sharp_selected = strcat('./batch/', floder_name, '/sharp/');
            mkdir(path_blur_selected); mkdir(path_sharp_selected);
            for k = 1:size(selected,2)
                 selected_blurname = image_names(selected(k),:);
                 selected_blurimg = imread(selected_blurname);
                 imwrite(selected_blurimg, strcat(path_blur_selected, selected_blurname(7:end)));
                 %
                 selected_sharpname = strcat('../../facedata/crop_face/', floder_name, '/sharp/',selected_blurname(7:end));
                 selected_sharpimg = imread(selected_sharpname);
                 imwrite(selected_sharpimg, strcat(path_sharp_selected, selected_blurname(7:end)));
            end
        elseif batch>1
            cont = 1;
            for j = 1:batch
                if size(batch_{j},2)>25
                    selected = batch_{j};
                    path_blur_selected = strcat('./batch/', floder_name, '_', num2str(cont, '%02d'), '/blurry/');
                    path_sharp_selected = strcat('./batch/', floder_name, '_', num2str(cont, '%02d'), '/sharp/');
                    mkdir(path_blur_selected);  mkdir(path_sharp_selected);
                    for k = 1:size(selected,2)
                        selected_blurname = image_names(selected(k),:);
                        selected_blurimg = imread(selected_blurname);
                        imwrite(selected_blurimg, strcat(path_blur_selected, selected_blurname(7:end)));
                        %
                        selected_sharpname = strcat('../../facedata/crop_face/', floder_name, '/sharp/',selected_blurname(7:end));
                        selected_sharpimg = imread(selected_sharpname);
                        imwrite(selected_sharpimg, strcat(path_sharp_selected, selected_blurname(7:end)));
                    end
                    cont = cont+1;
                end
            end
        end


        delete('bbox.txt');  delete('imagelist.txt');  
        delete('./image/*.png');    delete('./image/image/*.png');
        pause(1)
    end
end





