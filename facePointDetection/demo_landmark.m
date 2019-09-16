clear; close all; clc

files = dir('../dataset/');
size0 = size(files); length = size0(1);
count = 0;
for i = 3:length
    disp(['video ', num2str(i), '...'])
    % step 1
    floder_name = files(i).name;
    blur_path = sprintf('%s%s%s', '../dataset/', floder_name, '/blurry/');
	data_path = sprintf('%s%s', '../dataset/', floder_name);
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
    
    % step 3, 4 and 5
    !FacePartDetect.exe data imagelist.txt bbox.txt
	copyfile('bbox.txt',data_path);
    !TestNet.exe bbox.txt image Input result.bin
    show_result();
    % step 6
    mkdir(sprintf('%s%s%s', '../dataset/', floder_name, '/face/'));
    dirtextOutput=dir(fullfile('./show_result/','*.txt'));
    for j=1:img_num  
        txt_name=['./show_result/', dirtextOutput(j).name]; 
        aa = load(txt_name);
        [mm, nn] = size(aa);
        if mm~=5||nn~=2
            error('must be 5 points')
        end
        copyfile(txt_name,sprintf('%s%s%s', '../dataset/', floder_name, '/face/'));
        count = count+1;
    end
    % delete files in /image/, /image/image/, and /show_results/
    delete('bbox.txt');  delete('imagelist.txt');  delete('result.bin') ;
    delete('./image/*.png');    delete('./image/image/*.png');
    delete('./show_result/*.txt');
end



