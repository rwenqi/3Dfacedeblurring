function show_result

fid = fopen('result.bin', 'rb');
imageNum = fread(fid, 1, 'int32');
pointNum = fread(fid, 1, 'int32');
assert(pointNum == 5);
valid = fread(fid, imageNum, 'int8');
assert(all(valid) == 1);
point = reshape(fread(fid, 2 * pointNum * imageNum, 'float64'), [2 * pointNum, imageNum])';
fclose(fid);

green(1, 1, :) = [0 255 0];
mkdir('show_result');
fid = fopen('bbox.txt', 'r');
for n1 = 1 : imageNum
    imageName = fscanf(fid, '%s', 1);
    imageTxtName = strrep(imageName, '.png', '.txt');
    imageTxtName = strrep(imageTxtName, 'image/', '');
    fscanf(fid, '%d', 4);
%     I = imread(['image/' imageName]);
%     if size(I, 3) == 1
%         I = repmat(I, [1 1 3]);
%     end
    pt = point(n1, :);
    pt = round(pt) + 1;
    fileName = sprintf('show_result/%s', imageTxtName);
    disp(fileName);
    fidTxt = fopen(fileName, 'a');
    for n2 = 1 : 5 % pointNum
        p = pt(n2 * 2 - 1 : n2 * 2);
%         I(p(2) - 1 : p(2) + 1, p(1) - 2 : p(1) + 2, :) = repmat(green, [3 5]);
%         I([p(2) - 2 p(2) + 2], p(1) - 1 : p(1) + 1, :) = repmat(green, [2 3]);
        fprintf(fidTxt, '%d %d\n',p(1),p(2));
    end
    fclose(fidTxt);
%     imwrite(I, ['show_result/' imageName]);
end
fclose(fid);
