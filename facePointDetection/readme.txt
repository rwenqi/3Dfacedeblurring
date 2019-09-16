参阅文献CVPR2013 《Deep Convolutional Network Cascade for Facial Point Detection》 主页地址：http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm

第一步：在matlab中运行generateList.m，运行命令[generateList ‘../dataset/文件夹名/blurry’]，例如[generateList '../dataset/vd001/blurry',后续步骤均以vd001为例]
此步骤在本文件生成新的‘imagelist.txt’文件
第二步：将上一步中文件夹（vd001）下的所有图片分别拷贝到imgae和image/image文件夹下，后面新的路径（vd002，……）进行覆盖操作
第三步：在本路径下的Windows命令窗口执行命令[FacePartDetect.exe data imagelist.txt bbox.txt]
此步完成之后会在本路径下生成新的bbox.txt，打开bbox文件，删除其中一个图片中检测出多张人脸中的错误数据，保证每张图片只有一组数据（4个数字），注意观察前后图片数据不要删错数据
第四步；继续在Windows命令窗口执行命令[TestNet.exe bbox.txt image Input result.bin]，生成新的文件result.bin
第五步：在matlab中执行show_result.m，对应的txt文件保存在show_result文件夹下
第六步：将第五步生成的全部txt文件拷贝到[dataset/文件夹名/face]下面