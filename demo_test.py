import tensorflow as tf 
import numpy as np
import argparse
from PIL import Image
import cv2
from scipy.io import loadmat,savemat
from faceReconstruction.preprocess_img import Preprocess
from faceReconstruction.load_data import *
from faceReconstruction.reconstruct_mesh import Reconstruction
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='in', help='determine whether gt or input')
    args = parser.parse_args()
    return args

def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def

def main():
	args = parse_args()
	# input and output folder
	input_path = './dataset_test'
	
	if args.type == 'in':
		type_path = '/face'
	elif args.type == 'gt':
		type_path = '/faceGT'
	else:
		print('type should be set to either gt or in')

	# save_path
	input_paths = os.listdir(input_path)
	input_pathss = [os.path.join(input_path, (input_p + type_path)) for input_p in input_paths]
	# output_pathss = input_pathss.replace('dataset', 'data')
	# print(output_pathss)

	# if not os.path.exists(save_path):
	# 	os.makedirs(save_path)	

	# read BFM face model
	facemodel = BFM()
	lm3D = load_lm3d()
	n = 0

	# build reconstruction model
	with tf.Graph().as_default() as graph,tf.device('/cpu:0'):

		graph_def = load_graph('faceReconstruction/network/model_mask3_pure.pb')
		images = tf.placeholder(name = 'input_imgs', shape = [None,224,224,3], dtype = tf.float32)

		tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})

		# output coefficients of R-Net (dim = 239) 
		coeff = graph.get_tensor_by_name('resnet/coeff:0')

		with tf.Session() as sess:
			for filename in input_pathss:
				save_path = filename.replace('dataset_test', 'testing_set')
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				# img_filename = filename.replace('face','blurry')
				for file in os.listdir(filename):
					if file.endswith('txt'):
						n += 1
						print(n)

						# load images and corresponding 5 facial landmarks
						img,lm = load_img(os.path.join(filename.replace('face','blurry'),file.replace('txt','png')),os.path.join(filename,file))
						# preprocess input image
						input_img,affine_params = Preprocess(img,lm,lm3D)

						coef = sess.run(coeff,feed_dict = {images: input_img})

						# reconstruct 3D face with output coefficients and BFM face model
						# face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d = Reconstruction(coef,facemodel)

						# input_img = np.squeeze(input_img)
						# shape = np.squeeze(face_shape,0)
						# texture = np.squeeze(face_texture,0)
						# landmarks_2d = np.squeeze(landmarks_2d,0)

						# save output files
						# cv2.imwrite(os.path.join(save_path,file.replace('.jpg','_crop.jpg')),input_img)
						np.savetxt(os.path.join(save_path,file), coef)
						# np.savetxt(os.path.join(save_path,file.replace('.jpg','_lm.txt')),landmarks_2d)
						# np.savetxt(os.path.join(save_path,file.replace('.jpg','_affine.txt')),affine_params)
						# save_obj(os.path.join(save_path,file.replace('.jpg','_mesh.obj')),shape,tri,np.clip(texture,0,255)/255)

if __name__ == '__main__':
	main()
