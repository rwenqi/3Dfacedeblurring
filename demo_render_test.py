import tensorflow as tf 
import numpy as np
from PIL import Image
import cv2
from scipy.io import loadmat,savemat
# from preprocess_img import Preprocess
from faceReconstruction.load_data import *
from faceReconstruction.preprocess_img import Restore
from faceReconstruction.reconstruct_mesh import Reconstruction_for_render
import os

def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def


def main():
	input_path = './testing_set'
	input_paths = os.listdir(input_path)
	input_pathss = [os.path.join(input_path, (input_p + '/face')) for input_p in input_paths]
	# if not os.path.exists(save_path):
	# 	os.makedirs(save_path)
	# coeff_path = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('coeff.txt')]
	facemodel = BFM()
	lm3D = load_lm3d()

	with tf.Graph().as_default() as graph,tf.device('/cpu:0'):

		tf.load_op_library('./faceReconstruction/mesh_render/rasterize_triangles_kernel.so')
		graph_def = load_graph('./faceReconstruction/mesh_render/render_op.pb')
		face_shape = tf.placeholder(name = 'shape', shape = [1,35709,3], dtype = tf.float32)
		face_norm = tf.placeholder(name = 'normal', shape = [1,35709,3], dtype = tf.float32)
		face_color = tf.placeholder(name = 'color', shape = [1,35709,3], dtype = tf.float32)
		tri = tf.placeholder(name = 'tri', shape = [70789,3], dtype = tf.float32)

		tf.import_graph_def(graph_def,name='render',input_map={'shape:0': face_shape,'normal:0':face_norm,'color:0':face_color,'tri:0':tri})

		img = graph.get_tensor_by_name('render/render_img:0')
		img = tf.squeeze(img,0)

		n = 0
		
		with tf.Session() as sess:
			for filename in input_pathss:
				data_path = filename.replace('testing_set', 'dataset_test')
				print(filename)
				save_path = data_path.replace('face', 'face_render')
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				render_dir = filename.replace('face', 'render')
				print(render_dir)
				if not os.path.exists(render_dir):
          				os.makedirs(render_dir)
				for file in os.listdir(filename):
					if file.endswith('txt'):
						n += 1
						print(n)
						coeff = np.loadtxt(os.path.join(filename,file))
						coeff = np.expand_dims(coeff,0)
						shape,normal,color,_ = Reconstruction_for_render(coeff,facemodel)
						color = color[:,:,::-1]
						render_img = sess.run(img,feed_dict = {face_shape: shape,face_norm:normal,face_color:color,tri:facemodel.face_buf})
						render_img = render_img.clip(0,255)
						render_img[:,:,3]*=255
						cv2.imwrite(os.path.join(save_path,file.replace('txt','png')),render_img)

						img_ori,lm = load_img(os.path.join(data_path.replace('face','blurry'),file.replace('txt','png')),os.path.join(data_path,file))
						render_img = Image.open(os.path.join(save_path,file.replace('txt','png'))).convert('RGB')
						render_img = Restore(img_ori,render_img,lm,lm3D)
						render_img.save(os.path.join(filename.replace('face','render'),file.replace('txt','png')))


if __name__ == '__main__':
	main()
