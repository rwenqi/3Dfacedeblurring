import numpy as np 
from PIL import Image
from scipy.io import loadmat

class BFM():
	def __init__(self):
		model_path = './faceReconstruction/BFM/BFM_model_front.mat'
		model = loadmat(model_path)
		self.meanshape = model['meanshape']
		self.idBase = model['idBase']
		self.exBase = model['exBase']
		self.meantex = model['meantex']
		self.texBase = model['texBase']
		self.point_buf = model['point_buf']
		self.face_buf = model['tri']
		self.keypoints = np.squeeze(model['keypoints']).astype(np.int32) - 1

def load_lm3d():

	Lm3D = loadmat('./faceReconstruction/BFM/similarity_Lm3D_all.mat')
	Lm3D = Lm3D['lm']
	lm_idx = np.array([31,37,40,43,46,49,55]) - 1
	Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
	Lm3D = Lm3D[[1,2,0,3,4],:]

	return Lm3D

def load_img(img_path,lm_path):

	image = Image.open(img_path)
	lm = np.loadtxt(lm_path)

	return image,lm

def save_obj(path,v,f,c):
	with open(path,'w') as file:
		for i in range(len(v)):
			file.write('v %f %f %f %f %f %f\n'%(v[i,0],v[i,1],v[i,2],c[i,0],c[i,1],c[i,2]))

		file.write('\n')

		for i in range(len(f)):
			file.write('f %d %d %d\n'%(f[i,0],f[i,1],f[i,2]))

	file.close()