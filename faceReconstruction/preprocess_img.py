import numpy as np
from scipy.io import loadmat,savemat
from PIL import Image

def POS(xp,x):
	npts = xp.shape[1]

	A = np.zeros([2*npts,8])

	A[0:2*npts-1:2,0:3] = x.transpose()
	A[0:2*npts-1:2,3] = 1

	A[1:2*npts:2,4:7] = x.transpose()
	A[1:2*npts:2,7] = 1;
	# print('A',A)

	b = np.reshape(xp.transpose(),[2*npts,1])
	# print('b',b)

	k,_,_,_ = np.linalg.lstsq(A,b)
	# print(k)
	# print('k',k)
	# k = k[1]

	R1 = k[0:3]
	R2 = k[4:7]
	sTx = k[3]
	sTy = k[7]
	s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
	t = np.stack([sTx,sTy],axis = 0)

	return t,s

def process_img(img,lm,t,s):
	w0,h0 = img.size
	img = img.transform(img.size, Image.AFFINE, (1, 0, t[0] - w0/2, 0, 1, h0/2 - t[1]))
	w = (w0/s*102).astype(np.int32)
	h = (h0/s*102).astype(np.int32)
	img = img.resize((w,h),resample = Image.BILINEAR)
	lm = np.stack([lm[:,0] - t[0] + w0/2,lm[:,1] - t[1] + h0/2],axis = 1)/s*102

	left = (w/2 - 112).astype(np.int32)
	right = left + 224
	up = (h/2 - 112).astype(np.int32)
	below = up + 224

	img = img.crop((left,up,right,below))
	img = np.array(img)
	img = img[:,:,::-1]
	img = np.expand_dims(img,0)

	return img

def restore_img(img,render_img,lm,t,s):
	w0,h0 = img.size
	w = (w0/s*102).astype(np.int32)
	h = (h0/s*102).astype(np.int32)

	render_img = np.array(render_img)
	if w > 224:
	    w_w = int((w-224)/2)
	    w_r = w-224-w_w
	else:
	    w_w = 0
	    w_r = 0
	if h > 224:
	    h_h = int((h-224)/2)
	    h_r = h-224-h_h
	else:
	    h_h = 0
	    h_r = 0
	img_pad = np.pad(render_img, ((h_h,h_r), (w_w,w_r), (0,0)), 'constant')

	img_pad = Image.fromarray(img_pad)
	img = img_pad.resize((w0,h0),resample = Image.BILINEAR)

	# w_reg = int(round(w0*1.5 - t[0]))
	# h_reg = int(round(h0/2 + t[1]))
	# img = img.crop((0,0,w_reg,h_reg))
	# img = np.array(img)
	# img = np.pad(img,((abs(h0-h_reg),0),(abs(w0-w_reg),0),(0,0)),'constant')
	# img = Image.fromarray(img)
	img = img.transform(img.size, Image.AFFINE, (1, 0, w0/2 - t[0], 0, 1, t[1] - h0/2))

	return img

def Restore(img,render_img,lm,lm3D):
	w0,h0 = img.size
	lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)
	t,s = POS(lm.transpose(),lm3D.transpose())
	img_new = restore_img(img,render_img,lm,t,s)

	return img_new

def Preprocess(img,lm,lm3D):

	w0,h0 = img.size
	lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)
	t,s = POS(lm.transpose(),lm3D.transpose())
	img_new = process_img(img,lm,t,s)
	affine_params = np.array([w0,h0,102.0/s,t[0],t[1]])

	return img_new,affine_params

