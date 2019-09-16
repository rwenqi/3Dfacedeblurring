import numpy as np

def Split_coeff(coeff):

	id_coeff = coeff[:,:80]
	ex_coeff = coeff[:,80:144]
	tex_coeff = coeff[:,144:224]
	angles = coeff[:,224:227]
	gamma = coeff[:,227:236]
	translation = coeff[:,236:]

	return id_coeff,ex_coeff,tex_coeff,angles,gamma,translation


def Shape_formation(id_coeff,ex_coeff,facemodel):
	face_shape = np.einsum('ij,aj->ai',facemodel.idBase,id_coeff) + \
				np.einsum('ij,aj->ai',facemodel.exBase,ex_coeff) + \
				facemodel.meanshape
	face_shape = np.reshape(face_shape,[1,-1,3])
	face_shape = face_shape - np.mean(np.reshape(facemodel.meanshape,[1,-1,3]), axis = 1, keepdims = True)

	return face_shape

def Compute_norm(face_shape,facemodel):

	face_id = facemodel.face_buf
	point_id = facemodel.point_buf
	shape = face_shape
	face_id = (face_id - 1).astype(np.int32)
	point_id = (point_id - 1).astype(np.int32)
	v1 = shape[:,face_id[:,0],:]
	v2 = shape[:,face_id[:,1],:]
	v3 = shape[:,face_id[:,2],:]
	e1 = v1 - v2
	e2 = v2 - v3
	face_norm = np.cross(e1,e2)
	face_norm = np.concatenate([face_norm,np.zeros([1,1,3])], axis = 1)
	v_norm = np.sum(face_norm[:,point_id,:], axis = 2)
	v_norm = v_norm/np.expand_dims(np.linalg.norm(v_norm,axis = 2),2)

	return v_norm

def Texture_formation(tex_coeff,facemodel):

	face_texture = np.einsum('ij,aj->ai',facemodel.texBase,tex_coeff) + facemodel.meantex
	face_texture = np.reshape(face_texture,[1,-1,3])

	return face_texture

def Compute_rotation_matrix(angles):

	angle_x = angles[:,0][0]
	angle_y = angles[:,1][0]
	angle_z = angles[:,2][0]

	rotation_X = np.array([1.0,0,0,\
		0,np.cos(angle_x),-np.sin(angle_x),\
		0,np.sin(angle_x),np.cos(angle_x)])
	rotation_Y = np.array([np.cos(angle_y),0,np.sin(angle_y),\
		0,1,0,\
		-np.sin(angle_y),0,np.cos(angle_y)])
	rotation_Z = np.array([np.cos(angle_z),-np.sin(angle_z),0,\
		np.sin(angle_z),np.cos(angle_z),0,\
		0,0,1])

	rotation_X = np.reshape(rotation_X,[1,3,3])
	rotation_Y = np.reshape(rotation_Y,[1,3,3])
	rotation_Z = np.reshape(rotation_Z,[1,3,3])

	rotation = np.matmul(np.matmul(rotation_Z,rotation_Y),rotation_X)
	rotation = np.transpose(rotation, axes = [0,2,1])  #transpose row and column (dimension 1 and 2)

	return rotation

def Projection_layer(face_shape,rotation,translation,focal=1015.0,center=112.0):

	t_offset = np.reshape(np.array([0.0,0.0,10.0]),[1,1,3])
	reverse_z = np.reshape(np.array([1.0,0,0,0,1,0,0,0,-1.0]),[1,3,3])
	p_matrix = np.concatenate([[focal],[0.0],[center],[0.0],[focal],[center],[0.0],[0.0],[1.0]],axis = 0)
	p_matrix = np.reshape(p_matrix,[1,3,3])

	face_shape_r = np.matmul(face_shape,rotation)
	face_shape_t = face_shape_r + np.reshape(translation,[1,1,3])
	face_shape_t = np.matmul(face_shape_t,reverse_z) + t_offset
	aug_projection = np.matmul(face_shape_t,np.transpose(p_matrix,[0,2,1]))
	face_projection = aug_projection[:,:,0:2]/np.reshape(aug_projection[:,:,2],[1,np.shape(aug_projection)[1],1])
	z_buffer = np.reshape(aug_projection[:,:,2],[1,-1,1])

	return face_projection,z_buffer

def Illumination_layer(face_texture,norm,gamma):

	num_vertex = np.shape(face_texture)[1]

	init_lit = np.array([0.8,0,0,0,0,0,0,0,0])
	gamma = gamma + np.reshape(init_lit,[1,9])

	# parameter of 9 SH function
	a0 = np.pi 
	a1 = 2*np.pi/np.sqrt(3.0)
	a2 = 2*np.pi/np.sqrt(8.0)
	c0 = 1/np.sqrt(4*np.pi)
	c1 = np.sqrt(3.0)/np.sqrt(4*np.pi)
	c2 = 3*np.sqrt(5.0)/np.sqrt(12*np.pi)

	Y0 = np.tile(np.reshape(a0*c0,[1,1,1]),[1,num_vertex,1]) 
	Y1 = np.reshape(-a1*c1*norm[:,:,1],[1,num_vertex,1]) 
	Y2 = np.reshape(a1*c1*norm[:,:,2],[1,num_vertex,1])
	Y3 = np.reshape(-a1*c1*norm[:,:,0],[1,num_vertex,1])
	Y4 = np.reshape(a2*c2*norm[:,:,0]*norm[:,:,1],[1,num_vertex,1])
	Y5 = np.reshape(-a2*c2*norm[:,:,1]*norm[:,:,2],[1,num_vertex,1])
	Y6 = np.reshape(a2*c2*0.5/np.sqrt(3.0)*(3*np.square(norm[:,:,2])-1),[1,num_vertex,1])
	Y7 = np.reshape(-a2*c2*norm[:,:,0]*norm[:,:,2],[1,num_vertex,1])
	Y8 = np.reshape(a2*c2*0.5*(np.square(norm[:,:,0])-np.square(norm[:,:,1])),[1,num_vertex,1])

	Y = np.concatenate([Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8],axis=2)

	# Y shape:[batch,N,9].

	lit_r = np.squeeze(np.matmul(Y,np.expand_dims(gamma,2)),2) #[batch,N,9] * [batch,9,1] = [batch,N]
	lit_g = np.squeeze(np.matmul(Y,np.expand_dims(gamma,2)),2)
	lit_b = np.squeeze(np.matmul(Y,np.expand_dims(gamma,2)),2)

	# shape:[batch,N,3]
	face_color = np.stack([lit_r*face_texture[:,:,0],lit_g*face_texture[:,:,1],lit_b*face_texture[:,:,2]],axis = 2)
	lighting = np.stack([lit_r,lit_g,lit_b],axis = 2)*128

	return face_color,lighting

def Reconstruction(coeff,facemodel):
	id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = Split_coeff(coeff)
	face_shape = Shape_formation(id_coeff, ex_coeff, facemodel)
	face_texture = Texture_formation(tex_coeff, facemodel)
	face_norm = Compute_norm(face_shape,facemodel)
	rotation = Compute_rotation_matrix(angles)
	face_norm_r = np.matmul(face_norm,rotation)
	face_projection,z_buffer = Projection_layer(face_shape,rotation,translation)
	face_projection = np.stack([face_projection[:,:,0],224 - face_projection[:,:,1]], axis = 2)
	landmarks_2d = face_projection[:,facemodel.keypoints,:]
	face_color,lighting = Illumination_layer(face_texture, face_norm_r, gamma)
	tri = facemodel.face_buf

	return face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d

def Reconstruction_for_render(coeff,facemodel):
	id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = Split_coeff(coeff)
	face_shape = Shape_formation(id_coeff, ex_coeff, facemodel)
	face_texture = Texture_formation(tex_coeff, facemodel)
	face_norm = Compute_norm(face_shape,facemodel)
	rotation = Compute_rotation_matrix(angles)
	face_shape_r = np.matmul(face_shape,rotation)
	face_shape_r = face_shape_r + np.reshape(translation,[1,1,3])
	face_norm_r = np.matmul(face_norm,rotation)
	face_color,lighting = Illumination_layer(face_texture, face_norm_r, gamma)
	tri = facemodel.face_buf

	return face_shape_r,face_norm_r,face_color,tri