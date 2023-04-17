def compute_face_phi_D_or_N(phi2d,fx,fy,ni,nj,phi_bc_south_type,phi_bc_north_type):
   import numpy as np

   phi2d_face_w=np.empty((ni+1,nj))
   phi2d_face_s=np.empty((ni,nj+1))
   phi2d_face_w[0:-1,:]=fx*phi2d+(1-fx)*np.roll(phi2d,1,axis=0)
   phi2d_face_s[:,0:-1]=fy*phi2d+(1-fy)*np.roll(phi2d,1,axis=1)

# west boundary 
# cyclic_x:
   phi2d_face_w[0,:]=0.5*(phi2d[0,:]+phi2d[-1,:])

# east boundary 
# cyclic_x:
   phi2d_face_w[-1,:]=0.5*(phi2d[0,:]+phi2d[-1,:])


# south boundary 
   phi2d_face_s[:,0]=0
   if phi_bc_south_type == 'n': 
# neumann
      phi2d_face_s[:,0]=phi2d[:,0]

# north boundary 
   phi2d_face_s[:,-1]=0
   if phi_bc_north_type == 'n': 
# neumann
      phi2d_face_s[:,-1]=phi2d[:,-1]
   
   return phi2d_face_w,phi2d_face_s

