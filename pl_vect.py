import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
#import gradients.py
from gradients import compute_face_phi,dphidx,dphidy,init
from compute_face_phi_D_or_N import compute_face_phi_D_or_N
import matplotlib.ticker as mtick
from matplotlib import ticker

plt.rcParams.update({'font.size': 22})


plt.interactive(True)

plt.close('all')

# read data file
tec=np.genfromtxt("tec.dat", dtype=None,comments="%")

#text='VARIABLES = X Y P U V u2 v2 w2 uv mu_sgs prod'

x=tec[:,0]
y=tec[:,1]
p=tec[:,2]
u=tec[:,3]
v=tec[:,4]
uu=tec[:,5]
vv=tec[:,6]
ww=tec[:,7]
uv=tec[:,8]
eps=tec[:,9]
k=0.5*(uu+vv+ww)

if max(y) == 1.:
   ni=170
   nj=194
   nu=1./10000.
else:
   nu=1./10595.
   if max(x) > 8.:
     nj=162
     ni=162
   else:
     ni=402
     nj=162

viscos=nu

u2d=np.reshape(u,(nj,ni))   #=mean{v_1}
v2d=np.reshape(v,(nj,ni))   #=mean{v_2}
p2d=np.reshape(p,(nj,ni))   #=mean{p}
x2d=np.reshape(x,(nj,ni))   #=x_1
y2d=np.reshape(y,(nj,ni))   #=x_2
uu2d=np.reshape(uu,(nj,ni)) #=mean{v'_1v'_1}
uv2d=np.reshape(uv,(nj,ni)) #=mean{v'_1v'_2}
vv2d=np.reshape(vv,(nj,ni)) #=mean{v'_2v'_2}
ww2d=np.reshape(ww,(nj,ni)) #=mean{v'_3v'_3}
k2d=np.reshape(k,(nj,ni))
eps2d=np.reshape(eps,(nj,ni))

u2d=np.transpose(u2d)
v2d=np.transpose(v2d)
p2d=np.transpose(p2d)
uu2d=np.transpose(uu2d)
vv2d=np.transpose(vv2d)
ww2d=np.transpose(ww2d)
uv2d=np.transpose(uv2d)
k2d=np.transpose(k2d)
eps2d=np.transpose(eps2d)


# set periodic b.c on west boundary
#u2d[0,:]=u2d[-1,:]
#v2d[0,:]=v2d[-1,:]
#p2d[0,:]=p2d[-1,:]
#uu2d[0,:]=uu2d[-1,:]


# read k and eps from a 2D RANS simulations. They should be used for computing the damping function f
k_eps_RANS = np.loadtxt("k_eps_RANS.dat")
k_RANS=k_eps_RANS[:,0]
eps_RANS=k_eps_RANS[:,1]
vist_RANS=k_eps_RANS[:,2]

ntstep=k_RANS[0]

k_RANS2d=np.reshape(k_RANS,(ni,nj))/ntstep
eps_RANS2d=np.reshape(eps_RANS,(ni,nj))/ntstep
vist_RANS2d=np.reshape(vist_RANS,(ni,nj))/ntstep

# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("mesh.dat")
xf=xc_yc[:,0]
yf=xc_yc[:,1]
x2d=np.reshape(xf,(nj-1,ni-1))
y2d=np.reshape(yf,(nj-1,ni-1))
x2d=np.transpose(x2d)
y2d=np.transpose(y2d)

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])

# compute geometric quantities
areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy = init(x2d,y2d,xp2d,yp2d)

# delete last row
u2d = np.delete(u2d, -1, 0)
v2d = np.delete(v2d, -1, 0)
p2d = np.delete(p2d, -1, 0)
k2d = np.delete(k2d, -1, 0)
uu2d = np.delete(uu2d, -1, 0)
vv2d = np.delete(vv2d, -1, 0)
ww2d = np.delete(ww2d, -1, 0)
uv2d = np.delete(uv2d, -1, 0)
eps2d = np.delete(eps2d, -1, 0)
k_RANS2d = np.delete(k_RANS2d, -1, 0)
eps_RANS2d = np.delete(eps_RANS2d, -1, 0)
vist_RANS2d = np.delete(vist_RANS2d, -1, 0)

# delete first row
u2d = np.delete(u2d, 0, 0)
v2d = np.delete(v2d, 0, 0)
p2d = np.delete(p2d, 0, 0)
k2d = np.delete(k2d, 0, 0)
uu2d = np.delete(uu2d, 0, 0)
vv2d = np.delete(vv2d, 0, 0)
ww2d = np.delete(ww2d, 0, 0)
uv2d = np.delete(uv2d, 0, 0)
eps2d = np.delete(eps2d, 0, 0)
k_RANS2d = np.delete(k_RANS2d, 0, 0)
eps_RANS2d = np.delete(eps_RANS2d, 0, 0)
vist_RANS2d = np.delete(vist_RANS2d, 0, 0)

# delete last columns
u2d = np.delete(u2d, -1, 1)
v2d = np.delete(v2d, -1, 1)
p2d = np.delete(p2d, -1, 1)
k2d = np.delete(k2d, -1, 1)
uu2d = np.delete(uu2d, -1, 1)
vv2d = np.delete(vv2d, -1, 1)
ww2d = np.delete(ww2d, -1, 1)
uv2d = np.delete(uv2d, -1, 1)
eps2d = np.delete(eps2d, -1, 1)
k_RANS2d = np.delete(k_RANS2d, -1, 1)
eps_RANS2d = np.delete(eps_RANS2d, -1, 1)
vist_RANS2d = np.delete(vist_RANS2d, -1, 1)

# delete first columns
u2d = np.delete(u2d, 0, 1)
v2d = np.delete(v2d, 0, 1)
p2d = np.delete(p2d, 0, 1)
k2d = np.delete(k2d, 0, 1)
uu2d = np.delete(uu2d, 0, 1)
vv2d = np.delete(vv2d, 0, 1)
ww2d = np.delete(ww2d, 0, 1)
uv2d = np.delete(uv2d, 0, 1)
eps2d = np.delete(eps2d, 0, 1)
k_RANS2d = np.delete(k_RANS2d, 0, 1)
eps_RANS2d = np.delete(eps_RANS2d, 0, 1)
vist_RANS2d = np.delete(vist_RANS2d, 0, 1)

ni = ni-2
nj = nj-2

# eps at last cell upper cell wrong. fix it.
eps2d[:,-1]=eps2d[:,-2]

# compute face value of U and V
u2d_face_w,u2d_face_s=compute_face_phi_D_or_N(u2d,fx,fy,ni,nj,'d','d') # the two last argument: Dirichlet = 0  b.c. at south and north 
v2d_face_w,v2d_face_s=compute_face_phi_D_or_N(v2d,fx,fy,ni,nj,'d','d')
p2d_face_w,p2d_face_s=compute_face_phi_D_or_N(p2d,fx,fy,ni,nj,'n','n') # the two last argument: Neumann b.c. at south and north 

# x derivatives
dudx=dphidx(u2d_face_w,u2d_face_s,areawx,areasx,vol)
dvdx=dphidx(v2d_face_w,v2d_face_s,areawx,areasx,vol)
dpdx=dphidx(p2d_face_w,p2d_face_s,areawx,areasx,vol)

# y derivatives
dudy=dphidy(u2d_face_w,u2d_face_s,areawy,areasy,vol)
dvdy=dphidy(v2d_face_w,v2d_face_s,areawy,areasy,vol)
dpdy=dphidy(p2d_face_w,p2d_face_s,areawy,areasy,vol)

omega2d=eps2d/k2d/0.09

########  Assignment 1.1 - Reynolds Stresses ##########
rho = 1
tau = rho * uv2d

tau2d_face_w, tau2d_face_s = compute_face_phi(tau, fx, fy,ni, nj)
dtaudx = dphidx(tau2d_face_w, tau2d_face_s, areawx, areasx, vol)
dtaudy = dphidy(tau2d_face_w, tau2d_face_s, areawy, areasy, vol)


#####   Assignment 1.2 ######  #TODO y equation


                                          
v1v1 = u2d ** 2                                                          #LHS 1st term
v1v1_face_w, v1v1_face_s = compute_face_phi(v1v1, fx, fy,ni, nj)
duudx = dphidx(v1v1_face_w, v1v1_face_s, areawx, areasx, vol)
 
# duudx = 2 * u2d * dudx                                                #LHS 1st term - chain rule

                                         
v1v2 = u2d * v2d                       #LHS 2nd term without chain rule
v1v2_face_w, v1v2_face_s = compute_face_phi(v1v2, fx, fy,ni, nj)
duvdy = dphidy(v1v1_face_w, v1v1_face_s, areawy, areasy, vol)

# duvdy_chain = (u2d * dvdy) + (v2d * dudy)                                #LHS 2nd term with chain rule


dudx_face_w, dudx_face_s = compute_face_phi(dudx, fx, fy,ni, nj)
dudx_2 = nu * dphidx(dudx_face_w, dudx_face_s, areawx, areasx, vol)    #RHS 2nd term  (R1 eq)

dudy_face_w, dudy_face_s = compute_face_phi(dudy, fx, fy,ni, nj)
dudy_2 = nu * dphidy(dudy_face_w, dudy_face_s, areawy, areasy, vol)    #RHS 3rd term 

u_2 = u2d**2    #(v1')^2 ~ (u')^2
u_2face_w, u_2face_s = compute_face_phi(u_2, fx, fy,ni, nj)      #computes face value of (v1')^2
du_2dx = dphidx(u_2face_w, u_2face_s, areawx, areasx, vol)       #RHS 4th term        


# ###

dv1v2dx = dphidx(v1v1_face_w, v1v1_face_s, areawx, areasx, vol)                                 
v2v2 = v2d * v2d                                                     #LHS 1st term
v2v2_face_w, v2v2_face_s = compute_face_phi(v2v2, fx, fy,ni, nj)
dvvdy = dphidy(v2v2_face_w, v2v2_face_s, areawy, areasy, vol)

dv2dx1 = dphidx(v2d_face_w, v2d_face_s, areawy, areasy, vol)
dv2dx1_face_w, dv2dx1_face_s = compute_face_phi(dv2dx1, fx, fy,ni, nj)
dvdx_2 = nu * dphidx(dv2dx1_face_w, dv2dx1_face_s, areawx, areasx, vol)

dv2dx2 = dphidy(v2d_face_w, v2d_face_s, areawy, areasy, vol)
dv2dx2_face_w, dv2dx2_face_s = compute_face_phi(dv2dx2, fx, fy,ni, nj)
dvdy_2 = nu * dphidy(dv2dx2_face_w, dv2dx2_face_s, areawy, areasy, vol)


v2_2 = v2d**2    #(v1')^2 ~ (u')^2
v2_2face_w, v2_2face_s = compute_face_phi(v2_2, fx, fy,ni, nj)      #computes face value of (v1')^2
dv_2dy = dphidy(v2_2face_w, v2_2face_s, areawy, areasy, vol)      #RHS 4th term        



####### Assignment 1.3 - Production Term   #########


# v1_2[i,j] = u2d[i,j]**2 
# v2_2[i,j] = v2d[i,j]**2
# u1_2 = np.zeros([ni,nj])
# u1_2[i,j] = u2d[i,j]*v2d[i,j] 
# u2_1 = np.zeros([ni,nj])
# u1_2[i,j] = v2d[i,j]*u2d[i,j] 
# p_k = np.zeros([ni,nj])
# p_k[i,j] = (-v1_2[i,j]*dv1dx1[i,j]) +  (-v2_2[i,j]*dv2dx2[i,j]) + (-u1_2[i,j]*dv1dx2[i,j]) + (-u2_1[i,j]*dv2dx1[i,j])

v1_2 = u2d**2
v2_2 = v2d**2
# u1_2 = np.zeros([ni,nj])
u1_2 = u2d*v2d
# u2_1 = np.zeros([ni,nj])
u2_1 = v2d*u2d
p_k = np.zeros([ni,nj])
p_k = (-v1_2*dudx) +  (-v2_2*dvdy) + (-u1_2*dudy) + (-u2_1*dvdx)
pk1 = np.zeros([ni,nj])
pk2 = np.zeros([ni,nj])
pk3 = np.zeros([ni,nj])
pk4 = np.zeros([ni,nj])
pk1= -v1_2*dudx
pk2= -v2_2*dv2dx2
pk3= -u1_2*dudy
pk4= -u2_1*dvdx




################################ vector plot

plt.figure()   #Fig 1
# plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
k=2# plot every forth vector
ss=1.2 #vector length
plt.quiver(xp2d[::k,::k],yp2d[::k,::k],u2d[::k,::k],v2d[::k,::k],width=0.001)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("vector plot")
#plt.savefig('vect_python.png')

################################ contour plot
plt.figure()   #Fig 2 
# plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
plt.pcolormesh(xp2d,yp2d,dudy, vmin=-5,vmax=5,cmap=plt.get_cmap('hot'),shading='gouraud')
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r"the gradient $\partial \bar{v}_1/\partial x_2$")
#plt.savefig('dudy.png')


#************
# plot uv
plt.figure()   #Fig 3
# plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
i=1
plt.plot(uv2d[i,:],yp2d[i,:],'b-')
i = 35
plt.plot(uv2d[i,:], yp2d[i,:],'r-')
plt.xlabel('$\overline{u^\prime v^\prime}$')
plt.ylabel('y/H')

#plt.savefig('uv_python.png')


###### plot - reynolds stress


plt.figure(figsize=(10,6))    #Fig 4
i=1
plt.plot(dtaudx[i,:],yp2d[i,:],'b-')
i = 35
plt.plot(dtaudx[i,:], yp2d[i,:],'r-')
# plt.ylim(0, 0.015)
# plt.xlim(-0.12, 0)
plt.xlabel(r'Reynolds Stress  $ (\tau_{ij}) $')
plt.ylabel('Velocity')
plt.legend(('$ i = 1 $', '$ i = 10 $'))

### plot - 1.2                                          #TODO - Add x and y labels, legend, plot title

plt.figure()    #Fig 5
i = 35
plt.plot(-dpdx[i,:], yp2d[i,:], 'b-.')
# plt.plot(dv1v1dx[i,:], yp2d[i,:], 'r-.')        
# plt.plot(dv1v2dy[i,:], yp2d[i,:], 'g-.')      
plt.legend(('$\partial p/\partial x$', '$\partial \bar{v}^\prime_1 \bar{v}^\prime_1/\partial x$'))  #TODO

plt.figure()   #Fig 6
i = 35
plt.plot(dudx_2[i,:], yp2d[i,:], 'y-.')
plt.plot(dudy_2[i,:], yp2d[i,:], 'k-.')
plt.plot(du_2dx[i,:], yp2d[i,:], 'r-.')     
plt.plot(dtaudy[i,:], yp2d[i,:], 'g-.')        


plt.figure()  #Fig 7
i = 35
plt.plot(dv1v2dx[i,:], yp2d[i,:], 'r-.')      
# plt.plot(dv2v2dy[i,:], yp2d[i,:], 'b-.')   

plt.figure()  #Fig 8
i = 35
plt.plot(-dpdy[i,:], yp2d[i,:], 'k-.')
plt.plot(dvdx_2[i,:], yp2d[i,:], 'r-.')    
plt.plot(dvdy_2[i,:], yp2d[i,:], 'b-.')   
plt.plot(v1v2[i,:], yp2d[i,:])    

# plt.figure()    #Fig 9
# i = 35


# plt.contourf(xp2d, yp2d, -dpdy, vmin = -60, vmax = -20, shading='gouraud')
# plt.colorbar()


### 1.3 

# normal_pk = np.zeros([ni,nj])
# shear_pk = np.zeros([ni,nj])
normal_pk = pk1 +pk2
shear_pk = pk3+pk4

plt.figure()    #Fig 10
i=35
plt.plot(normal_pk[i,:],yp2d[i,:],'b-')
plt.plot(shear_pk[i,:], yp2d[i,:],'r-')
# plt.ylim(0, 0.015)
# plt.xlim(-0.12, 0)
plt.xlabel('Production term  $ (P_K) $')
plt.ylabel('Y')
plt.legend(('$ Normal $', '$ Shear $'))


#1.3 Plot
plt.figure(figsize=(10,6))    #Fig 11
i=10
plt.plot(p_k[i,:],yp2d[i,:],'b-')
i = 50
plt.plot(p_k[i,:], yp2d[i,:],'r-')
# i = 150
# plt.plot(p_k[i,:], yp2d[i,:],'g-')
plt.ylim(0, 0.1)
# plt.ylim(0, 0.015)
# plt.xlim(-0.12, 0)
plt.xlabel('Production term  $ (P_K) $')
plt.ylabel('Y')
plt.legend(('$ i = 10 $', '$ i = 50 $','$ i = 150 $'))


###   1.4 Plot

plt.figure()
i = 35
# plt.plot(p_k[i,:],yp2d[i,:],'b-')
plt.plot(eps_RANS2d[i,:], yp2d[i,:], 'r-.')
plt.plot(eps2d[i,:], yp2d[i,:], 'b-')

plt.show(block = 'True')


