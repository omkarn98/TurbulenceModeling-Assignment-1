import scipy.io as sio
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
#import gradients.py
from gradients import compute_face_phi,dphidx,dphidy,init
plt.rcParams.update({'font.size': 12})

# modify global setting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

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

u2d=np.reshape(u,(nj,ni))
v2d=np.reshape(v,(nj,ni))
p2d=np.reshape(p,(nj,ni))
x2d=np.reshape(x,(nj,ni))
y2d=np.reshape(y,(nj,ni))
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

k_RANS2d=np.reshape(k_RANS,(nj,ni))/ntstep
eps_RANS2d=np.reshape(eps_RANS,(nj,ni))/ntstep        #dissipation term
vist_RANS2d=np.reshape(vist_RANS,(nj,ni))/ntstep      #turbulent viscosity

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
u2d_face_w,u2d_face_s=compute_face_phi(u2d,fx,fy,ni,nj)
v2d_face_w,v2d_face_s=compute_face_phi(v2d,fx,fy,ni,nj)
p2d_face_w,p2d_face_s=compute_face_phi(p2d,fx,fy,ni,nj)

# x derivatives
dudx=dphidx(u2d_face_w,u2d_face_s,areawx,areasx,vol)
dvdx=dphidx(v2d_face_w,v2d_face_s,areawx,areasx,vol)
dpdx=dphidx(p2d_face_w,p2d_face_s,areawx,areasx,vol)

# x derivatives
dudy=dphidx(u2d_face_w,u2d_face_s,areawy,areasy,vol)
dvdy=dphidx(v2d_face_w,v2d_face_s,areawy,areasy,vol)
dpdy=dphidx(p2d_face_w,p2d_face_s,areawy,areasy,vol)

omega2d=eps2d/k2d/0.09

tau = np.zeros([ni,nj])

########  Assignment 1.1 - Reynolds Stresses ##########
rho = 1
for i in range(1, ni-1):
   for j in range(1, nj-1):
      tau[i,j] = rho * uv2d[i,j]

tau2d_face_w, tau2d_face_s = compute_face_phi(tau, fx, fy,ni, nj)
dtaudx = dphidx(tau2d_face_w, tau2d_face_s, areawy, areasy, vol)

#####   Assignment 1.2 ######  #TODO y equation


v1v1 = np.zeros([ni, nj])                                           #LHS 1st term
v1v1[i,j] = u2d[i,j] * u2d[i,j]
v1v1_face_w, v1v1_face_s = compute_face_phi(v1v1, fx, fy,ni, nj)
dv1v1dx = dphidx(v1v1_face_w, v1v1_face_s, areawy, areasy, vol)

v1v2 = np.zeros([ni, nj])                                           #LHS 2nd term
v1v2[i,j] = u2d[i,j] * v2d[i,j]
v1v2_face_w, v1v2_face_s = compute_face_phi(v1v2, fx, fy,ni, nj)
dv1v2dy = dphidy(v1v1_face_w, v1v1_face_s, areawx, areasx, vol)

dv1dx1 = dphidx(u2d_face_w, u2d_face_s, areawy, areasy, vol)
dv1dx1_face_w, dv1dx1_face_s = compute_face_phi(dv1dx1, fx, fy,ni, nj)
dv1dx1_2 = nu * dphidx(dv1dx1_face_w, dv1dx1_face_s, areawy, areasy, vol)    #RHS 2nd term  (R1 eq)

dv1dx2 = dphidy(u2d_face_w, u2d_face_s, areawx, areasx, vol)
dv1dx2_face_w, dv1dx2_face_s = compute_face_phi(dv1dx2, fx, fy,ni, nj)
dv1dx2_2 = nu * dphidy(dv1dx2_face_w, dv1dx2_face_s, areawx, areasx, vol)    #RHS 3rd term 

v1_2 = np.zeros([ni,nj])
v1_2[i,j] = u2d[i,j]**2    #(v1')^2 ~ (u')^2
v1_2face_w, v1_2face_s = compute_face_phi(v1_2, fx, fy,ni, nj)      #computes face value of (v1')^2
dv1_2dx1 = dphidx(v1_2face_w, v1_2face_s, areawy, areasy, vol)      #RHS 4th term        


dtaudy = dphidy(tau2d_face_w, tau2d_face_s, areawx, areasx, vol)

###

dv1v2dx = dphidx(v1v1_face_w, v1v1_face_s, areawy, areasy, vol)
v2v2 = np.zeros([ni, nj])                                           #LHS 1st term
v2v2[i,j] = v2d[i,j] * v2d[i,j]
v2v2_face_w, v2v2_face_s = compute_face_phi(v2v2, fx, fy,ni, nj)
dv2v2dy = dphidy(v2v2_face_w, v2v2_face_s, areawx, areasx, vol)

dv2dx1 = dphidx(v2d_face_w, v2d_face_s, areawy, areasy, vol)
dv2dx1_face_w, dv2dx1_face_s = compute_face_phi(dv2dx1, fx, fy,ni, nj)
dv2dx1_2 = nu * dphidx(dv2dx1_face_w, dv2dx1_face_s, areawy, areasy, vol)

dv2dx2 = dphidy(v2d_face_w, v2d_face_s, areawx, areasx, vol)
dv2dx2_face_w, dv2dx2_face_s = compute_face_phi(dv2dx2, fx, fy,ni, nj)
dv2dx2_2 = nu * dphidy(dv2dx2_face_w, dv2dx2_face_s, areawx, areasx, vol)

v2_2 = np.zeros([ni,nj])
v2_2[i,j] = v2d[i,j]**2    #(v1')^2 ~ (u')^2
v2_2face_w, v2_2face_s = compute_face_phi(v2_2, fx, fy,ni, nj)      #computes face value of (v1')^2
dv2_2dx2 = dphidy(v2_2face_w, v2_2face_s, areawx, areasx, vol)      #RHS 4th term        



####### Assignment 1.3 - Production Term   #########







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
i=10
plt.plot(uv2d[i,:],yp2d[i,:],'b-')
i = 50
plt.plot(uv2d[i,:], yp2d[i,:],'r-')
plt.xlabel('$\overline{u^\prime v^\prime}$')
plt.ylabel('y/H')
#plt.savefig('uv_python.png')


###### plot - reynolds stress


plt.figure(figsize=(10,6))    #Fig 4
i=1
plt.plot(dtaudx[i,:],yp2d[i,:],'b-')
i = 10
plt.plot(dtaudx[i,:], yp2d[i,:],'r-')
# plt.ylim(0, 0.015)
# plt.xlim(-0.12, 0)
plt.xlabel(r'Reynolds Stress  $ (\tau_{ij}) $')
plt.ylabel('Velocity')
plt.legend(('$ i = 1 $', '$ i = 10 $'))

### plot - 1.2                                                                #TODO - Add x and y labels, legend, plot title

plt.figure()    #Fig 5
i = 35
plt.plot(-dpdx[i,:], yp2d[i,:], 'b-.')
plt.plot(dv1v1dx[i,:], yp2d[i,:], 'r-.')        #0 gradient
# plt.plot(dv1v2dy[i,:], yp2d[i,:], 'g-.')      #0 gradeint
plt.legend(('$\partial p/\partial x$', '$\partial \bar{v}^\prime_1 \bar{v}^\prime_1/\partial x$'))  #TODO

plt.figure()   #Fig 6
i = 35
plt.plot(dv1dx1_2[i,:], yp2d[i,:], 'y-.')
plt.plot(dv1dx2_2[i,:], yp2d[i,:], 'k-.')
plt.plot(dv1_2dx1[i,:], yp2d[i,:], 'r-.')     #0 gradient
plt.plot(dtaudy[i,:], yp2d[i,:], 'g-.')        


plt.figure()  #Fig 7
i = 35
plt.plot(dv1v2dx[i,:], yp2d[i,:], 'r-.')     #0 gradient 
# plt.plot(dv2v2dy[i,:], yp2d[i,:], 'b-.')   #0 gradient

plt.figure()  #Fig 8
i = 35
plt.plot(-dpdy[i,:], yp2d[i,:], 'k-.')
plt.plot(dv2dx1_2[i,:], yp2d[i,:], 'r-.')    # 0 gradient
plt.plot(dv2dx2_2[i,:], yp2d[i,:], 'b-.')  # 0 gradient 

plt.figure()    #Fig 9
i = 35
plt.plot(dv2_2dx2[i,:], yp2d[i,:])   # 0 gradient 
plt.plot(dtaudx[i,:], yp2d[i,:])

# plt.contourf(xp2d, yp2d, -dpdy, vmin = -60, vmax = -20, shading='gouraud')
# plt.colorbar()

plt.show(block = 'True')


