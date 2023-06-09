import scipy.io as sio
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
#import gradients.py
from gradients import compute_face_phi,dphidx,dphidy,init
from compute_face_phi_D_or_N import compute_face_phi_D_or_N
import matplotlib.ticker as mtick
from matplotlib import ticker

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'


# plt.rcParams.update({'font.size': 22})


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
uu2d_face_w, uu2d_face_s = compute_face_phi(uu2d, fx, fy, ni, nj)
vv2d_face_w, vv2d_face_s = compute_face_phi(vv2d, fx, fy, ni, nj)
uv2d_face_w, uv2d_face_s = compute_face_phi(uv2d, fx, fy, ni, nj)


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

uu_2 = uu2d**2    #(v1')^2 ~ (u')^2
u_2face_w, u_2face_s = compute_face_phi(uu_2, fx, fy,ni, nj)      #computes face value of (v1')^2
du_2dx = dphidx(u_2face_w, u_2face_s, areawx, areasx, vol)       #RHS 4th term        


# ###

dv1v2dx = dphidx(v1v1_face_w, v1v1_face_s, areawx, areasx, vol)                                 
v2v2 = v2d * v2d                                                     #LHS 1st term
v2v2_face_w, v2v2_face_s = compute_face_phi(v2v2, fx, fy,ni, nj)
dvvdy = dphidy(v2v2_face_w, v2v2_face_s, areawy, areasy, vol)

dv2dx1_face_w, dv2dx1_face_s = compute_face_phi(dvdx, fx, fy,ni, nj)
dvdx_2 = nu * dphidx(dv2dx1_face_w, dv2dx1_face_s, areawx, areasx, vol)

dv2dx2 = dphidy(v2d_face_w, v2d_face_s, areawy, areasy, vol)
dv2dx2_face_w, dv2dx2_face_s = compute_face_phi(dv2dx2, fx, fy,ni, nj)
dvdy_2 = nu * dphidy(dv2dx2_face_w, dv2dx2_face_s, areawy, areasy, vol)


vv_2 = vv2d**2    #(v1')^2 ~ (u')^2
v2_2face_w, v2_2face_s = compute_face_phi(vv_2, fx, fy,ni, nj)      #computes face value of (v1')^2
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

v1_1 = u2d**2
v2_2 = v2d**2
u1_2 = u2d*v2d
u2_1 = v2d*u2d
p_k = (-v1_1*dudx) +  (-v2_2*dvdy) + (-u1_2*dudy) + (-u2_1*dvdx)
pk1= -v1_1*dudx
pk2= -v2_2*dv2dx2
pk3= -u1_2*dudy
pk4= -u2_1*dvdx


#### Assignment 1.5 ###
c_mu = 0.09
c1 = 1.5
c2 = 0.6
c1_w = 0.5
c2_w = 0.3
sigmak = 1
rho = 1

#production
P_11 = -2 * ((uu2d * dudx) + (uv2d * dudy))
P_22 = -2 * ((vv2d * dvdy) + (uv2d * dvdx))
P_12 = -(uu2d * dvdx) -(uv2d * dudx) -(uv2d * dvdy) -(vv2d * dudy)

#viscous terms

duudx = dphidx(uu2d_face_w, uu2d_face_s, areawx, areasx, vol)
duudx_face_w, duudx_face_s = compute_face_phi(duudx, fx, fy, ni, nj)
duudx_2 = dphidx(duudx_face_w, duudx_face_s, areawx, areasx, vol)

duudy = dphidy(uu2d_face_w, uu2d_face_s, areawy, areasy, vol)
duudy_face_w, duudy_face_s = compute_face_phi(duudy, fx, fy, ni, nj)
duudy_2 = dphidy(duudy_face_w, duudy_face_s, areawy, areasy, vol)

dvvdx = dphidx(vv2d_face_w, vv2d_face_s, areawx, areasx, vol)
dvvdx_face_w, dvvdx_face_s = compute_face_phi(dvvdx, fx, fy, ni, nj)
dvvdx_2 = dphidx(dvvdx_face_w, dvvdx_face_s, areawx, areasx, vol)


dvvdy = dphidy(vv2d_face_w, vv2d_face_s, areawy, areasy, vol)
dvvdy_face_w, dvvdy_face_s = compute_face_phi(dvvdy, fx, fy, ni, nj)
dvvdy_2 = dphidy(dvvdy_face_w, dvvdy_face_s, areawy, areasy, vol)


duvdx = dphidx(uv2d_face_w, uv2d_face_s, areawx, areasx, vol)
duvdx_face_w, duvdx_face_s = compute_face_phi(duvdx, fx, fy, ni, nj)
duvdx_2 = dphidx(duvdx_face_w, duvdx_face_s, areawx, areasx, vol)


duvdy = dphidy(uv2d_face_w, uv2d_face_s, areawy, areasy, vol)
duvdy_face_w, duvdy_face_s = compute_face_phi(duvdy, fx, fy, ni, nj)
duvdy_2 = dphidy(duvdy_face_w, duvdy_face_s, areawy, areasy, vol)

Vt_11 = nu * (duudx_2 + duudy_2)
Vt_22 = nu * (dvvdx_2 + dvvdy_2)
Vt_12 = nu * (duvdx_2 + duvdy_2)

#Pressure-strain terms
phi_11_1 = -c1 * rho * np.multiply(np.divide(eps_RANS2d, k_RANS2d), uu2d - 2*k_RANS2d/3)
phi_12_1 = -c1 * rho * np.multiply(np.divide(eps_RANS2d, k_RANS2d), uv2d)
phi_11_2 = -c2 * rho * (P_11 -2 * p_k/3)
phi_12_2 = -c2 * rho * P_12
phi_22_2 = -c2 * rho * (P_22 -2 * p_k/3)

walldistance = np.zeros((ni,nj))
walldistN = np.zeros((ni,1))
walldistS = np.zeros((ni,1))

n1n = np.zeros((ni,1))
n1s = np.zeros((ni,1))
n2n = np.zeros((ni,1))
n2s = np.zeros((ni,1))

X_n = np.zeros((ni,1))
X_s = np.zeros((ni,1))

Y_n = np.zeros((ni,1))
Y_s = np.zeros((ni,1))

f = np.zeros((ni, nj))
N_1 = np.zeros((ni, nj))
N_2 = np.zeros((ni, nj))

for i in range(ni):
   dist_n = np.sqrt((x2d[i+1, -1] - x2d[i, -1])*2 + (y2d[i+1, -1] - y2d[i, -1])*2)
   dist_s = np.sqrt((x2d[i+1, 0] - x2d[i, 0])*2 + (y2d[i+1, 0] - y2d[i, 0])*2)

   c1n = (x2d[i+1, -1] - x2d[i, -1]) / dist_n
   c1s = (x2d[i+1, 0] - x2d[i, 0]) / dist_n
   c2n = (y2d[i+1, -1] - y2d[i, -1]) / dist_s
   c2s = (y2d[i+1, 0] - y2d[i, 0]) / dist_s
   n1n[i] = c2n
   n1s[i] = -c2s
   n2n[i] = -c1n
   n2s[i] = c1s

   X_n[i] = (x2d[i+1, -1] + x2d[i, -1]) / 2
   X_s[i] = (x2d[i+1, 0] + x2d[i, 0]) / 2
   Y_n[i] = (y2d[i+1, -1] + y2d[i, -1]) / 2
   Y_s[i] = (y2d[i+1, 0] + y2d[i, 0]) / 2

   eq = np.zeros((ni, nj))

for i in range(ni):
   for j in range(nj):
      walldistN = np.sqrt((xp2d[i,j] - X_n)**2 + (yp2d[i,j] - Y_n)**2)
      walldistS = np.sqrt((xp2d[i,j] - X_s)**2 + (yp2d[i,j] - Y_s)**2)
      walldistN_min = np.amin(walldistN)
      walldistS_min = np.amin(walldistS)

      walldistance[i,j] = np.minimum(walldistN_min, walldistS_min)

      if walldistN_min < walldistS_min :
         N_1[i,j] = n1n[i]
         N_2[i,j] = n2n[i]
      else:
         N_1[i,j] = n1s[i]
         N_2[i,j] = n2s[i]
      
      eq[i,j] = k_RANS2d[i,j] ** (3/2) / 2.55 * walldistance[i,j] * eps_RANS2d[i,j]   
      f[i,j] = np.minimum(eq[i,j], 1)

N_11 = N_1 * N_1
N_22 = N_2 * N_2
N_12 = N_1 * N_2

phi_11_1wall = c1_w * ((eps_RANS2d / k_RANS2d) * (-2 * (uu2d * N_11) - (uv2d * N_12) + (vv2d * N_22))) * f
phi_12_1wall = c1_w * ((eps_RANS2d / k_RANS2d) * (-1.5 * (uu2d * N_12) + (uv2d * N_22) + (uv2d * N_11) + (vv2d * N_12))) * f
phi_11_2wall = c2_w * f * ((-2 * phi_11_2 * N_11) - (phi_12_2 * N_12) + (phi_22_2 * N_22))
phi_12_2wall = c2_w * f * (1.5 *( (phi_11_2 * N_12) + (phi_12_2 * N_22) + (phi_12_2 * N_11) + (phi_22_2 * N_12)))

phi_11 = phi_11_1 + phi_11_2 + phi_11_1wall + phi_11_2wall
phi_12 = phi_12_1 + phi_12_2 + phi_12_1wall + phi_12_2wall

#Dissipation term 
eps_11 = 2*eps_RANS2d / 3
eps_12 = np.zeros((ni,nj))   #because kornicker delta is 0

#Diffusion term
vist = c_mu * k_RANS2d**2 / eps_RANS2d
vist_face_w, vist_face_s = compute_face_phi(vist, fx, fy, ni, nj)
dvistdx = dphidx(vist_face_w, vist_face_s, areawx, areasx, vol)
dvistdy = dphidy(vist_face_w, vist_face_s, areawy, areasy, vol)
D11 = vist * (duudx_2 + duudy_2) + ((duudx + duudy) * (dvistdx + dvistdy)) / sigmak
D12 = vist * (duvdx_2 + duvdy_2) + ((duvdx + duvdy) * (dvistdx + dvistdy)) / sigmak

#### Assignment 1.6 - Reynolds stresses using Boussinseq Assumption 

visc_t = c_mu * k_RANS2d**2 / eps_RANS2d

reystress_11 = -2*visc_t*dudx + 2/3*k_RANS2d
reystress_22 = -2*visc_t*dvdy + 2/3*k_RANS2d
reystress_12 = -2*visc_t*0.5*(dudy + dvdx)    # s_ij is symmetric, thus reystress_12 = reystress_21

production_k12 = 2*visc_t* 0.5*(dudy + dvdx)*0.5*(dudy + dvdx)
production_k11 = 2*visc_t*dudx*dudx
production_k22 = 2*visc_t*dvdy*dvdy

production_k_total = production_k11 + production_k12 + production_k22 + production_k12
print(np.min(production_k_total))
print(np.max(p_k))

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
plt.legend(('i = 1', 'i = 35'))

#plt.savefig('uv_python.png')


###### plot - reynolds stress


plt.figure()     #Fig 4
i=1
plt.plot(dtaudx[i,:],yp2d[i,:],'b-')
i = 35
plt.plot(dtaudx[i,:], yp2d[i,:],'r-')
# plt.ylim(0, 0.015)
# plt.xlim(-0.12, 0)
plt.xlabel(r'Reynolds Stress  $ (\tau_{ij}) $')
plt.ylabel('y')
plt.legend(('i = 1', 'i = 35'))

# ### plot - 1.2                                          #TODO - Add x and y labels, legend, plot title

plt.figure()    #Fig 5
i = 10
# 
plt.plot(duudx[i,:], yp2d[i,:], 'b--')        
plt.plot(duvdy[i,:], yp2d[i,:], 'r--')      
plt.legend(( r'$\partial \bar{u}^\prime_1 \bar{u}^\prime_1/\partial x$', r'$\partial \bar{u}^\prime_1 \bar{v}^\prime_1/\partial x$'))
# plt.xlabel(r'Reynolds Stress  $ (\tau_{ij}) $')
plt.ylabel('y')
plt.ylim(0,0.15)
# 

plt.figure()   #Fig 6
i = 35
plt.plot(-dpdx[i,:], yp2d[i,:], 'b-.')
plt.plot(dudx_2[i,:], yp2d[i,:], 'y-.')
plt.plot(dudy_2[i,:], yp2d[i,:], 'k-.')
plt.plot(du_2dx[i,:], yp2d[i,:], 'r-.')     
plt.plot(-dtaudy[i,:], yp2d[i,:], 'g-.') 
plt.legend((r'$\partial \bar{p}/\partial x$', r'$\partial \bar{u}^{\prime 2}_1 /\partial x$', r'$\overline{u^\prime v^\prime}/\partial y$', r'$\mu \partial^2 \bar{u}_1/\partial x^2$', r'$\mu \partial^2 \bar{u}_1/\partial y^2$'))
plt.ylabel('y')
# plt.ylim(-0.15, 0.15)


plt.figure()  #Fig 7
i = 35
plt.plot(dv1v2dx[i,:], yp2d[i,:], 'r-.')      
plt.plot(dvvdy[i,:], yp2d[i,:], 'b-.') 
plt.ylabel('y')
plt.legend((r'$\partial{\bar{v}_1\bar{v}_2}/\partial x_1 $', r'$\partial{\bar{v}_2 \bar{v}_2}/\partial{x}_2$'))  

plt.figure()  #Fig 8
i = 35
plt.plot(-dpdy[i,:], yp2d[i,:], 'k-.')
plt.plot(dvdx_2[i,:], yp2d[i,:], 'r-.')    
plt.plot(dvdy_2[i,:], yp2d[i,:], 'b-.')   
plt.plot(dv_2dy[i,:], yp2d[i,:], 'g-.')  
plt.plot(dtaudx[i,:], yp2d[i,:], 'm-.')  
plt.legend((r'$-\partial \bar{p}/\partial y $', r'$\mu \partial^2 \bar{u}_2/\partial x^2$', r'$\mu \partial^2 \bar{u}_2/\partial y^2$',r'$\partial \overline{{{v}^\prime}^2}/\partial y$', r'$\partial \overline{u^\prime v^\prime}/\partial x$'))
plt.ylabel('y')

 
# # plt.figure()    #Fig 9
# i = 35
# plt.contourf(xp2d, yp2d, -dpdy, vmin = -60, vmax = -20, shading='gouraud')
# plt.colorbar()

# #1.3 Plot
plt.figure()    #Fig 9
i=10
plt.plot(p_k[i,:],yp2d[i,:],'b-.')
i = 35
plt.plot(p_k[i,:], yp2d[i,:],'r-.')
# plt.ylim(0, 0.1)
plt.xlabel('Production term  $ (P^k) $')
plt.ylabel('Y')
plt.legend(('i = 10 ', ' i = 50 '))

plt.figure()   #Fig 10
i = 35
plt.plot(pk1[i,:], yp2d[i,:], 'r-.')
plt.plot(pk2[i,:], yp2d[i,:], 'b-.')
plt.plot(pk3[i,:], yp2d[i,:], 'k-.')
plt.plot(pk4[i,:], yp2d[i,:], 'g-.')
plt.ylabel('y/H')
plt.xlabel('Production Term')
plt.legend((r'$P^k_{11}$', r'$P^k_{22}$', r'$P^k_{12}$', r'$P^k_{21}$'))


# ###   1.4 Plot
plt.figure()    #Fig 11
i = 35
plt.plot(p_k[i,:],yp2d[i,:],'b-.')
plt.plot(eps_RANS2d[i,:], yp2d[i,:], 'r-.')
plt.ylabel('y/H')
plt.legend((r'Production Term $(P^k)$', r'Dissipation $(\epsilon)$'))

# 1.6 plots ##

plt.figure()   #Fig    12             
# plt.subplots_adjust(left=0.20,top=0.80,bottom=0.20)
i = 50
plt.plot(uv2d[i,:], yp2d[i,:], 'b-.')
plt.plot(reystress_12[i,:-1], yp2d[i,:-1], 'r-.')
plt.xlabel(r'Reynolds Stress $\overline{u^\prime v^\prime}$ at i=50')
plt.ylabel('y/H')
plt.legend((r'$\overline{u^\prime v^\prime}$ - database', r'$\overline{u^\prime v^\prime}$ - calculated'))


plt.figure()   #Fig    13             
i = 50
plt.plot(uu2d[i,:], yp2d[i,:], 'b-.')
plt.plot(reystress_11[i,:-1], yp2d[i,:-1], 'r-.')
plt.xlabel(r'Reynolds Stress $\overline{u^\prime u^\prime}$ at i=50')
plt.ylabel('y/H')
plt.legend((r'$\overline{u^\prime u^\prime}$ - database', r'$\overline{u^\prime u^\prime}$ - calculated'))

# plt.figure()
# i = 52
# plt.plot(pk1[i,:], yp2d[i,:])
# plt.plot(pk2[i,:], yp2d[i,:])
# plt.plot(pk3[i,:], yp2d[i,:])
# plt.plot(pk4[i,:], yp2d[i,:])
# plt.xlabel('Individual Production terms$')
# plt.ylabel('Y')
# plt.legend(('pk_1', 'pk_2', 'pk_3', 'pk_4'))
# plt.xlim(-0.05, 0.05)


# plot 1.7 

plt.figure()  
plt.pcolormesh(xp2d,yp2d,p_k, vmin=-0.2,vmax=0.2,cmap=plt.get_cmap('hot'),shading='gouraud')
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Production Term")

plt.figure()
i = 55
plt.plot(production_k12[i,:], yp2d[i,:], 'r-.')
plt.plot(eps_RANS2d[i,:], yp2d[i,:], 'b-.')
plt.legend((r'$P_k$ at i = 55', r'$\epsilon RANS$ at i = 55'))
plt.ylabel('y/H')

# plot 1.5 
i = 35
plt.figure()
plt.plot(Vt_11[i,:], yp2d[i,:], 'r-.')
plt.plot(P_11[i,:], yp2d[i,:], 'b-.')
plt.plot(phi_11[i,:], yp2d[i,:], 'k-.')
plt.plot(D11[i,:], yp2d[i,:], 'g-.')
plt.plot(eps_11[i,:], yp2d[i,:], 'c-.')
plt.ylabel('y/H')
plt.xlabel('At x = 35')
# plt.xlabel('Production Term')
plt.legend((r'$\mu_t$', r'$P_{11}$', r'$\phi_{11}$', r'$D_{11}$', r'$\epsilon_{11}$'))

plt.figure()
plt.plot(Vt_12[i,:], yp2d[i,:], 'r-.')
plt.plot(P_12[i,:], yp2d[i,:], 'b-.')
plt.plot(phi_12[i,:], yp2d[i,:], 'k-.')
plt.plot(D12[i,:], yp2d[i,:], 'g-.')
plt.plot(eps_12[i,:], yp2d[i,:], 'c-.')
plt.ylabel('y/H')
plt.xlabel('At x = 35')
# plt.xlabel('Production Term')
plt.legend((r'$\mu_t$', r'$P_{12}$', r'$\phi_{12}$', r'$D_{12}$', r'$\epsilon_{12}$'))

plt.show(block = 'True')


