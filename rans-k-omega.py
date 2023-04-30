import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from joblib import dump, load

# from tdma import tdma
#from IPython import display
plt.rcParams.update({'font.size': 22})

plt.interactive(True)

plt.close('all')

# This file can be downloaded at

#

# exemple of 1d Channel flow with a k-omegaa model. Re=u_tau*h/nu=5200 (h=half 
# channel height).
#
# Discretization described in detail in
# http://www.tfd.chalmers.se/~lada/comp_fluid_dynamics/

# folder = 'D:\Chalmers - Academic Files\SP4\Turbulence Modeling\TurbulenceModeling-Assignment-1\TurbulenceModeling-Assignment-1'
folder = './'
filename = str(folder)+'model-svr.bin'
model = load(str(folder)+'model-svr.bin')
scaler_dudy = load(str(folder)+'scalar-dudy-svr.bin')
dudy_max, dudy_min = np.loadtxt(str(folder)+ 'min-max-svr.txt')

# max number of iterations
niter=25000

plt.rcParams.update({'font.size': 22})


# friction velocity u_*=1
# half channel width=1
#

# create the grid

nj=30 # coarse grid
nj=98 # fine grid
njm1=nj-1
#yfac=1.6 # coarse grid
yfac=1.15 # fine grid
dy=0.1
yc=np.zeros(nj)
delta_y=np.zeros(nj)
yc[0]=0.
for j in range(1,int(nj/2)):
    yc[j]=yc[j-1]+dy
    dy=yfac*dy

ymax=yc[int(nj/2)-1]

# cell faces
for j in range(0,int(nj/2)):
   yc[j]=yc[j]/ymax
   yc[nj-1-j]=2.-yc[j-1]
yc[nj-1]=2.

# cell centres
yp=np.zeros(nj)
for j in range(1,nj-1):
   yp[j]=0.5*(yc[j]+yc[j-1])
yp[nj-1]=yc[nj-1]

# viscosity
viscos=1./5200.

# under-relaxation
urf=0.5

# plot k for each iteration at node jmon
jmon=8 

# turbulent constants 
c_omega_1= 5./9.
c_omega_2=3./40.
prand_omega=2.0
prand_k=2.0
cmu=0.09

small=1.e-10
great=1.e10

# initialaze
u=np.zeros(nj)
k=np.ones(nj)*1.e-4
y=np.zeros(nj)
om=np.ones(nj)*1.
vist=np.ones(nj)*100.*viscos
dn=np.zeros(nj)
ds=np.zeros(nj)
dy_s=np.zeros(nj)
fy=np.zeros(nj)
tau_w=np.zeros(niter)
k_iter=np.zeros(niter)
om_iter=np.zeros(niter)


# do a loop over all nodes (except the boundary nodes)
for j in range(1,nj-1):

# compute dy_s
   dy_s[j]=yp[j]-yp[j-1]

# compute dy_n
   dy_n=yp[j+1]-yp[j]

# compute deltay
   delta_y[j]=yc[j]-yc[j-1]
 
   dn[j]=1./dy_n
   ds[j]=1./dy_s[j]

# interpolation factor
   del1=yc[j]-yp[j]
   del2=yp[j+1]-yc[j]
   fy[j]=del1/(del1+del2)

u[1]=0.


vist[0]=0.
vist[nj-1]=0.
k[0]=0.
k[nj-1]=0.


su=np.zeros(nj)
sp=np.zeros(nj)
an=np.zeros(nj)
as1=np.zeros(nj)
ap=np.zeros(nj)
# do max. niter iterations
for n in range(1,niter):

    for j in range(1,nj-1):

# compute turbulent viscosity
      vist_old=vist[j]
      vist[j]=urf*k[j]/om[j]+(1.-urf)*vist_old


# solve u
    for j in range(1,nj-1):

# driving pressure gradient
      su[j]=delta_y[j]

      sp[j]=0.

# interpolate turbulent viscosity to faces
      vist_n=fy[j]*vist[j+1]+(1.-fy[j])*vist[j]
      vist_s=fy[j-1]*vist[j]+(1.-fy[j-1])*vist[j-1]

# compute an & as
      an[j]=(vist_n+viscos)*dn[j]
      as1[j]=(vist_s+viscos)*ds[j]

# boundary conditions for u
    u[0]=0.
    u[nj-1]=0.


    for j in range(1,nj-1):
# compute ap
      ap[j]=an[j]+as1[j]-sp[j]

# under-relaxate
      ap[j]= ap[j]/urf
      su[j]= su[j]+(1.0-urf)*ap[j]*u[j]

# use Gauss-Seidel
      u[j]=(an[j]*u[j+1]+as1[j]*u[j-1]+su[j])/ap[j]


# monitor the development of u_tau in node jmon
    tau_w[n]=viscos*u[1]/yp[1]

# print iteration info
    tau_target=1
    #print(f"\n{'---iter: '}{n:2d}, {'wall shear stress: '}{tau_w[n]:.2e}\n")
    print(f"\n{'---iter: '}{n:2d}, {'wall shear stress: '}{tau_w[n]:.2e},{'  tau_w_target='}{tau_target:.2e}\n")

# check for convergence (when converged, the wall shear stress must be one)
    ntot=n
    if abs(tau_w[n]-1) < 0.001:
# do at least 1000 iter 
        if n > 1000:
           print('Converged!')
           break

# solve k
    dudy=np.gradient(u,yp)
    dudy2=dudy**2
    for j in range(1,nj-1):

# production term
      su[j]=vist[j]*dudy2[j]*delta_y[j]

# dissipation term
      ret=k[j]/(viscos*om[j])
      ret=max(ret,1.e-5)

      sp[j]=-cmu*om[j]*delta_y[j]

# compute an & as
      vist_n=fy[j]*vist[j+1]+(1.-fy[j])*vist[j]
      an[j]=(vist_n/prand_k+viscos)*dn[j]
      vist_s=fy[j-1]*vist[j]+(1.-fy[j-1])*vist[j-1]
      as1[j]=(vist_s/prand_k+viscos)*ds[j]

# boundary conditions for k
    k[0]=0.
    k[nj-1]=0.

    for j in range(1,nj-1):
# compute ap
      ap[j]=an[j]+as1[j]-sp[j]

# under-relaxate
      ap[j]= ap[j]/urf
      su[j]= su[j]+(1.-urf)*ap[j]*k[j]

# use Gauss-Seidel
      k[j]=(an[j]*k[j+1]+as1[j]*k[j-1]+su[j])/ap[j]


# monitor the development of k in node jmon
    k_iter[n]=k[jmon]

#****** solve om-eq.
    for j in range(1,nj-1):
# compute an & as
      vist_n=fy[j]*vist[j+1]+(1.-fy[j])*vist[j]
      an[j]=(vist_n/prand_omega+viscos)*dn[j]
      vist_s=fy[j-1]*vist[j]+(1.-fy[j-1])*vist[j-1]
      as1[j]=(vist_s/prand_omega+viscos)*ds[j]

# production term
      su[j]=c_omega_1*dudy2[j]*delta_y[j]

# dissipation term
      sp[j]=-c_omega_2*om[j]*delta_y[j]

# b.c. south wall
    dy=yp[1]
    omega=6.*viscos/0.075/dy**2
    sp[1]=-great
    su[1]=great*omega

# b.c. north wall
    dy=yp[nj-1]-yp[nj-2]
    omega=6.*viscos/0.075/dy**2
    sp[nj-2]=-great
    su[nj-2]=great*omega

    for j in range(1,nj-1):
# compute ap
      ap[j]=an[j]+as1[j]-sp[j]

# under-relaxate
      ap[j]= ap[j]/urf
      su[j]= su[j]+(1.-urf)*ap[j]*om[j]

# use Gauss-Seidel
      om[j]=(an[j]*om[j+1]+as1[j]*om[j-1]+su[j])/ap[j]

    om_iter[n]=om[jmon]


# dudy_min_number = np.zeros([nj]) 
# dudy_max_number = np.zeros([nj])
# N = np.zeros([nj])
# y_svr = np.zeros([nj])

# if n > 1000:
#    for i in range(1, nj-1):
      
#  #flatten
# dudy= dudy.flatten()

# #count values larger/smaller than max/min
# dudy_min_number = (dudy < dudy_min)
# dudy_max_number = (dudy > dudy_max)

# #set limits
# dudy = np.minimum(dudy, dudy_max)
# dudy = np.maximum(dudy, dudy_min)

# #size
# N = len(dudy)

# #re-sclae
# dudy = dudy.reshape(-1,1)
# dudy = scaler_dudy.transform(dudy)

# #predict
# X = np.zeros((N,1))
# X[:,0] = dudy[:,0]

# #compute dudy
# y_svr = model.predict(X)


# compute shear stress
uv=-vist*dudy

DNS_mean=np.genfromtxt("LM_Channel_5200_mean_prof.dat",comments="%")
y_DNS=DNS_mean[:,0];
yplus_DNS=DNS_mean[:,1];
u_DNS=DNS_mean[:,2];

DNS_stress=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat",comments="%")
u2_DNS=DNS_stress[:,2];
v2_DNS=DNS_stress[:,3];
w2_DNS=DNS_stress[:,4];
uv_DNS=DNS_stress[:,5];


k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)

ustar=tau_w[ntot]**0.5
yplus = yp*ustar/viscos
uplus=u/ustar

#ML Training and prediction 

#count values larger/smaller than max/min
dudy_min_number = (dudy < dudy_min)
dudy_max_number = (dudy > dudy_max)

#set limits
dudy_test = np.minimum(dudy, dudy_max)
dudy_test = np.maximum(dudy, dudy_min)

#size
N = len(dudy_test)

# re-shape
dudy_test=dudy_test.reshape(-1, 1)

# scale input data 
scaler_dudy=StandardScaler()
dudy_test=scaler_dudy.fit_transform(dudy_test)

# setup X (input) and y (output)
X=np.zeros((N,1))
y = cmu
X[:,0]=dudy_test[:,0]


#  re-shape test data
dudy_test=dudy_test.reshape(-1, 1)


# setup X (input) for testing (predicting)
X_test=np.zeros((N,1))
X_test[:,0]=dudy_test[:,0]

# predict cmu
cmu_predict= model.predict(X_test)

# find difference between ML prediction and target
cmu_error=np.std(cmu_predict-cmu)/\
(np.mean(cmu_predict**2))**0.5
print('\nRMS error using ML turbulence model',cmu_error)

yplus_ML = yplus
vist_ML = vist
k_ML = k

u_ML = (vist_ML * k_ML * omega * cmu_predict) ** (1/4)

# plot u
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(u,yp,'b-',label="CFD")
plt.plot(u_DNS,y_DNS,'r-',label="DNS")
plt.plot(u_ML,yp,'g-',label="ML")

plt.ylabel("y")
plt.xlabel("$U$")
plt.legend(loc="best",prop=dict(size=18))
# plt.savefig('u_5200.png')

# plot u log-scale
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.semilogx(yplus,uplus,'b-',label="CFD")
plt.semilogx(yplus_DNS,u_DNS,'r-',label="DNS")
plt.semilogx(yplus_ML,u_ML,'g-',label="ML")
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")
plt.axis([1, 5200, 0, 28])
plt.legend(loc="best",prop=dict(size=18))
# plt.savefig('u_log-5200.png')

# plot k
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(k,yp,'b-',label="CFD")
plt.plot(k_DNS,y_DNS,'r-',label="DNS")
plt.plot(k_ML,yp,'g-',label="ML")
# plt.plot(cmu,y_DNS,'r-',label="DNS")
# plt.plot(cmu_predict,yp,'g-',label="ML")
plt.legend(loc="best",prop=dict(size=18))
plt.xlabel('k')
plt.ylabel('y')
# plt.savefig('k_5200.png')


# plot tau_w versus iteration number
# fig1,ax1 = plt.subplots()
# plt.subplots_adjust(left=0.20,bottom=0.20)
# plt.plot(tau_w[0:ntot],'b-')
# #plt.plot(tau_w[0:ntot],'b-')
# plt.title('wall shear stress')
# plt.xlabel('Iteration number')
# plt.ylabel('tauw')
# plt.savefig('tauw.png')

# plot k(jmon) versus iteration number
# fig1,ax1 = plt.subplots()
# plt.subplots_adjust(left=0.20,bottom=0.20)
# plt.plot(k_iter[0:ntot],'b-')
# #plt.plot(k_iter[0:ntot],'b-')
# plt.title('k in node jmon')
# plt.xlabel('Iteration number')
# plt.ylabel('k')
# plt.savefig('k_iter.png')

# plot om(jmon) versus iteration number
# fig1,ax1 = plt.subplots()
# plt.subplots_adjust(left=0.20,bottom=0.20)
# plt.plot(om_iter[0:ntot],'b-')
# #plt.plot(om_iter[0:ntot],'b-')
# plt.title('omega in node jmon')
# plt.xlabel('Iteration number')
# plt.ylabel('omega')
# plt.savefig('om_iter.png')

# save data
data=np.zeros((nj,7))
data[:,0]=yp
data[:,1]=u
data[:,2]=k
data[:,3]=om
data[:,4]=vist
data[:,5]=uv
data[:,6]=yc
np.savetxt('yp_u_k_om_vist_uv_yc_PDH_5200.dat', data)



plt.show(block = 'True')

