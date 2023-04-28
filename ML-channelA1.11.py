import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({'font.size': 22})

viscos=1/5200

plt.close('all')
plt.interactive(True)

# load DNS data
DNS_mean=np.genfromtxt("LM_Channel_5200_mean_prof.dat",comments="%")
y_DNS=DNS_mean[:,0]
yplus_DNS=DNS_mean[:,1]
u_DNS=DNS_mean[:,2]
dudy_DNS=np.gradient(u_DNS,y_DNS)

DNS_stress=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat",comments="%")
uu_DNS=DNS_stress[:,2]
vv_DNS=DNS_stress[:,3]
ww_DNS=DNS_stress[:,4]
uv_DNS=DNS_stress[:,5]
k_DNS=0.5*(uu_DNS+vv_DNS+ww_DNS)

DNS_RSTE=np.genfromtxt("LM_Channel_5200_RSTE_k_prof.dat",comments="%")
eps_DNS=DNS_RSTE[:,7]/viscos # it is scaled with ustar**4/viscos

# fix wall
eps_DNS[0]=eps_DNS[1]
vist_DNS=abs(uv_DNS)/dudy_DNS

# load data from k-omega RANS
data = np.loadtxt('y_u_k_om_uv_5200-RANS-code.dat')
y_rans = data[:,0]
k_rans = data[:,2]
# interpolate to DNS grid
k_rans_DNS=np.interp(y_DNS, y_rans, k_rans)


# vist and diss of k-omega model agree well with DNS, but not k. Hence omega is taken from diss and vist
# vist = cmu*k**2/eps
# omega = eps/k = eps/(vist*eps/cmu)**0.5 = (eps/vist/cmu)**0.5
omega_DNS=(eps_DNS/0.09/vist_DNS)**0.5


# turbulence model: uv = -cmu*k/omega*dudy => cmu=-uv/(k*dudy)*omega
# Input data: dudy
# output, to be predicted: cmu. interpolate to k-omega grid
cmu_DNS=-uv_DNS/(k_DNS*dudy_DNS)*omega_DNS
# fix cmu at the wall
cmu_DNS[0]=1
cmu_all_data=cmu_DNS

# input dudy
dudy_all_data=dudy_DNS
vist_all_data = vist_DNS
uv_all_data = uv_DNS


# choose values for 30 < y+ < 1000
index_choose=np.nonzero((yplus_DNS > 30 )  & (yplus_DNS< 1000 ))
yplus_DNS=yplus_DNS[index_choose]
dudy_all_data= dudy_all_data[index_choose]
cmu_all_data= cmu_all_data[index_choose]
vist_all_data = vist_all_data[index_choose]
uv_all_data = uv_all_data[index_choose]
#   ....... do this for all varibles


# create indices for all data
index= np.arange(0,len(cmu_all_data), dtype=int)

# number of elements of test data, 20%
n_test=int(0.2*len(cmu_all_data))

# pick 20% elements randomly (test data)
index_test=np.random.choice(index, size=n_test, replace=False)
# pick every 5th elements 
index_test=index[::5]

dudy_test=dudy_all_data[index_test]
cmu_out_test=cmu_all_data[index_test]
vist_test = vist_all_data[index_test]
uv_out_test = uv_all_data[index_test]

n_test=len(dudy_test)

# delete testing data from 'all data' => training data
dudy_in=np.delete(dudy_all_data,index_test)
vist_in=np.delete(vist_all_data,index_test)
cmu_out=np.delete(cmu_all_data,index_test)
uv_out=np.delete(uv_all_data,index_test)
n_svr=len(cmu_out)

# re-shape
dudy_in=dudy_in.reshape(-1, 1)
vist_in=vist_in.reshape(-1,1)

# # find min & max  
# dudy_min = np.min(dudy_in)
# dudy_max = np.max(dudy_in)

# scale input data 
scaler_dudy=StandardScaler()
scaler_vist = StandardScaler()
dudy_in=scaler_dudy.fit_transform(dudy_in)
vist_in=scaler_vist.fit_transform(vist_in)

# setup X (input) and y (output)
X=np.zeros((n_svr,2))
y=uv_out
X[:,0]=dudy_in[:,0]
X[:,1]=vist_in[:,0]

print('starting SVR')

# choose Machine Learning model
C=1
eps=0.0001
# use Linear model
# model = LinearSVR(epsilon = eps , C = C, max_iter=1000)
model = SVR(kernel='rbf', epsilon = eps, C = C)

# Fit the model
svr = model.fit(X, y.flatten())

#  re-shape test data
dudy_test=dudy_test.reshape(-1, 1)
vist_test = vist_test.reshape(-1, 1)

# scale test data
dudy_test=scaler_dudy.transform(dudy_test)
vist_test=scaler_dudy.transform(vist_test)

# setup X (input) for testing (predicting)
X_test=np.zeros((n_test,2))
X_test[:,0]=dudy_test[:,0]
X_test[:,1]=vist_test[:,0]

# predict cmu
uv_predict= model.predict(X_test)

# find difference between ML prediction and target
uv_error=np.std(uv_predict-uv_out_test)/\
(np.mean(uv_predict**2))**0.5
print('\nRMS error using ML turbulence model',uv_error)


cmu_predict = np.zeros(len(uv_predict))

for i in range(len(uv_predict)):
    cmu_predict[i] = -uv_predict[i] / vist_all_data[index_test][i] / dudy_all_data[index_test][i]
yplus_test = yplus_DNS[index_test]

################### 2D scatter top view plot all points, both test and y_svr
# fig1,ax1 = plt.subplots()
# plt.subplots_adjust(left=0.20,bottom=0.20)
# ax=plt.gca()

# plot all points
plt.figure()
plt.scatter(scaler_dudy.inverse_transform(dudy_test), uv_out_test,marker='o', s=20.2,c='green',label='Target')
plt.scatter(scaler_dudy.inverse_transform(dudy_test), uv_predict,marker='o', s=20.2,c='blue',label='Predicted')

#label axes
plt.ylabel(r"$\overline{u'v'}$")
# plt.ylabel(r'$C_\mu$')
plt.xlabel('$\partial U^+/\partial y$')
# plt.axis([0,2500,0,1.1])
# plt.legend(loc="upper left",prop=dict(size=12))

plt.figure()
plt.scatter(scaler_dudy.inverse_transform(dudy_test), uv_out_test,marker='o', s=20.2,c='green', label = 'Target')
plt.scatter(scaler_dudy.inverse_transform(dudy_test), uv_predict,marker='o', s=20.2,c='blue', label = "Predicted")
plt.legend(('Target', 'Predicted'))
plt.ylabel("$\overline{u'v'}$")
plt.xlabel("$\partial U^+/\partial y$")
# plt.axis([0, 100, 0.4,1])


plt.figure()
plt.scatter(yplus_DNS, cmu_all_data, marker = "o", label = "DNS")
plt.scatter(yplus_test, cmu_predict, marker = "o", label = "Machine Learning")
plt.xlabel("$y+$")
plt.ylabel(r"$C_\mu$")
plt.show(block = 'True')


# dump(model, 'model-svr.bin')
# dump(scaler_dudy, 'scalar-dudy-svr.bin')
# np.savetxt('min-max-svr.txt', [dudy_max, dudy_min])
# # plt.savefig('scatter-cmu-vs-dudy-svr-and-test.png',bbox_inches='tight')

