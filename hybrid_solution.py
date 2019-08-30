import numpy as np
import matplotlib.pyplot as plt
from math import *
import scipy.sparse as sp

L = 50  # length of single straight root (cm)
a = 0.2  # radius (cm)
kz = 4.32e-2  # axial conductivity (cm^3 / day)
kr = 1.728e-4  # radial conductivity (1 / day)
p_s = -200  # static soil matric potiential (cm)
p0 = -1000  # dirichlet bc at top (cm)

# Exact solution (Daniel)
c = 2 * a * pi * kr / kz
p_r = lambda z: p_s + d[0] * exp(sqrt(c) * z) + d[1] * exp(-sqrt(c) * z)

# Boundary conditions
AA = np.array([[1, 1], [sqrt(c) * exp(sqrt(c) * (-L)), -sqrt(c) * exp(-sqrt(c) * (-L))] ])
bb = np.array([p0 - p_s, 0])  # <--- put 0 instead of -1 if you want neglect gravitation
d = np.linalg.solve(AA, bb)

# Exact solution (Félicien)
tau = sqrt(2 * pi * a * kr / kz)  # tau = sqrt(c)
psi_x = lambda z: p_s + (p0 - p_s) / cosh(tau * L) * cosh(z * tau)  # in z

za_ = np.linspace(0, -L, 101)
pr = list(map(p_r, za_))

zb_ = np.linspace(0, L, 101)
psix = list(map(psi_x,zb_))

# Hybrid solution
# 1) Root system construction
lseg = 0.5
Nseg = round(L/lseg)
seg_num=np.arange(1,Nseg+1)
radius=np.zeros(Nseg)
kr_seg=np.zeros(Nseg)
kx_seg=np.zeros(Nseg)
l=np.zeros(Nseg)
z=np.zeros(Nseg)
prev=np.zeros(Nseg,dtype=np.int8)

for i in (seg_num): 
    radius[i-1]=a
    l[i-1]=lseg
    kr_seg[i-1]=kr
    kx_seg[i-1]=kz
    prev[i-1]=i-1

kappa=(2*pi*radius*kr_seg*kx_seg)**0.5
tau=(2*pi*radius*kr_seg/kx_seg)**0.5  


j=seg_num-1;
i=prev-1;

rows=i+1;
columns=i+1;
values=-kappa[j]/np.sinh(tau[j]*l[j])-kappa[j]*np.tanh(tau[j]*l[j]/2)

rows=np.concatenate([rows,j+1])
columns=np.concatenate([columns,i+1])
values=np.concatenate([values,kappa[j]/np.sinh(tau[j]*l[j])])

rows=np.concatenate([rows,i+1])
columns=np.concatenate([columns,j+1])
values=np.concatenate([values,-kappa[j]*np.tanh(tau[j]*l[j]/2)+kappa[j]/np.tanh(tau[j]*l[j])])
                       
rows=np.concatenate([rows,j+1])
columns=np.concatenate([columns,j+1])
values=np.concatenate([values,-kappa[j]/np.tanh(tau[j]*l[j])])

A = sp.coo_matrix((values, (rows, columns)), shape=(Nseg+1, Nseg+1))

def dropcols_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()

def droprows_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.row, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.row -= idx_to_drop.searchsorted(C.row)    # decrement column indices
    C._shape = (C.shape[0]- len(idx_to_drop), C.shape[1] )
    return C.tocsr()

temp=dropcols_coo(A,0)
a=droprows_coo(temp,0)

prev_collar=np.in1d(prev, 0)

j=seg_num-1;
i=prev-1;

rows=i+1;
columns=np.ones((Nseg))-1
Psi_sr=(columns+1)*-np.abs(p_s)

values=-Psi_sr[j]*kappa[j]*np.tanh(tau[j]*l[j]/2)

rows=np.concatenate([rows,j+1])
columns=np.concatenate([columns,np.ones(Nseg)-1])
values=np.concatenate([values,-Psi_sr[j]*kappa[j]*np.tanh(tau[j]*l[j]/2)])

B = sp.coo_matrix((values, (rows, columns)), shape=(Nseg+1,1))
b=droprows_coo(B,0)

b[prev_collar]=b[prev_collar]-p0*kappa[prev_collar]/np.sinh(tau[prev_collar]*l[prev_collar])

X=sp.linalg.spsolve(a,b)
Psi_basal=X
prev_temp=prev-1
prev_temp[prev_collar]=0;
Psi_proximal=Psi_basal[prev_temp]
Jr=2*kappa*np.tanh(tau*l/2)*(Psi_sr-(Psi_proximal+Psi_basal)/2)
z_hybrid=np.linspace(-0.5,-50,100)

y = np.linspace(0,-50,10)

# Plot results
fig = plt.figure(figsize = (7, 7))

plt.plot(pr, za_, 'b--')
plt.plot(psix, zb_-L, 'g')
plt.plot(np.insert(Psi_basal,0,p0,axis=0), np.insert(z_hybrid,0,0,axis=0), 'r:')
plt.plot(list(map(p_r, y)),y,"r*")

plt.xlabel("Xylem pressure [cm]")
plt.ylabel("Depth [cm]")
plt.ylim((-50,-48))
plt.xlim((-247,-246))
plt.show()

Psi_final = np.insert(Psi_basal,0,p0,axis=0)
z_final = np.insert(z_hybrid,0,0,axis=0)

f=open("/home/femeunier/Documents/UCLouvain/benchmark/collaborative-comparison/root_water_flow/M31/hybrid",'w+')
np.savetxt("/home/femeunier/Documents/UCLouvain/benchmark/collaborative-comparison/root_water_flow/M31/hybrid", (z_final,Psi_final), fmt="%f", delimiter=',',)
f.close()