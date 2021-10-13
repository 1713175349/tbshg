import sys
from os.path import dirname
import os
import test2
import numpy as np

# H1=test2.Hamiltoniank()

#print(H1.nowk)

# lattice=np.array([[],
#                   [],
#                   []])

def readfile(H1,fn:str):
    fp=open(fn+"_hr.dat","r")
    fp.readline()
    wcentnum=int(fp.readline().strip())
    overlapnum=int(fp.readline().strip())
    wlinenum = int(np.ceil(overlapnum/15))
    wtR=[]
    for i in range(wlinenum):
        wtR.extend([float(x) for x in fp.readline().split()])
    H1.wcentnum=wcentnum
    H1.overlapnum=overlapnum
    H1.wtR=wtR
    H1.setup()
    parameterlinenum = wcentnum*wcentnum*overlapnum
    Hr=np.zeros((overlapnum,wcentnum,wcentnum),dtype=np.complex128)
    ln=0
    R=np.zeros((overlapnum,3))
    Rr=np.zeros((overlapnum,3))
    for i in range(parameterlinenum):
        l = fp.readline()
        a=l.split()
        #这里可能要注意顺序
        Hr[ln//(wcentnum*wcentnum),ln%wcentnum,ln%(wcentnum*wcentnum)//wcentnum]=complex(float(a[-2]),float(a[-1]))
        if(ln%(wcentnum*wcentnum) == 0):
            R[ln//(wcentnum*wcentnum)] = [float(a[0]),float(a[1]),float(a[2])]
        ln += 1
    H1.R=R
    # for i in range(overlapnum):
    #     Hr[i]=(Hr[i]+Hr[overlapnum-1-i].transpose().conj())/2
    H1.Hr=Hr
    fp.close()
    lattice=np.zeros((3,3))
    fp=open(os.path.join(os.path.dirname(fn),"POSCAR"),"r")
    fp.readline()
    weight=float(fp.readline().strip())
    lattice=np.zeros((3,3))
    lattice[0]=[float(ad)*weight for ad in fp.readline().split()]
    lattice[1]=[float(ad)*weight for ad in fp.readline().split()]
    lattice[2]=[float(ad)*weight for ad in fp.readline().split()]
    H1.lat=lattice
    for i in range(overlapnum):
        Rr[i]=np.dot(R[i],lattice)
    H1.Rr=Rr
    fp.close()
    fp=open(fn+"_centres.xyz","r")
    fp.readline()
    fp.readline()
    wcent=np.zeros((wcentnum,3))
    for i in range(wcentnum):
        wcent[i]=[float(ad)*weight for ad in fp.readline().split()[1:]]
    H1.wcent=wcent
    fp.close()



solver=test2.solveshg()
readfile(solver.H1,"/mnt/c/Users/dell/Desktop/cc/wannier90")
# readfile(solver.H1,"/mnt/c/Users/dell/Desktop/bn-w/wannier90")

solver.H1.bandgapadd=0.0

solver.ksi=0.02
nkx=168
nky=168
nkz=1
thickness=6
epsilon0=0.0552

noccu = 7


hv=np.linspace(0,5,201)

directs=[
        # [0,0,0],
        # [0,0,1],
        # [0,0,2],
        # [0,1,1],
        # [0,1,2],
        # [0,2,2],

        # [1,0,0],
        [1,0,1],
        # [1,0,2],
        # # [1,1,0],
        # [1,1,1],
        # [1,1,2],
        # # [1,2,0],
        # # [1,2,1],
        # [1,2,2],

        # [2,0,0],
        # [2,0,1],
        # [2,0,2],
        # # [2,1,0],
        # [2,1,1],
        # [2,1,2],
        # # [2,2,0],
        # # [2,2,1],
        # [2,2,2],
        ]


kx=np.linspace(-0.5,0.5,nkx+1)[:-1]#+1/2/nkx
ky=np.linspace(-0.5,0.5,nky+1)[:-1]#+1/2/nky
kz=np.linspace(-0.5,0.5,nkz+1)[:-1]+1/2/nkz
Kx,Ky,Kz=np.meshgrid(kx,ky,kz)
kx=Kx.reshape(-1)
ky=Ky.reshape(-1)
kz=Kz.reshape(-1)
# kx=[0]
# ky=[0]
# kz=[0.5]
Rlat=np.zeros((3,3))
lattice=solver.H1.lat
# lattice=np.zeros((3,3))
nkpts=len(kx)
kmesh=np.zeros((nkpts,3))
kweight=np.zeros(nkpts)
Rlat[0]=2*np.pi*np.cross(lattice[1],lattice[2])/np.linalg.det(lattice)
Rlat[1]=2*np.pi*np.cross(lattice[2],lattice[0])/np.linalg.det(lattice)
Rlat[2]=2*np.pi*np.cross(lattice[0],lattice[1])/np.linalg.det(lattice)
Vcell=np.linalg.det(lattice)
for i in range(nkpts):
    kmesh[i]=kx[i]*Rlat[0]+ky[i]*Rlat[1]+kz[i]*Rlat[1]
    kweight[i]=1
# print(kmesh)


# kmesh=np.zeros((nkpts,3))
# for i in range(nkpts):
#     kmesh[i]=[(np.random.rand()-0.5)*10000,(np.random.rand()-0.5)*10000,0]


kweight=kweight/nkpts




solver.nkpts=nkpts
solver.ndirects=len(directs)
solver.nhv=len(hv)
solver.setup()
solver.kmesh=kmesh
solver.kweight=kweight
solver.hv=hv
solver.directs=directs
#print("".join([["x","y","z"][i] for i in solver.directs[0]]))
myweight=np.zeros((solver.H1.wcentnum,))
for i in range(len(myweight)):
    if i<noccu:
        myweight[i] = 1
solver.H1.energyweight=myweight


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numproc = comm.Get_size()

everyk=int(np.ceil(nkpts/numproc))

everykslice=[]

up=0
for i in range(numproc):
    if up+everyk <= nkpts:
        everykslice.append(slice(up,up+everyk,1))
        up+=everyk
    else:
        everykslice.append(slice(up,nkpts,1))

if rank == 0:
    print(numproc,nkpts)

myslice=everykslice[rank]

for i in range(myslice.start,myslice.stop,myslice.step):
    print("proc:",rank,"  processing:",(i-myslice.start)/everyk)
    solver.updatek(i)
import pickle

comm.Barrier()
shg_directs=[]
for j in range(len(directs)):
    shg_k=solver.getshg(j)
    if rank!=0:
        comm.send(shg_k[:,myslice],dest=0,tag=rank*100+j)
        print("send success")
    else:
        for i in range(1,numproc):
            shg_k[:,everykslice[i]]=comm.recv(source=i,tag=i*100+j)

    comm.Barrier()
    shg_final=np.zeros(shg_k.shape[0],dtype=np.complex128)
    for i in range(shg_k.shape[0]):
        shg_final[i]=np.dot(shg_k[i],kweight)/Vcell/epsilon0
    shg_final = shg_final*lattice[2][2]/thickness
    shg_directs.append(shg_final)
    shg=np.sum(shg_k,axis=1)

 




if rank == 0:
    with open("shg_directs.pkl","wb") as fp:
        pickle.dump(shg_directs,fp)

# ita
comm.Barrier()
shg_directs=[]
for j in range(len(directs)):
    shg_k=solver.getita(j)
    if rank!=0:
        comm.send(shg_k[:,myslice],dest=0,tag=rank*100+j)
        print("send success")
    else:
        for i in range(1,numproc):
            shg_k[:,everykslice[i]]=comm.recv(source=i,tag=i*100+j)

    comm.Barrier()
    shg_final=np.zeros(shg_k.shape[0],dtype=np.complex128)
    for i in range(shg_k.shape[0]):
        shg_final[i]=np.dot(shg_k[i],kweight)/Vcell/epsilon0
    shg_final = shg_final*lattice[2][2]/thickness
    shg_directs.append(shg_final)
    shg=np.sum(shg_k,axis=1)

 


if rank == 0:
    with open("ita_directs.pkl","wb") as fp:
        pickle.dump(shg_directs,fp)



#chi
comm.Barrier()
shg_directs=[]
chidk=[]
for j in range(len(directs)):
    shg_k=solver.getchi(j)
    if rank!=0:
        comm.send(shg_k[:,myslice],dest=0,tag=rank*100+j)
        print("send success")
    else:
        for i in range(1,numproc):
            shg_k[:,everykslice[i]]=comm.recv(source=i,tag=i*100+j)

    comm.Barrier()
    shg_final=np.zeros(shg_k.shape[0],dtype=np.complex128)
    for i in range(shg_k.shape[0]):
        shg_final[i]=np.dot(shg_k[i],kweight)/Vcell/epsilon0
    shg_final = shg_final*lattice[2][2]/thickness
    shg_directs.append(shg_final)
    chidk.append(shg_k)
    shg=np.sum(shg_k,axis=1)

 


if rank == 0:
    with open("chi_directs.pkl","wb") as fp:
        pickle.dump(shg_directs,fp)
    with open("chi_directsperk.pkl","wb") as fp:
        pickle.dump(chidk,fp)


#sigma
comm.Barrier()
shg_directs=[]
for j in range(len(directs)):
    shg_k=solver.getsigma(j)
    if rank!=0:
        comm.send(shg_k[:,myslice],dest=0,tag=rank*100+j)
        print("send success")
    else:
        for i in range(1,numproc):
            shg_k[:,everykslice[i]]=comm.recv(source=i,tag=i*100+j)

    comm.Barrier()
    shg_final=np.zeros(shg_k.shape[0],dtype=np.complex128)
    for i in range(shg_k.shape[0]):
        shg_final[i]=np.dot(shg_k[i],kweight)/Vcell/epsilon0
    shg_final = shg_final*lattice[2][2]/thickness
    shg_directs.append(shg_final)
    shg=np.sum(shg_k,axis=1)

 


if rank == 0:
    with open("sigma_directs.pkl","wb") as fp:
        pickle.dump(shg_directs,fp)

