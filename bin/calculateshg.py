import sys
from os.path import dirname
import os
from tbshg import tbshg
import numpy as np

from mpi4py import MPI
import pickle
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numproc = comm.Get_size()

test2 = tbshg
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



def readconfig(fn:str):
    with open(fn,"r") as fp:
        buf=""
        out={}
        while 1:
            line = fp.readline()
            if line == "":
                break
            if line.strip() == "" or line.strip()[0]=="#":
                continue
            if "#" in line:
                line=line[:line.index("#")]

            buf += line.strip()
            if line.strip()[-1]=="\\":
                buf=buf[:-1]
                continue
            k,v=[j.strip() for j in buf.split(":")]
            out[k]=v
            buf=""
    return out
            
confs = readconfig(sys.argv[1])

solver=test2.solveshg()
readfile(solver.H1,confs["datafilepath"])
# readfile(solver.H1,"/mnt/c/Users/dell/Desktop/bn-w/wannier90")

solver.H1.bandgapadd=float(confs["bandgapadd"])

solver.ksi=float(confs["ksi"])
nkx,nky,nkz=[int(i) for i in confs["nkx,nky,nkz"].split(",")]
nkxs,nkys,nkzs=[int(i) for i in confs["nkxs,nkys,nkzs"].split(",")]
thickness=6
epsilon0=0.0552

noccu = int(confs["nocc"])


hv=np.linspace(*[int(i) for i in confs["hvrange"].split(",")])

directs=[[int(j) for j in i.split()] for i in confs["directs"].split(",")]

iswriteperk = bool(int(confs["writeperk"]))

# raise ValueError("a 必须是数字")

kx=np.linspace(-0.5,0.5,nkx+1)[:-1]+nkxs
ky=np.linspace(-0.5,0.5,nky+1)[:-1]+nkys
kz=np.linspace(-0.5,0.5,nkz+1)[:-1]+nkzs
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

if rank == 0:
    with open("kmesh.pkl","wb") as fp:
        pickle.dump(kmesh,fp)
    with open("hv.pkl","wb") as fp:
        pickle.dump(hv,fp)


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


comm.Barrier()
shg_directs=[]
shgdk=[]
for j in range(len(directs)):
    shg_k=solver.getshg(j)
    if rank!=0:
        comm.send(shg_k[:,myslice],dest=0,tag=rank*100+j+10000)
        print("send success:",rank,"shg")
    else:
        for i in range(1,numproc):
            print("recev:",i)
            shg_k[:,everykslice[i]]=comm.recv(source=i,tag=i*100+j+10000)

    comm.Barrier()
    if rank != 0:
        continue
    shg_final=np.zeros(shg_k.shape[0],dtype=np.complex128)
    for i in range(shg_k.shape[0]):
        shg_final[i]=np.dot(shg_k[i],kweight)/Vcell/epsilon0
    shg_final = shg_final*lattice[2][2]/thickness
    shg_directs.append(shg_final)
    shgdk.append(shg_k)
    shg=np.sum(shg_k,axis=1)






if rank == 0:
    with open("shg_directs.pkl","wb") as fp:
        pickle.dump(shg_directs,fp)
    if iswriteperk:
        with open("shg_directsperk.pkl","wb") as fp:
            pickle.dump(shgdk,fp)

shgdk=[]
# ita
comm.Barrier()
shg_directs=[]
itadk=[]
for j in range(len(directs)):
    shg_k=solver.getita(j)
    if rank!=0:
        comm.send(shg_k[:,myslice],dest=0,tag=rank*100+j+20000)
        print("send success")
    else:
        for i in range(1,numproc):
            shg_k[:,everykslice[i]]=comm.recv(source=i,tag=i*100+j+20000)

    comm.Barrier()
    if rank != 0:
        continue
    shg_final=np.zeros(shg_k.shape[0],dtype=np.complex128)
    for i in range(shg_k.shape[0]):
        shg_final[i]=np.dot(shg_k[i],kweight)/Vcell/epsilon0
    shg_final = shg_final*lattice[2][2]/thickness
    shg_directs.append(shg_final)
    itadk.append(shg_k)
    shg=np.sum(shg_k,axis=1)




if rank == 0:
    with open("ita_directs.pkl","wb") as fp:
        pickle.dump(shg_directs,fp)
    if iswriteperk:
        with open("ita_directsperk.pkl","wb") as fp:
            pickle.dump(itadk,fp)

itadk=[]


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
    if rank != 0:
        continue
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
    if iswriteperk:
        with open("chi_directsperk.pkl","wb") as fp:
            pickle.dump(chidk,fp)

chidk=[]

#sigma
comm.Barrier()
shg_directs=[]
sigmadk=[]
for j in range(len(directs)):
    shg_k=solver.getsigma(j)
    if rank!=0:
        comm.send(shg_k[:,myslice],dest=0,tag=rank*100+j)
        print("send success")
    else:
        for i in range(1,numproc):
            shg_k[:,everykslice[i]]=comm.recv(source=i,tag=i*100+j)

    comm.Barrier()
    if rank != 0:
        continue
    shg_final=np.zeros(shg_k.shape[0],dtype=np.complex128)
    for i in range(shg_k.shape[0]):
        shg_final[i]=np.dot(shg_k[i],kweight)/Vcell/epsilon0
    shg_final = shg_final*lattice[2][2]/thickness
    shg_directs.append(shg_final)
    sigmadk.append(shg_k)
    shg=np.sum(shg_k,axis=1)




if rank == 0:
    with open("sigma_directs.pkl","wb") as fp:
        pickle.dump(shg_directs,fp)
    if iswriteperk:
        with open("sigma_directsperk.pkl","wb") as fp:
            pickle.dump(sigmadk,fp)

sigmadk=[]
comm.Barrier()