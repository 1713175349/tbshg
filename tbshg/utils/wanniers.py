import numpy as np
import os
import re
import sys
from tbshg import tbshg_core


def readwannier_H(confs):
    fn=confs["datafilepath"]
    H1=tbshg_core.Hamiltoniank()
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

    #设置剪刀差
    H1.bandgapadd=float(confs["bandgapadd"])

    #设置占据数
    noccu = int(confs["nocc"])
    myweight=np.zeros((H1.wcentnum,))
    for i in range(len(myweight)):
        if i<noccu:
            myweight[i] = 1
    H1.energyweight=myweight

    return H1


def readwannierfolder_H(directory:str="./",prefix="wannier90",cutoff:float=-1,bandgapadd:float=0,occnum:int=0)->tbshg_core.Hamiltoniank:
    H1=tbshg_core.Hamiltoniank()
    fn=os.path.join(directory,prefix)
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

    #厄米化参数
    # for i in range(overlapnum):
    #     Hr[i]=(Hr[i]+Hr[overlapnum-1-i].transpose().conj())/2

    
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
    # 跳过前两行
    fp.readline()
    fp.readline()
    wcent=np.zeros((wcentnum,3))
    for i in range(wcentnum):
        wcent[i]=[float(ad)*weight for ad in fp.readline().split()[1:]]
    H1.wcent=wcent
    fp.close()

    #设置剪刀差
    H1.bandgapadd=float(bandgapadd)

    #设置占据数
    noccu = int(occnum)
    myweight=np.zeros((H1.wcentnum,))
    for i in range(len(myweight)):
        if i<noccu:
            myweight[i] = 1
    H1.energyweight=myweight

    if cutoff>0:
        for Ri in range(overlapnum):
            for i in range(wcentnum):
                for j in range(wcentnum):
                    if np.linalg.norm(Rr[Ri]+wcent[j]-wcent[i])>cutoff:
                        Hr[Ri][i][j]=0

    H1.Hr=Hr

    return H1


def generate_wcentres_by_wout(woutfile:str):
    with open(woutfile,"r") as fp:
        cents=[]
        while True:
            line=fp.readline()
            if "Final State" in line.strip():
                while True:
                    line=fp.readline()
                    if "Sum" in line.strip():
                        break
                    #match coordinates in brackets
                    coor=[float(i) for i in re.findall(r"\(([^\]]*)\)",line)[0].split(",")]
                    cents.append(coor)
                break
    return cents

def write_wcentres(cents:list,fn:str="wannier90_centres.xyz"):
    with open(fn,"w") as fp:
        fp.write("{}\n".format(len(cents)))
        fp.write("\n")
        for i in range(len(cents)):
           fp.write("X  {}  {}  {}\n".format(cents[i][0],cents[i][1],cents[i][2]))