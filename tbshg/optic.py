from tokenize import Double
from .utils.constant import epsilon0
from .hamiltonian import tbHamiltonian,Hamiltonian
from . import tbshg_core
from .utils.wanniers import readwannierfolder_H
from .utils.configs import readconfig
from .utils.kmesh import KPOINTS_mesh
import os
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

class optproperty(object):
    """
    用于计算各种光学性质
    """
    def __init__(self,H:tbHamiltonian,ksi:float=0.02):
        self.H0=H
        self.solver=tbshg_core.solveopt(self.H0.H,ksi)
        self.config=None
    
    @property
    def Vcell(self):
        return self.H0.Vcell
        

    @classmethod
    def fromconfig(cls,fn:str):
        config=readconfig(fn)
        path0=config["datafilepath"]
        dirn=os.path.split(path0)[0]
        seedname=os.path.split(path0)[1]
        H=tbHamiltonian.from_wannier_dir(directory=dirn,prefix=seedname,
                                        bandgapadd=float(config["bandgapadd"] if config.get("bandgapadd") else 0.0),
                                        cutoff=float(config["cutoff"] if config.get("cutoff") else -1),
                                        fermi=float(config["fermi"])
                                        )
        out = cls(H,float(config["ksi"]))
        out.config=config
        out.solver.setup_mc() # 初始化蒙特卡洛方法
        out.solver.max_occ=float(config.get("max_occ") if config.get("max_occ") else 2.0)
        return out
    
    @classmethod
    def fromconfig_mm(cls,fn:str):
        config=readconfig(fn)
        path0=config["datafilepath"]
        dirn=os.path.split(path0)[0]
        seedname=os.path.split(path0)[1]
        exclude_bands=5
        if config.get("exclude_bands"):
            exclude_bands=int(config["exclude_bands"])
        H=tbHamiltonian.from_mm_binary(dirn,exclude_bands=exclude_bands)
        out = cls(H,float(config["ksi"]))
        out.config=config
        out.solver.setup_mc() # 初始化蒙特卡洛方法
        out.solver.max_occ=float(config.get("max_occ") if config.get("max_occ") else 2.0)
        return out
    
    def get_kmesh(self):
        # TODO: 计划构建一个新函数，将对称性引入，提高速度
        nkx,nky,nkz=[int(i) for i in self.config["nkx,nky,nkz"].split(",")]
        nkxs,nkys,nkzs=[float(i) for i in self.config["nkxs,nkys,nkzs"].split(",")]
        
        return KPOINTS_mesh(self.H0,nkx,nky,nkz,shift=[nkxs,nkys,nkzs])

    def get_hv(self):
        hv=np.linspace(*[int(i) for i in self.config["hvrange"].split(",")])
        return hv

    def get_directindices(self):
        directs=[[int(j) for j in i.split()] for i in self.config["directs"].split(",")]
        return directs

    def solve_shg(self,kmesh:np.ndarray,hv:np.ndarray,directindices:np.ndarray,savek:bool=False,show_progress:bool=False):
        """
        计算shg,如果savek会返回每个k点的shg值,返回值的单位为nm/V
        """
        shg_k=None
        nkpts=len(kmesh)
        if savek:
            shg_k=np.zeros((len(kmesh),len(directindices),len(kmesh)),dtype=np.complex128)
        shg_i=np.zeros((len(directindices),len(hv)),dtype=np.complex128)
        for i in range(len(kmesh)):
            if show_progress:
                print("rank: ",comm.rank,", kpoint: ",i,"/",nkpts,"progress: ",i/nkpts,flush=True)
            self.H0.solve_one_kpoint(kmesh[i])
            kshg = self.solver.get_shg_f(kmesh[i],hv,directindices)
            shg_i+=kshg
            if savek:
                shg_k[i]=kshg
        if savek:
            return shg_i/self.Vcell/epsilon0/nkpts,shg_k/self.Vcell/epsilon0
        else:
            return shg_i/self.Vcell/epsilon0/nkpts

    def solve_shg_mc(self,kmesh:np.ndarray,hv:np.ndarray,directindices:np.ndarray,savek:bool=False,show_progress:bool=False,times:int=10000):
        """
        计算shg,如果savek会返回每个k点的shg值,返回值的单位为nm/V
        此方法是用蒙特卡洛方法计算的，当采样达到200万次时，对于13条能带的单点，数值基本可以稳定
        """
        shg_k=None
        nkpts=len(kmesh)
        if savek:
            shg_k=np.zeros((len(kmesh),len(directindices),len(kmesh)),dtype=np.complex128)
        shg_i=np.zeros((len(directindices),len(hv)),dtype=np.complex128)
        for i in range(len(kmesh)):
            if show_progress:
                print("rank: ",comm.rank,", kpoint: ",i,"/",nkpts,"progress: ",i/nkpts,flush=True)
            self.H0.solve_one_kpoint(kmesh[i])
            kshg = self.solver.get_shg_mc(kmesh[i],hv,directindices,times)
            shg_i+=kshg
            if savek:
                shg_k[i]=kshg
        if savek:
            return shg_i/self.Vcell/epsilon0/nkpts,shg_k/self.Vcell/epsilon0
        else:
            return shg_i/self.Vcell/epsilon0/nkpts


    def solve_linechi(self,kmesh:np.ndarray,hv:np.ndarray,directindices:np.ndarray,savek:bool=False,show_progress:bool=False):
        """
        计算线性极化率
        """
        nkpts=len(kmesh)
        linechi_k=None
        if savek:
            linechi_k=np.zeros((len(kmesh),len(directindices),len(kmesh)),dtype=np.complex128)
        linechi_i=np.zeros((len(directindices),len(hv)),dtype=np.complex128)
        for i in range(len(kmesh)):
            if show_progress:
                print("rank: ",comm.rank,", kpoint: ",i,"/",nkpts,"progress: ",i/nkpts,flush=True)
            self.H0.solve_one_kpoint(kmesh[i])
            kchi = self.solver.get_linechi(kmesh[i],hv,directindices)
            linechi_i+=kchi

        return linechi_i/self.Vcell/epsilon0/nkpts

    def solve_shg_from_config(self):
        """
        计算shg从配置文件的设定
        """
        kmesh=self.get_kmesh()
        hv=self.get_hv()
        directindices=self.get_directindices()
        return parallel_kmesh(self.solve_shg,kmesh,hv,directindices)
    
    def solve_shg_mc_from_config(self,times:int=10000):
        """
        计算shg从配置文件的设定,使用蒙特卡洛方法
        """
        kmesh=self.get_kmesh()
        hv=self.get_hv()
        directindices=self.get_directindices()
        return parallel_kmesh(self.solve_shg_mc,kmesh,hv,directindices,times=times)

    def solve_linechi_from_config(self):
        """
        计算线性极化率从配置文件的设定
        """
        kmesh=self.get_kmesh()
        hv=self.get_hv()
        directindices=[[int(j) for j in i.split()] for i in self.config["directslinear"].split(",")]
        return parallel_kmesh(self.solve_linechi,kmesh,hv,directindices)


def parallel_kmesh(solverfunc,Kmesh:KPOINTS_mesh,hv:np.ndarray,directindices:np.ndarray,**kwargs):
    """
    并行计算kmesh
    """
    
    kmesh_splited = np.array_split(Kmesh.kmesh,comm.size)
    result = solverfunc(kmesh_splited[comm.rank],hv=hv,directindices=directindices,show_progress=True,**kwargs)

    if comm.rank==0:
        result_i=np.zeros((comm.size,len(directindices),len(hv)),dtype=np.complex128)
        result_i[0]=result
        for i in range(1,comm.size):
            if len(kmesh_splited[i])>0:
                result_i[i]=comm.recv(source=i,tag=i)
        
    else:
        if len(kmesh_splited[comm.rank])>0:
            comm.send(result,dest=0,tag=comm.rank)
    comm.Barrier()
    if comm.rank==0:
        weights=np.array([i.shape[0] for i in kmesh_splited])
        result0=0
        for i in range(comm.size):
            result0+=result_i[i]*weights[i]

        return result0/np.sum(weights)


