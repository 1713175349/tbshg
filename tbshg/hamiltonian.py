import numpy as np
from tbshg import tbshg_core
from .utils.wanniers import readwannierfolder_H
class Hamiltonian(object):
    H=None
    def solve_one_point(self,k:np.ndarray)->np.ndarray:
        raise NotImplementedError

class tbHamiltonian(Hamiltonian):
    """"""
    def __init__(self,H:tbshg_core.Hamiltoniank=None,fermi:float=0):
        if H is None:
            self.H=tbshg_core.Hamiltoniank()
        else:
            self.H=H
        self.fermi=fermi #费米能级，用于更新占据数
    
    def updateHr(self,R,Hr):
        self.H.R=R
        self.H.Rr=np.dot(R,self.H.lat)
        self.H.Hr=Hr
        self.H.setup()
    
    def solve_one_kpoint(self,kpoint=[0,0,0]):
        self.H.runonekpoint(kpoint)
        ev = self.H.energy

        energyweight = np.zeros(ev.shape)
        T=0.00001
        for i in range(ev.shape[0]):
            energyweight[i] = (1 if ev[i]<self.fermi else 0)
        self.H.energyweight = energyweight

    @classmethod
    def from_wannier_dir(cls,directory:str="./",prefix:str="wannier90",bandgapadd:float=0,cutoff:float=0,fermi:float=0,occnum:int=0):
        
        H=readwannierfolder_H(directory,prefix,cutoff,bandgapadd,occnum=occnum)
        fermi=fermi

        return cls(H,fermi)


    @property
    def lattice(self):
        return self.H.lat

    @property
    def Vcell(self):
        return np.linalg.det(self.H.lat)

    @property
    def Rlattice(self):
        Rlat=np.zeros((3,3))
        lattice=self.H.lat
        # lattice=np.zeros((3,3))
        Rlat[0]=2*np.pi*np.cross(lattice[1],lattice[2])/np.linalg.det(lattice)
        Rlat[1]=2*np.pi*np.cross(lattice[2],lattice[0])/np.linalg.det(lattice)
        Rlat[2]=2*np.pi*np.cross(lattice[0],lattice[1])/np.linalg.det(lattice)
        return Rlat
        
    