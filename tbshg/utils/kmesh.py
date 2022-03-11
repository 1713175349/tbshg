from ..hamiltonian import tbHamiltonian
import numpy as np  

class KPOINTS_mesh(object):
    def __init__(self,H:tbHamiltonian,nkx:np.int32,nky:np.int32,nkz:np.int32,shift=[0,0,0]):
        self.nkx=nkx
        self.nky=nky
        self.nkz=nkz
        self.shift=np.array(shift)
        self.kx=np.linspace(-0.5,0.5,nkx+1)[:-1]
        self.ky=np.linspace(-0.5,0.5,nky+1)[:-1]
        self.kz=np.linspace(-0.5,0.5,nkz+1)[:-1]
        Kx,Ky,Kz=np.meshgrid(self.kx,self.ky,self.kz)
        kx=Kx.reshape(-1)
        ky=Ky.reshape(-1)
        kz=Kz.reshape(-1)

        self.k_rec=np.vstack((kx,ky,kz)).T+np.broadcast_to(self.shift,(len(kx),3))

        self.kmesh = np.dot(self.k_rec,H.Rlattice)

    def get_kmesh(self):
        return self.kmesh
