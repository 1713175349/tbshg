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
        self.is_mm_binary = False
        self.fermi=fermi #费米能级，用于更新占据数
    
    def updateHr(self,R,Hr):
        self.H.R=R
        self.H.Rr=np.dot(R,self.H.lat)
        self.H.Hr=Hr
        self.H.setup()
    
    def solve_one_point(self,k:np.ndarray)->np.ndarray:
        return self.solve_one_kpoint(k)

    def solve_one_kpoint(self,kpoint=[0,0,0]):
        if self.is_mm_binary:
            self.H.kvector_now=kpoint
            kindex=np.argmin(np.linalg.norm(self.kpoints-kpoint,axis=1))
            if np.linalg.norm(self.kpoints-kpoint,axis=1)[kindex]>1e-5:
                raise ValueError("kpoint not found")
            self.H.energy=self.eigval[kindex]
            self.H.energyweight=self.occu[kindex]
            px,py,pz=read_mm_binary(self.mm_binary_path,self.nbands,kindex)
            self.H.setvnm(0,px)
            self.H.setvnm(1,py)
            self.H.setvnm(2,pz)
            self.H.updaternm()
            return None

        self.H.runonekpoint(kpoint)
        ev = self.H.energy

        energyweight = np.zeros(ev.shape)
        T=0.00001
        for i in range(ev.shape[0]):
            energyweight[i] = (1 if ev[i]<self.fermi else 0)
        self.H.energyweight = energyweight
        return ev

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

    @classmethod
    def from_mm_binary(cls,filedirs:str,exclude_bands:int=5,fermi:float=0):
        from tbshg import tbshg_core
        import os
        eigvalfile=os.path.join(filedirs,"EIGENVAL")
        with open(eigvalfile,"r") as f:
            lines=f.readline()
            lines=f.readline()
            lines=f.readline()
            lines=f.readline()
            lines=f.readline()
            lines=f.readline()
            # print(lines,lines.split())
            nelectron,nkpoints,nbands=[int(i) for i in  lines.split()]
            eigvals=[]
            kpoints=[]
            kweights=[]
            occunum=[]
            for i in range(nkpoints):
                lines=f.readline()
                lines=f.readline()
                kpoints.append([float(i) for i in lines.split()[:3]])
                kweights.append(float(lines.split()[3]))
                eigvals.append([])
                occunum.append([])
                for j in range(nbands):
                    lines=f.readline()
                    eigvals[i].append(float(lines.split()[1]))
                    occunum[i].append(float(lines.split()[2]))

        H=tbshg_core.Hamiltoniank()
        H.wcentnum=nbands-exclude_bands
        H.overlapnum=0
        H.setup()

        lattice=np.zeros((3,3))
        fp=open(os.path.join(filedirs,"POSCAR"),"r")
        fp.readline()
        weight=float(fp.readline().strip())
        lattice=np.zeros((3,3))
        lattice[0]=[float(ad)*weight for ad in fp.readline().split()]
        lattice[1]=[float(ad)*weight for ad in fp.readline().split()]
        lattice[2]=[float(ad)*weight for ad in fp.readline().split()]
        H.lat=lattice

        out=cls(H,fermi)
        out.is_mm_binary = True
        out.nbands = nbands
        out.mm_binary_path = os.path.join(filedirs,"mm_binary")
        
        out.eigval=np.array(eigvals)[:,:(nbands-exclude_bands)]
        out.kpoints=np.array(kpoints)
        out.occu=np.array(occunum)[:,:(nbands-exclude_bands)]
        out.weights=np.array(kweights)

        return out
        
def make_kpath(npoints:int,kpath:np.ndarray):
    """
    生成kpath
    """
    import numpy as np
    kpath=np.array(kpath)
    kpoints=[]
    kpoints.append(kpath[0])
    for i in range(kpath.shape[0]-1):
        kpoints.extend(np.linspace(kpath[i],kpath[i+1],npoints+1).tolist()[:-1])
    return np.array(kpoints)

def read_mm_binary(filename:str,nbands:int,ikpoints:int,nspin:int=1):
    """
    读取mm_binary文件
    """
    import numpy as np
    import struct
    with open(filename,"rb") as f:
        data=f.read()
        # offset nbands ikpoints nspin np.complex128
        f.seek(nbands*nbands*ikpoints*nspin*3*16)
        #read px matrix
        #px=np.zeros((nbands,nbands),dtype=np.complex128)
        """
        Quantity[1, "\[HBar]"]^2/Quantity[1, "ElectronMass"]/
  Quantity[1, "eV"]/Quantity[1, "Angstroms"]^2
        """
        px=np.fromfile(f,dtype=np.complex128,count=nbands*nbands).reshape(nbands,nbands)*(-1j*7.61996423)
        #read py matrix
        #py=np.zeros((nbands,nbands),dtype=np.complex128)
        py=np.fromfile(f,dtype=np.complex128,count=nbands*nbands).reshape(nbands,nbands)*(-1j*7.61996423)
        #read pz matrix
        #pz=np.zeros((nbands,nbands),dtype=np.complex128)
        pz=np.fromfile(f,dtype=np.complex128,count=nbands*nbands).reshape(nbands,nbands)*(-1j*7.61996423)
        return px,py,pz