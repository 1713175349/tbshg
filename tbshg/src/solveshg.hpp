#pragma once
#include"Eigen/Dense"
#include<Eigen/Eigenvalues>
#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<pybind11/numpy.h>
#include "hamiltoniank.hpp"

// #define newversion

class solveshg
{
private:
    int nowkindex;//当前计算k点
public:
    // solveshg(/* args */);
    // ~solveshg();
    Hamiltoniank H1;//哈密顿
    int nkpts;
    int ndirects;
    int nhv;
    Eigen::MatrixX3d kmesh;//k点网格
    Eigen::VectorXd kweight;//k点权重
    Eigen::VectorXd energyweight;//能带权重
    Eigen::MatrixX3i directindice;//方向指标
    Eigen::VectorXd hv;//光子能量
    Eigen::Vector<Eigen::MatrixXcd,Eigen::Dynamic> shg;//(张量指标，hv,kvector)
    Eigen::Vector<Eigen::MatrixXcd,Eigen::Dynamic> chi;
    Eigen::Vector<Eigen::MatrixXcd,Eigen::Dynamic> ita;
    Eigen::Vector<Eigen::MatrixXcd,Eigen::Dynamic> sigma;
    double ksi=0.025;
    void setup();
    void updatechi();
    void updateita();
    void updatesigma();
    void updateshg();
    void updatebyvaspH(int ki){nowkindex=ki;updatechi();updateita();updatesigma();updateshg();}

    void updatek(int ki);
    Eigen::MatrixXcd getchi(int i){return chi(i);}
    Eigen::MatrixXcd getita(int i){return ita(i);}
    Eigen::MatrixXcd getsigma(int i){return sigma(i);}
    Eigen::MatrixXcd getshg(int i){return shg(i);}
};


void solveshg::setup(){
    kmesh.resize(nkpts,3);
    kweight.resize(nkpts);
    directindice.resize(ndirects,3);
    hv.resize(nhv);
    shg.resize(ndirects);
    chi.resize(ndirects);
    ita.resize(ndirects);
    sigma.resize(ndirects);
    for (int i = 0; i < ndirects; i++)
    {
        shg(i).resize(nhv,nkpts);
        chi(i).resize(nhv,nkpts);
        ita(i).resize(nhv,nkpts);
        sigma(i).resize(nhv,nkpts);
    }
    
}

void solveshg::updatechi(){
    complex<double> fac1=0;
    
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        int a=directindice(idirect,0);
        int b=directindice(idirect,1);
        int c=directindice(idirect,2);

        for (int ihv = 0; ihv < nhv; ihv++)
        {
            chi(idirect)(ihv,nowkindex)=0;
        }
        
//可能有问题
        for (int m = 0; m < H1.wcentnum; m++)
        {
            for (int l = 0; l < H1.wcentnum; l++)
            {
                for (int n = 0; n < H1.wcentnum; n++)
                {
                    fac1=0;
                   
                    double &&Eln=H1.energy(l)-H1.energy(n);
                    double &&Eml=H1.energy(m)-H1.energy(l);
                    double &&Emn=H1.energy(m)-H1.energy(n);
                    if (abs(Eln-Eml) > 0.0001){
                        
                        fac1=H1.rnm(a)(n,m)*(H1.rnm(b)(m,l)*H1.rnm(c)(l,n)+H1.rnm(c)(m,l)*H1.rnm(b)(l,n))/(Eln-Eml);
                    //     if(abs(fac1)>0.001 &&idirect==1){
                    //     std::cout<<n<<"-"<<l<<"-"<<m<<":"<<fac1<<std::endl;}
                    
                    }
                    // std::cout<<fac1<<std::endl;
                    for (int ihv = 0; ihv < nhv; ihv++)
                    {
                        complex<double> &&omega=hv(ihv)+complex<double>(0,1)*ksi;
                        complex<double> &&fac2=(H1.energyweight(m)-H1.energyweight(l))/(Eml-omega)
                                        +(H1.energyweight(l)-H1.energyweight(n))/(Eln-omega)
                                        +2*(H1.energyweight(n)-H1.energyweight(m))/(Emn-2.0*omega);
                        // if(n==9 && m==23 && l==19){
                        //     std::cout<<n<<l<<m<<"fac2:"<<omega<<fac2<<(H1.energyweight(m)-H1.energyweight(l))/(Eml-omega)<<std::endl;
                        // }
                        // if(abs(fac2)>0.001 && idirect ==1 && ihv==44 && nowkindex==3){
                        // std::cout<<n<<"-"<<l<<"-"<<m<<":"<<chi(idirect)(ihv,nowkindex)<<fac1<<fac2<<std::endl;}
                        // if (n==15 && m==0 && l==14 && idirect==1 && ihv==44 && nowkindex<4){
                        //     std::cout<<n<<"-"<<l<<"-"<<m<<"-"<<ihv<<":"<<fac1<<H1.rnm(a)(n,m)<<H1.rnm(b)(m,l)<<H1.rnm(c)(l,n)<<H1.rnm(c)(m,l)<<H1.rnm(b)(l,n)<<std::endl;
                        // }
                        chi(idirect)(ihv,nowkindex)+=0.5*fac1*fac2; //这里0.5把对称化部分除了
                    }
                    
                }
                
            }
            
        }
        
    }
    
}

void solveshg::updateita(){
    complex<double> fac1,fac2,fac3,fac12;
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        int a=directindice(idirect,0);
        int b=directindice(idirect,1);
        int c=directindice(idirect,2);

        for (int ihv = 0; ihv < nhv; ihv++)
        {
            ita(idirect)(ihv,nowkindex)=0;
        }
        
//可能有问题
        for (int ihv = 0; ihv < nhv; ihv++)
        {
            complex<double> &&omega=hv(ihv)+complex<double>(0,1)*ksi;
            for (int n = 0; n < H1.wcentnum; n++)
            {
                for (int m = 0; m < H1.wcentnum; m++)
                {
                    double && Emn=H1.energy(m)-H1.energy(n);
                    if (abs(H1.energyweight(n)-H1.energyweight(m))>0.0001){
                        fac12=-8.0*complex<double>(0,1)*(H1.energyweight(n)-H1.energyweight(m))*H1.rnm(a)(n,m)*(
                            (H1.vnm(b)(m,m)-H1.vnm(b)(n,n))*H1.rnm(c)(m,n)
                            +(H1.vnm(c)(m,m)-H1.vnm(c)(n,n))*H1.rnm(b)(m,n)
                        )/2.0/(Emn*Emn*(Emn-2.0*omega));
                        
                        ita(idirect)(ihv,nowkindex)+=fac12;
                    }

                    for (int l = 0; l < H1.wcentnum; l++)
                    {
                        double &&Eln=H1.energy(l)-H1.energy(n);
                        double &&Eml=H1.energy(m)-H1.energy(l);

                        fac1=0;
                        fac2=0;
                        fac3=0;
                        if(abs(H1.energyweight(n)-H1.energyweight(l))>0.0001){
                            fac1=Emn*(H1.energyweight(n)-H1.energyweight(l))*H1.rnm(a)(n,m)*(
                            H1.rnm(b)(m,l)*H1.rnm(c)(l,n)
                            +H1.rnm(c)(m,l)*H1.rnm(b)(l,n)
                        )/2.0/(Eln*Eln*(Eln-omega));
                        
                        }
                        #ifndef newversion
                            if(abs(H1.energyweight(l)-H1.energyweight(m))>0.0001){
                                fac2=Emn*(H1.energyweight(l)-H1.energyweight(m))*H1.rnm(a)(n,m)*(
                                H1.rnm(b)(m,l)*H1.rnm(c)(l,n)
                                +H1.rnm(c)(m,l)*H1.rnm(b)(l,n)
                            )/2.0/(Eml*Eml*(Eml-omega));
                            
                            }
                        #else
                            if(abs(H1.energyweight(l)-H1.energyweight(m))>0.0001){
                                fac2=-Emn*(H1.energyweight(l)-H1.energyweight(m))*H1.rnm(a)(n,m)*(
                                H1.rnm(b)(m,l)*H1.rnm(c)(l,n)
                                +H1.rnm(c)(m,l)*H1.rnm(b)(l,n)
                            )/2.0/(Eml*Eml*(Eml-omega));

                            }
                        #endif

                        #ifndef newversion
                            if(abs(H1.energyweight(n)-H1.energyweight(m))>0.0001){
                                fac3=-2*(H1.energyweight(n)-H1.energyweight(m))*H1.rnm(a)(n,m)*(
                                H1.rnm(b)(m,l)*H1.rnm(c)(l,n)
                                +H1.rnm(c)(m,l)*H1.rnm(b)(l,n)
                            )/2.0/(Emn*Emn*(Emn-2.0*omega))*(Eln-Eml);
                            }
                        #else

                            if(abs(H1.energyweight(n)-H1.energyweight(m))>0.0001){
                                fac3=2*(H1.energyweight(n)-H1.energyweight(m))*H1.rnm(a)(n,m)*(
                                H1.rnm(b)(m,l)*H1.rnm(c)(l,n)
                                +H1.rnm(c)(m,l)*H1.rnm(b)(l,n)
                            )/2.0/(Emn*Emn*(Emn-2.0*omega))*(-(Eln-Eml));
                            // if(abs(fac3)>0.1){
                            // std::cout<<omega<<n<<m<<l<<fac3<<std::endl;}
                            }
                        #endif
                        ita(idirect)(ihv,nowkindex)+=(fac1+fac2+fac3);
                    }
                    
                }
                
            }
            
        }
        
    }
}

void solveshg::updatesigma(){
        complex<double> fac1,fac2,fac3,fac12;
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        int a=directindice(idirect,0);
        int b=directindice(idirect,1);
        int c=directindice(idirect,2);

        for (int ihv = 0; ihv < nhv; ihv++)
        {
            sigma(idirect)(ihv,nowkindex)=0;
        }
        
//可能有问题
        for (int ihv = 0; ihv < nhv; ihv++)
        {
            complex<double> &&omega=hv(ihv)+complex<double>(0,1)*ksi;
            for (int n = 0; n < H1.wcentnum; n++)
            {
                for (int m = 0; m < H1.wcentnum; m++)
                {
                    double && Emn=H1.energy(m)-H1.energy(n);
                    if (abs(H1.energyweight(n)-H1.energyweight(m))>0.0001){
                        #ifndef newversion
                            fac12 = complex<double>(0,1)*(H1.energyweight(n)-H1.energyweight(m))*H1.rnm(a)(n,m)*(
                                (H1.vnm(b)(m,m)-H1.vnm(b)(n,n))*H1.rnm(c)(m,n)
                                +(H1.vnm(c)(m,m)-H1.vnm(c)(n,n))*H1.rnm(b)(m,n)
                            )/2.0/(Emn*Emn*(Emn-1.0*omega));  //这里似乎应该是Emn-2*omega ，本来为Emn-omega //修改
                        #else
                            fac12 = complex<double>(0,1)*(H1.energyweight(n)-H1.energyweight(m))*(H1.vnm(a)(n,n)-H1.vnm(a)(m,m))*(
                                H1.rnm(b)(m,n)*H1.rnm(c)(n,m)
                                +H1.rnm(c)(m,n)*H1.rnm(b)(n,m)
                            )/2.0/(Emn*Emn*(Emn-1.0*omega));  //这里似乎应该是Emn-2*omega ，本来为Emn-omega //修改
                        #endif
                        sigma(idirect)(ihv,nowkindex)+=fac12/2.0;
                    }

                    for (int l = 0; l < H1.wcentnum; l++)
                    {
                        double &&Enl=H1.energy(n)-H1.energy(l);
                        double &&Elm=H1.energy(l)-H1.energy(m);

                        fac1=0;
                        fac2=0;
                        fac3=0;
                        #ifndef newversion

                            if(abs(H1.energyweight(n)-H1.energyweight(m))>0.0001){
                                fac3=(H1.energyweight(n)-H1.energyweight(m))*(
                                Enl*H1.rnm(a)(l,m)*(H1.rnm(b)(m,n)*H1.rnm(c)(n,l)+H1.rnm(c)(m,n)*H1.rnm(b)(n,l))/2.0
                                -Elm*H1.rnm(a)(n,l)*(H1.rnm(b)(l,m)*H1.rnm(c)(m,n)+H1.rnm(c)(l,m)*H1.rnm(b)(m,n))/2.0
                            )/(Emn*Emn*(Emn-omega));
                            }
                        #else
                            if(abs(H1.energyweight(n)-H1.energyweight(m))>0.0001){
                                fac3=complex<double>(0,1)*(H1.energyweight(n)-H1.energyweight(m))*(
                                Enl*H1.rnm(a)(l,m)*(H1.rnm(b)(m,n)*H1.rnm(c)(n,l)+H1.rnm(c)(m,n)*H1.rnm(b)(n,l))/2.0
                                -Elm*H1.rnm(a)(n,l)*(H1.rnm(b)(l,m)*H1.rnm(c)(m,n)+H1.rnm(c)(l,m)*H1.rnm(b)(m,n))/2.0
                            )/(Emn*Emn*(Emn-omega));
                            }
                        #endif
                        sigma(idirect)(ihv,nowkindex)+=(fac3/2.0);
                    }
                    
                }
                
            }
            
        }
        
    }
}

void solveshg::updateshg(){
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        for (int i = 0; i < nhv; i++)
        {
            shg(idirect)(i,nowkindex)=chi(idirect)(i,nowkindex)+ita(idirect)(i,nowkindex)+sigma(idirect)(i,nowkindex);
        }
        
    }
    
}

void solveshg::updatek(int ki){
    
    H1.runonekpoints(kmesh.row(ki));
    nowkindex=ki;
    updatechi();
    
    updateita();
    
    updatesigma();
    updateshg();
    // H1.setup();
    // if (ki==3){
    //     std::cout<<"chi3:"<< H1.energy<<std::endl;
    // }
    // std::cout<<"chi:\n"<<chi(0)<<std::endl;
    // std::cout<<"ita:\n"<<ita(0)<<std::endl;
    // std::cout<<"sigma:\n"<<sigma(0)<<std::endl;
}