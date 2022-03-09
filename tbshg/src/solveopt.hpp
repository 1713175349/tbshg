#pragma once
#include"Eigen/Dense"
#include<Eigen/Eigenvalues>
#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<pybind11/numpy.h>
#include "hamiltoniank.hpp"
#include <cmath>
#include <memory>
//#define newversion

class solveopt
{
private:
    double ksi=0.02;
    std::shared_ptr<Hamiltoniank> H1;//哈密顿矩阵
public:
    // solveopt(/* args */);
    // ~solveopt();
    //Hamiltoniank H1;//哈密顿
    
    // int nkpts;
    // int ndirects;
    // int nhv;
    // Eigen::MatrixX3d kmesh;//k点网格
    // Eigen::VectorXd kweight;//k点权重
    // Eigen::VectorXd energyweight;//能带权重
    //Eigen::MatrixX3i directindice;//方向指标
    //Eigen::VectorXd hv;//光子能量
    // Eigen::Vector<Eigen::MatrixXcd,Eigen::Dynamic> shg;//(张量指标，hv,kvector)
    // Eigen::Vector<Eigen::MatrixXcd,Eigen::Dynamic> chi;
    // Eigen::Vector<Eigen::MatrixXcd,Eigen::Dynamic> ita;
    // Eigen::Vector<Eigen::MatrixXcd,Eigen::Dynamic> sigma;

    // Eigen::MatrixXcd shg;//(张量指标，hv)
    // Eigen::MatrixXcd chi;
    // Eigen::MatrixXcd ita;
    // Eigen::MatrixXcd sigma;
    
    solveopt()=default;
    solveopt(std::shared_ptr<Hamiltoniank> H1_,double ksi_=0.02):H1(H1_),ksi(ksi_){}

    void set_H1(std::shared_ptr<Hamiltoniank> H1_){
        H1=H1_;
    }
    std::shared_ptr<Hamiltoniank> get_H1(){
        return H1;
    }
    
    double set_ksi(double ksi_){
        ksi=ksi_;
        return ksi;
    }
    double get_ksi(){
        return ksi;
    }

    Eigen::MatrixXcd get_shg(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);
    Eigen::MatrixXcd get_chi(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);
    Eigen::MatrixXcd get_ita(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);
    Eigen::MatrixXcd get_sigma(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);

};

Eigen::MatrixXcd solveopt::get_chi(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){

    H1->runonekpoints(kvector);

    const auto & energyeigenval = H1->energy;
    int bandnum=H1->wcentnum;
    int ndirects=directindice.rows();
    int nhv=hv.size();
    const auto & rnm=H1->rnm;
    const auto & vnm=H1->vnm;
    const auto & energyweight=H1->energyweight;


    
    
    //当前k点chi存储变量
    Eigen::MatrixXcd chi(directindice.rows(),hv.size());
    chi.setZero();
    // std::cout<<chi.sum()<<std::endl;
    complex<double> fac1=0;
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        
        int a=directindice(idirect,0);
        int b=directindice(idirect,1);
        int c=directindice(idirect,2);

//可能有问题
        for (int m = 0; m < bandnum; m++)
        {
            for (int l = 0; l < bandnum; l++)
            {
                for (int n = 0; n < bandnum; n++)
                {
                    fac1=0;
                   
                    double &&Eln=energyeigenval(l)-energyeigenval(n);
                    double &&Eml=energyeigenval(m)-energyeigenval(l);
                    double &&Emn=energyeigenval(m)-energyeigenval(n);
                    if (abs(Eln-Eml) > 0.0001){
                        
                        fac1=rnm(a)(n,m)*(rnm(b)(m,l)*rnm(c)(l,n)+rnm(c)(m,l)*rnm(b)(l,n))/(Eln-Eml);
                    //     if(abs(fac1)>0.001 &&idirect==1){
                    //     std::cout<<n<<"-"<<l<<"-"<<m<<":"<<fac1<<std::endl;}
                    
                    }
                    // std::cout<<fac1<<std::endl;
                    for (int ihv = 0; ihv < nhv; ihv++)
                    {
                        complex<double> &&omega=hv(ihv)+complex<double>(0,1)*ksi;
                        complex<double> &&fac2=(energyweight(m)-energyweight(l))/(Eml-omega)
                                        +(energyweight(l)-energyweight(n))/(Eln-omega)
                                        +2*(energyweight(n)-energyweight(m))/(Emn-2.0*omega);
                        // if(n==9 && m==23 && l==19){
                        //     std::cout<<n<<l<<m<<"fac2:"<<omega<<fac2<<(energyweight(m)-energyweight(l))/(Eml-omega)<<std::endl;
                        // }
                        // if(abs(fac2)>0.001 && idirect ==1 && ihv==44 && nowkindex==3){
                        // std::cout<<n<<"-"<<l<<"-"<<m<<":"<<chi(idirect,ihv)<<fac1<<fac2<<std::endl;}
                        // if (n==15 && m==0 && l==14 && idirect==1 && ihv==44 && nowkindex<4){
                        //     std::cout<<n<<"-"<<l<<"-"<<m<<"-"<<ihv<<":"<<fac1<<rnm(a)(n,m)<<rnm(b)(m,l)<<rnm(c)(l,n)<<rnm(c)(m,l)<<rnm(b)(l,n)<<std::endl;
                        // }
                        chi(idirect,ihv)+=0.5*fac1*fac2; //这里0.5把对称化部分除了
                    }
                    
                }
                
            }
            
        }
        
    }
    
    return chi;
    
}

Eigen::MatrixXcd solveopt::get_ita(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){
    H1->runonekpoints(kvector);
    const auto & energyeigenval = H1->energy;
    int bandnum=H1->wcentnum;
    int ndirects=directindice.rows();
    int nhv=hv.size();
    const auto & rnm=H1->rnm;
    const auto & vnm=H1->vnm;
    const auto & energyweight=H1->energyweight;

    //当前k点ita存储变量
    Eigen::MatrixXcd ita(directindice.rows(),hv.size());
    ita.setZero();

    complex<double> fac1,fac2,fac3,fac12;
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        int a=directindice(idirect,0);
        int b=directindice(idirect,1);
        int c=directindice(idirect,2);

        
//可能有问题
        for (int ihv = 0; ihv < nhv; ihv++)
        {
            complex<double> &&omega=hv(ihv)+complex<double>(0,1)*ksi;
            for (int n = 0; n < bandnum; n++)
            {
                for (int m = 0; m < bandnum; m++)
                {
                    double && Emn=energyeigenval(m)-energyeigenval(n);
                    if (abs(energyweight(n)-energyweight(m))>0.0001){
                        fac12=-8.0*complex<double>(0,1)*(energyweight(n)-energyweight(m))*rnm(a)(n,m)*(
                            (vnm(b)(m,m)-vnm(b)(n,n))*rnm(c)(m,n)
                            +(vnm(c)(m,m)-vnm(c)(n,n))*rnm(b)(m,n)
                        )/2.0/(Emn*Emn*(Emn-2.0*omega));
                        
                        // fac12 为nan时输出调试信息
                        // if(std::isfinite(fac12.real())==false){
                        //     std::cout<<"fac12:"<<fac12<<std::endl;
                        //     std::cout<<"Emn:"<<Emn<<std::endl;
                        //     std::cout<<"omega:"<<omega<<std::endl;
                        //     std::cout<<"n:"<<n<<std::endl;
                        //     std::cout<<"m:"<<m<<std::endl;
                        //     std::cout<<"energyweight(n):"<<energyweight(n)<<std::endl;
                        //     std::cout<<"energyweight(m):"<<energyweight(m)<<std::endl;
                        //     std::cout<<"rnm(a)(n,m):"<<rnm(a)(n,m)<<std::endl;
                        //     std::cout<<"vnm(b)(m,m):"<<vnm(b)(m,m)<<std::endl;
                        //     std::cout<<"vnm(b)(n,n):"<<vnm(b)(n,n)<<std::endl;
                        //     std::cout<<"rnm(c)(m,n):"<<rnm(c)(m,n)<<std::endl;
                        //     std::cout<<"vnm(c)(m,m):"<<vnm(c)(m,m)<<std::endl;
                        //     std::cout<<"vnm(c)(n,n):"<<vnm(c)(n,n)<<std::endl;
                        //     std::cout<<"rnm(b)(m,n):"<<rnm(b)(m,n)<<std::endl;
                        //     std::cout<<"Emn:"<<Emn<<std::endl;
                        //     std::cout<<"omega:"<<omega<<std::endl;
                        //     std::cout<<"n:"<<n<<std::endl;
                        //     std::cout<<"m:"<<m<<std::endl;
                        //     std::cout<<"energyweight(n):"<<energyweight(n)<<std::endl;
                        // }
                        ita(idirect,ihv)+=fac12;
                    }

                    for (int l = 0; l < bandnum; l++)
                    {
                        double &&Eln=energyeigenval(l)-energyeigenval(n);
                        double &&Eml=energyeigenval(m)-energyeigenval(l);

                        fac1=0;
                        fac2=0;
                        fac3=0;
                        if(abs(energyweight(n)-energyweight(l))>0.0001){
                            fac1=Emn*(energyweight(n)-energyweight(l))*rnm(a)(n,m)*(
                            rnm(b)(m,l)*rnm(c)(l,n)
                            +rnm(c)(m,l)*rnm(b)(l,n)
                        )/2.0/(Eln*Eln*(Eln-omega));
                        
                        }
                        #ifndef newversion
                            if(abs(energyweight(l)-energyweight(m))>0.0001){
                                fac2=Emn*(energyweight(l)-energyweight(m))*rnm(a)(n,m)*(
                                rnm(b)(m,l)*rnm(c)(l,n)
                                +rnm(c)(m,l)*rnm(b)(l,n)
                            )/2.0/(Eml*Eml*(Eml-omega));
                            
                            }
                        #else
                            if(abs(energyweight(l)-energyweight(m))>0.0001){
                                fac2=-Emn*(energyweight(l)-energyweight(m))*rnm(a)(n,m)*(
                                rnm(b)(m,l)*rnm(c)(l,n)
                                +rnm(c)(m,l)*rnm(b)(l,n)
                            )/2.0/(Eml*Eml*(Eml-omega));

                            }
                        #endif

                        #ifndef newversion
                            if(abs(energyweight(n)-energyweight(m))>0.0001){
                                fac3=-2*(energyweight(n)-energyweight(m))*rnm(a)(n,m)*(
                                rnm(b)(m,l)*rnm(c)(l,n)
                                +rnm(c)(m,l)*rnm(b)(l,n)
                            )/2.0/(Emn*Emn*(Emn-2.0*omega))*(Eln-Eml);
                            }
                        #else

                            if(abs(energyweight(n)-energyweight(m))>0.0001){
                                fac3=2*(energyweight(n)-energyweight(m))*rnm(a)(n,m)*(
                                rnm(b)(m,l)*rnm(c)(l,n)
                                +rnm(c)(m,l)*rnm(b)(l,n)
                            )/2.0/(Emn*Emn*(Emn-2.0*omega))*(-(Eln-Eml));
                            // if(abs(fac3)>0.1){
                            // std::cout<<omega<<n<<m<<l<<fac3<<std::endl;}
                            }
                        #endif
                        ita(idirect,ihv)+=(fac1+fac2+fac3);
                    }
                    
                }
                
            }
            
        }
        
    }
    return ita;
}

Eigen::MatrixXcd solveopt::get_sigma(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){
    H1->runonekpoints(kvector);
    const auto & energyeigenval = H1->energy;
    int bandnum=H1->wcentnum;
    int ndirects=directindice.rows();
    int nhv=hv.size();
    const auto & rnm=H1->rnm;
    const auto & vnm=H1->vnm;
    const auto & energyweight=H1->energyweight;

    //当前k点sigma存储变量
    Eigen::MatrixXcd sigma(directindice.rows(),hv.size());
    sigma.setZero();

    complex<double> fac1,fac2,fac3,fac12;
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        int a=directindice(idirect,0);
        int b=directindice(idirect,1);
        int c=directindice(idirect,2);

        
//可能有问题
        for (int ihv = 0; ihv < nhv; ihv++)
        {
            complex<double> &&omega=hv(ihv)+complex<double>(0,1)*ksi;
            for (int n = 0; n < bandnum; n++)
            {
                for (int m = 0; m < bandnum; m++)
                {
                    double && Emn=energyeigenval(m)-energyeigenval(n);
                    if (abs(energyweight(n)-energyweight(m))>0.0001){
                        #ifndef newversion
                            fac12 = complex<double>(0,1)*(energyweight(n)-energyweight(m))*rnm(a)(n,m)*(
                                (vnm(b)(m,m)-vnm(b)(n,n))*rnm(c)(m,n)
                                +(vnm(c)(m,m)-vnm(c)(n,n))*rnm(b)(m,n)
                            )/2.0/(Emn*Emn*(Emn-1.0*omega));  //这里似乎应该是Emn-2*omega ，本来为Emn-omega //修改
                        #else
                            fac12 = complex<double>(0,1)*(energyweight(n)-energyweight(m))*(vnm(a)(n,n)-vnm(a)(m,m))*(
                                rnm(b)(m,n)*rnm(c)(n,m)
                                +rnm(c)(m,n)*rnm(b)(n,m)
                            )/2.0/(Emn*Emn*(Emn-1.0*omega));  //这里似乎应该是Emn-2*omega ，本来为Emn-omega //修改
                        #endif
                        sigma(idirect,ihv)+=fac12/2.0;
                    }

                    for (int l = 0; l < bandnum; l++)
                    {
                        double &&Enl=energyeigenval(n)-energyeigenval(l);
                        double &&Elm=energyeigenval(l)-energyeigenval(m);

                        fac1=0;
                        fac2=0;
                        fac3=0;
                        #ifndef newversion

                            if(abs(energyweight(n)-energyweight(m))>0.0001){
                                fac3=(energyweight(n)-energyweight(m))*(
                                Enl*rnm(a)(l,m)*(rnm(b)(m,n)*rnm(c)(n,l)+rnm(c)(m,n)*rnm(b)(n,l))/2.0
                                -Elm*rnm(a)(n,l)*(rnm(b)(l,m)*rnm(c)(m,n)+rnm(c)(l,m)*rnm(b)(m,n))/2.0
                            )/(Emn*Emn*(Emn-omega));
                            }
                        #else
                            if(abs(energyweight(n)-energyweight(m))>0.0001){
                                fac3=complex<double>(0,1)*(energyweight(n)-energyweight(m))*(
                                Enl*rnm(a)(l,m)*(rnm(b)(m,n)*rnm(c)(n,l)+rnm(c)(m,n)*rnm(b)(n,l))/2.0
                                -Elm*rnm(a)(n,l)*(rnm(b)(l,m)*rnm(c)(m,n)+rnm(c)(l,m)*rnm(b)(m,n))/2.0
                            )/(Emn*Emn*(Emn-omega));
                            }
                        #endif
                        sigma(idirect,ihv)+=(fac3/2.0);
                    }
                    
                }
                
            }
            
        }
        
    }
    return sigma;
}

Eigen::MatrixXcd solveopt::get_shg(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){
    return get_sigma(kvector,hv,directindice)+get_ita(kvector,hv,directindice) +get_chi(kvector,hv,directindice);
}

