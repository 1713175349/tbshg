#pragma once
#include"Eigen/Dense"
#include<Eigen/Eigenvalues>
#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<pybind11/numpy.h>
#include "hamiltoniank.hpp"
#include <cmath>
#include <memory>
#include <complex>

#include <functional>
#include <random>
//#define newversion

class solveopt
{
private:
    double ksi=0.02;
    double etolerance=1e-6;
    double max_occ=2.0; //每条能带的最大占据数
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

    double set_max_occ(double max_occ_){
        max_occ=max_occ_;
        return max_occ;
    }
    double get_max_occ(){
        return max_occ;
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

    Eigen::MatrixXcd get_shg_f(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);
    Eigen::MatrixXcd get_chi_f(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);
    Eigen::MatrixXcd get_ita_f(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);
    Eigen::MatrixXcd get_sigma_f(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);


    std::unique_ptr<std::random_device> rd;
    std::unique_ptr<std::mt19937> gen;
    std::unique_ptr<std::uniform_int_distribution<>> dis;
    void setup_mc();//配置蒙特卡洛需要的随机数生成器
    int get_random_band();//随机选择一个band
    Eigen::MatrixXcd get_chi_mc(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice,int times=10000); //monte carlo 方法计算chi
    Eigen::MatrixXcd get_ita_mc(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice,int times=10000); //monte carlo 方法计算ita
    Eigen::MatrixXcd get_sigma_mc(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice,int times=10000); //monte carlo 方法计算sigma
    Eigen::MatrixXcd get_shg_mc(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice,int times=10000); //monte carlo 方法计算shg

    Eigen::MatrixXcd get_linechi(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX2i directindice);

    //采用无规相近似的求和，用以大大加快求和速度
    Eigen::MatrixXcd get_inter(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);
    Eigen::MatrixXcd get_intra(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);
    Eigen::MatrixXcd get_modu(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);
    Eigen::MatrixXcd get_shg_rpa(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice);
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
                    if ((energyweight(l) == energyweight(n)) and (energyweight(l) == energyweight(m)))
                    {
                        continue;
                    }
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
    
    return max_occ*chi;
    
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
                        if ((energyweight(l) == energyweight(n)) and (energyweight(l) == energyweight(m)))
                        {
                            continue;
                        }
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
    return max_occ*ita;
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

                        if ((energyweight(l) == energyweight(n)) and (energyweight(l) == energyweight(m)))
                        {
                            continue;
                        }
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
    return max_occ*sigma;
}

Eigen::MatrixXcd solveopt::get_shg(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){
    return get_sigma(kvector,hv,directindice)+get_ita(kvector,hv,directindice) +get_chi(kvector,hv,directindice);
}

Eigen::MatrixXcd solveopt::get_shg_rpa(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){
    return get_inter(kvector,hv,directindice)+get_intra(kvector,hv,directindice) +get_modu(kvector,hv,directindice);
}

Eigen::MatrixXcd solveopt::get_linechi(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX2i directindice){
    /*
    \chi^1_{ba}(-\omega,\omega)=e^2/hbr \sum_{nm,k} \frac{(r^b_{mn}r^a_{nm}f_{mn})}   {(\omega_{nm}-\omega)}
    */
    
    H1->runonekpoints(kvector);

    const auto & energyeigenval = H1->energy;
    int bandnum=H1->wcentnum;
    int ndirects=directindice.rows();
    int nhv=hv.size();
    const auto & rnm=H1->rnm;
    const auto & vnm=H1->vnm;
    const auto & energyweight=H1->energyweight;

    //当前k点chi存储变量
    Eigen::MatrixXcd chi1(directindice.rows(),hv.size());
    chi1.setZero();
    for (int idirect = 0; idirect<ndirects;idirect++){
        int b=directindice(idirect,0); //响应极化的偏振方向
        int a=directindice(idirect,1);
        for(int m=0;m<bandnum;m++){
            for (int n=0;n<bandnum;n++){
                double Enm=energyeigenval(n)-energyeigenval(m);
                
                complex<double> frac1=rnm(b)(m,n)*rnm(a)(n,m)*(energyweight(m)-energyweight(n));
                for(int ihv=0;ihv<nhv;ihv++){
                    auto && omega=hv(ihv)+complex<double>(0,1)*ksi;
                    chi1(idirect,ihv)+=frac1/(Enm-omega);
                }

            }

        }
    }   
    return max_occ*chi1;
}

void solveopt::setup_mc(){
    int seed=std::random_device{}();
    int bandnum=H1->wcentnum;
    gen.reset(new std::mt19937(seed));
    dis.reset(new std::uniform_int_distribution<>(0,bandnum-1));
}

int solveopt::get_random_band(){
    return (*dis)(*gen);
}


Eigen::MatrixXcd solveopt::get_chi_mc(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice,int times){

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
    

//对于每个响应极化方向，从不同的band中随机选择一个band
    int n,l,m;
    for (int ti = 0; ti < times; ti++)
    {
        m=get_random_band();
        n=get_random_band();
        l=get_random_band();
        // 非随机的遍历代码，用于调试，确保公式一致
        // m=ti%bandnum;
        // n=ti/(bandnum*bandnum);
        // l=(ti%(bandnum*bandnum))/bandnum;
        // if (n>=bandnum){
        //     break;
        // }
        if ((energyweight(l) == energyweight(n)) and (energyweight(l)==energyweight(m)) )
        {
            continue;
        }
        for (int idirect = 0; idirect < ndirects; idirect++)
        {
            int a=directindice(idirect,0);
            int b=directindice(idirect,1);
            int c=directindice(idirect,2);
            
            fac1 = 0;

            double &&Eln = energyeigenval(l) - energyeigenval(n);
            double &&Eml = energyeigenval(m) - energyeigenval(l);
            double &&Emn = energyeigenval(m) - energyeigenval(n);
            if (abs(Eln - Eml) > 0.0001)
            {

                fac1 = rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / (Eln - Eml);
                //     if(abs(fac1)>0.001 &&idirect==1){
                //     std::cout<<n<<"-"<<l<<"-"<<m<<":"<<fac1<<std::endl;}
            }
            // std::cout<<fac1<<std::endl;
            for (int ihv = 0; ihv < nhv; ihv++)
            {
                complex<double> &&omega = hv(ihv) + complex<double>(0, 1) * ksi;
                complex<double> &&fac2 = (energyweight(m) - energyweight(l)) / (Eml - omega) + (energyweight(l) - energyweight(n)) / (Eln - omega) + 2 * (energyweight(n) - energyweight(m)) / (Emn - 2.0 * omega);
                // if(n==9 && m==23 && l==19){
                //     std::cout<<n<<l<<m<<"fac2:"<<omega<<fac2<<(energyweight(m)-energyweight(l))/(Eml-omega)<<std::endl;
                // }
                // if(abs(fac2)>0.001 && idirect ==1 && ihv==44 && nowkindex==3){
                // std::cout<<n<<"-"<<l<<"-"<<m<<":"<<chi(idirect,ihv)<<fac1<<fac2<<std::endl;}
                // if (n==15 && m==0 && l==14 && idirect==1 && ihv==44 && nowkindex<4){
                //     std::cout<<n<<"-"<<l<<"-"<<m<<"-"<<ihv<<":"<<fac1<<rnm(a)(n,m)<<rnm(b)(m,l)<<rnm(c)(l,n)<<rnm(c)(m,l)<<rnm(b)(l,n)<<std::endl;
                // }
                chi(idirect, ihv) += 0.5 * fac1 * fac2; //这里0.5把对称化部分除了
            }
        }
    }
    
    return max_occ*chi*bandnum*bandnum*bandnum/times;
    
}


Eigen::MatrixXcd solveopt::get_ita_mc(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice,int times){
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
    int n,l,m;
    for (int ti = 0; ti < times; ti++)
    {
        m = get_random_band();
        n = get_random_band();
        l = get_random_band();
        // 非随机的遍历代码，用于调试，确保公式一致
        // m=ti%bandnum;
        // n=ti/(bandnum*bandnum);
        // l=(ti%(bandnum*bandnum))/bandnum;
        // if (n>=bandnum){
        //     break;
        // }
        if ((energyweight(l) == energyweight(n)) and (energyweight(l)==energyweight(m)) )
        {
            continue;
        }
        for (int idirect = 0; idirect < ndirects; idirect++)
        {
            int a = directindice(idirect, 0);
            int b = directindice(idirect, 1);
            int c = directindice(idirect, 2);

            //可能有问题
            for (int ihv = 0; ihv < nhv; ihv++)
            {
                complex<double> &&omega = hv(ihv) + complex<double>(0, 1) * ksi;

                double &&Emn = energyeigenval(m) - energyeigenval(n);
                if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                {
                    fac12 = -8.0 * complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * ((vnm(b)(m, m) - vnm(b)(n, n)) * rnm(c)(m, n) + (vnm(c)(m, m) - vnm(c)(n, n)) * rnm(b)(m, n)) / 2.0 / (Emn * Emn * (Emn - 2.0 * omega));
                    fac12 /= bandnum;
                    ita(idirect, ihv) += fac12;
                }

                double &&Eln = energyeigenval(l) - energyeigenval(n);
                double &&Eml = energyeigenval(m) - energyeigenval(l);

                fac1 = 0;
                fac2 = 0;
                fac3 = 0;
                if (abs(energyweight(n) - energyweight(l)) > 0.0001)
                {
                    fac1 = Emn * (energyweight(n) - energyweight(l)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Eln * Eln * (Eln - omega));
                }
#ifndef newversion
                if (abs(energyweight(l) - energyweight(m)) > 0.0001)
                {
                    fac2 = Emn * (energyweight(l) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Eml * Eml * (Eml - omega));
                }
#else
                if (abs(energyweight(l) - energyweight(m)) > 0.0001)
                {
                    fac2 = -Emn * (energyweight(l) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Eml * Eml * (Eml - omega));
                }
#endif

#ifndef newversion
                if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                {
                    fac3 = -2 * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Emn * Emn * (Emn - 2.0 * omega)) * (Eln - Eml);
                }
#else

                if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                {
                    fac3 = 2 * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Emn * Emn * (Emn - 2.0 * omega)) * (-(Eln - Eml));
                    // if(abs(fac3)>0.1){
                    // std::cout<<omega<<n<<m<<l<<fac3<<std::endl;}
                }
#endif
                ita(idirect, ihv) += (fac1 + fac2 + fac3);
            }
        }
    }
    return max_occ*ita*bandnum*bandnum*bandnum/times;
}


Eigen::MatrixXcd solveopt::get_sigma_mc(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice,int times){
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
    int n,l,m;
    for (int ti = 0; ti < times; ti++)
    {
        m = get_random_band();
        n = get_random_band();
        l = get_random_band();
        // 非随机的遍历代码，用于调试，确保公式一致
        // m=ti%bandnum;
        // n=ti/(bandnum*bandnum);
        // l=(ti%(bandnum*bandnum))/bandnum;
        // if (n>=bandnum){
        //     break;
        // }
        if ((energyweight(l) == energyweight(n)) and (energyweight(l)==energyweight(m)) )
        {
            continue;
        }
        for (int idirect = 0; idirect < ndirects; idirect++)
        {
            int a = directindice(idirect, 0);
            int b = directindice(idirect, 1);
            int c = directindice(idirect, 2);

            //可能有问题
            for (int ihv = 0; ihv < nhv; ihv++)
            {
                complex<double> &&omega = hv(ihv) + complex<double>(0, 1) * ksi;

                double &&Emn = energyeigenval(m) - energyeigenval(n);
                if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                {
#ifndef newversion
                    fac12 = complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * ((vnm(b)(m, m) - vnm(b)(n, n)) * rnm(c)(m, n) + (vnm(c)(m, m) - vnm(c)(n, n)) * rnm(b)(m, n)) / 2.0 / (Emn * Emn * (Emn - 1.0 * omega)); //这里似乎应该是Emn-2*omega ，本来为Emn-omega //修改
#else
                    fac12 = complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * (vnm(a)(n, n) - vnm(a)(m, m)) * (rnm(b)(m, n) * rnm(c)(n, m) + rnm(c)(m, n) * rnm(b)(n, m)) / 2.0 / (Emn * Emn * (Emn - 1.0 * omega)); //这里似乎应该是Emn-2*omega ，本来为Emn-omega //修改
#endif
                    fac12/=bandnum;
                    sigma(idirect, ihv) += fac12 / 2.0;
                }

                double &&Enl = energyeigenval(n) - energyeigenval(l);
                double &&Elm = energyeigenval(l) - energyeigenval(m);

                fac1 = 0;
                fac2 = 0;
                fac3 = 0;
#ifndef newversion

                if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                {
                    fac3 = (energyweight(n) - energyweight(m)) * (Enl * rnm(a)(l, m) * (rnm(b)(m, n) * rnm(c)(n, l) + rnm(c)(m, n) * rnm(b)(n, l)) / 2.0 - Elm * rnm(a)(n, l) * (rnm(b)(l, m) * rnm(c)(m, n) + rnm(c)(l, m) * rnm(b)(m, n)) / 2.0) / (Emn * Emn * (Emn - omega));
                }
#else
                if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                {
                    fac3 = complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * (Enl * rnm(a)(l, m) * (rnm(b)(m, n) * rnm(c)(n, l) + rnm(c)(m, n) * rnm(b)(n, l)) / 2.0 - Elm * rnm(a)(n, l) * (rnm(b)(l, m) * rnm(c)(m, n) + rnm(c)(l, m) * rnm(b)(m, n)) / 2.0) / (Emn * Emn * (Emn - omega));
                }
#endif
               
                sigma(idirect, ihv) += (fac3 / 2.0);
            }
        }
    }
    return max_occ*sigma*bandnum*bandnum*bandnum/times;
}

Eigen::MatrixXcd solveopt::get_shg_mc(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice,int times){
    return get_sigma_mc(kvector,hv,directindice,times)+get_ita_mc(kvector,hv,directindice,times) +get_chi_mc(kvector,hv,directindice,times);
}


Eigen::MatrixXcd solveopt::get_inter(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){
    H1->runonekpoints(kvector);

    const auto & energyeigenval = H1->energy;
    int bandnum=H1->wcentnum;
    int ndirects=directindice.rows();
    int nhv=hv.size();
    const auto & rnm=H1->rnm;
    const auto & vnm=H1->vnm;
    const auto & energyweight=H1->energyweight;
    //当前k点inter存储变量
    Eigen::MatrixXcd inter(directindice.rows(),hv.size());
    inter.setZero();
    // std::cout<<chi.sum()<<std::endl;
    complex<double> fac1,fac2;
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        
        int a=directindice(idirect,0);
        int b=directindice(idirect,1);
        int c=directindice(idirect,2);
        for (int n=0; n < bandnum; n++)
        {
            if (energyweight(n)<0.5){
                continue;
            }
            for (int m=0; m < bandnum; m++)
            {
                if (energyweight(m)>0.5){
                    continue;
                }
                double && emn=energyeigenval(m)-energyeigenval(n);
                for (int l=0;l<bandnum;l++){
                    if (l==n || l==m){
                        continue;
                    }

                    
                   

                    double && eln=energyeigenval(l)-energyeigenval(n);
                    double && eml=energyeigenval(m)-energyeigenval(l);
                    //去掉共振峰
                    if (abs(eln-eml)<etolerance || abs(emn+eln)<etolerance || abs(emn+eml)<etolerance || abs(eln)<etolerance || abs(emn)<etolerance || abs(eml)<etolerance){
                        continue;
                    }
                    complex<double> && rbc=rnm(c)(m,l)*rnm(b)(l,n)+rnm(b)(m,l)*rnm(c)(l,n);
                    complex<double> && rab=rnm(a)(m,n)*rnm(b)(n,l)+rnm(b)(m,n)*rnm(a)(n,l);
                    complex<double> && rca=rnm(c)(l,m)*rnm(a)(m,n)+rnm(a)(l,m)*rnm(c)(m,n);
                    fac1=2.0*rnm(a)(n,m)*rbc/(eln-eml);
                    fac2=-(rnm(c)(l,m)*rab/(-eln-emn)-rnm(b)(n,l)*rca/(-eml-emn));
                    for (int ihv=0;ihv<nhv;ihv++){
                        complex<double> && hv0=hv(ihv)+complex<double>(0,1)*ksi;
                        inter(idirect,ihv)+=fac1/(emn-2.0*hv0)+fac2/(emn-hv0);
                    }
                }
            }
        }
    }
    return max_occ*inter;

}


Eigen::MatrixXcd solveopt::get_intra(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){
    H1->runonekpoints(kvector);

    const auto & energyeigenval = H1->energy;
    int bandnum=H1->wcentnum;
    int ndirects=directindice.rows();
    int nhv=hv.size();
    const auto & rnm=H1->rnm;
    const auto & vnm=H1->vnm;
    const auto & energyweight=H1->energyweight;
    //当前k点intra存储变量
    Eigen::MatrixXcd intra(directindice.rows(),hv.size());
    intra.setZero();
    // std::cout<<chi.sum()<<std::endl;
    complex<double> fac1,fac2;
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        
        int a=directindice(idirect,0);
        int b=directindice(idirect,1);
        int c=directindice(idirect,2);
        for (int n=0; n < bandnum; n++)
        {
            if (energyweight(n)<0.5){
                continue;
            }
            for (int m=0; m < bandnum; m++)
            {
                if (energyweight(m)>0.5){
                    continue;
                }
                double && emn=energyeigenval(m)-energyeigenval(n);
                for (int l=0;l<bandnum;l++){
                    if (l==n || l==m){
                        continue;
                    }

                    
                   

                    double && eln=energyeigenval(l)-energyeigenval(n);
                    double && eml=energyeigenval(m)-energyeigenval(l);
                    //去掉共振峰
                    if (abs(eln-eml)<etolerance || abs(emn+eln)<etolerance || abs(emn+eml)<etolerance || abs(eln)<etolerance || abs(emn)<etolerance || abs(eml)<etolerance){
                        continue;
                    }
                    complex<double> && rbc=rnm(c)(m,l)*rnm(b)(l,n)+rnm(b)(m,l)*rnm(c)(l,n);
                    complex<double> && rab=rnm(a)(m,n)*rnm(b)(n,l)+rnm(b)(m,n)*rnm(a)(n,l);
                    complex<double> && rca=rnm(c)(l,m)*rnm(a)(m,n)+rnm(a)(l,m)*rnm(c)(m,n);
                    complex<double> && rdbc=(vnm(b)(m,m)-vnm(b)(n,n))*rnm(c)(l,n)+(vnm(c)(m,m)-vnm(c)(n,n))*rnm(b)(l,n);
                    fac1=-complex<double>(0,8.0)*(rnm(a)(n,m)*rdbc)/emn/emn+2.0*rnm(a)(n,m)*rbc*(eml-eln)/emn/emn;
                    fac2=(eln*rnm(b)(n,l)*rca-eml*rnm(c)(l,m)*rab)/emn/emn;
                    for (int ihv=0;ihv<nhv;ihv++){
                        complex<double> && hv0=hv(ihv)+complex<double>(0,1)*ksi;
                        intra(idirect,ihv)+=fac1/(emn-2.0*hv0)+fac2/(emn-hv0);
                    }
                }
            }
        }
    }
    return max_occ*intra;

}


Eigen::MatrixXcd solveopt::get_modu(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){
    H1->runonekpoints(kvector);

    const auto & energyeigenval = H1->energy;
    int bandnum=H1->wcentnum;
    int ndirects=directindice.rows();
    int nhv=hv.size();
    const auto & rnm=H1->rnm;
    const auto & vnm=H1->vnm;
    const auto & energyweight=H1->energyweight;
    //当前k点modu存储变量
    Eigen::MatrixXcd modu(directindice.rows(),hv.size());
    modu.setZero();
    // std::cout<<chi.sum()<<std::endl;
    complex<double> fac1,fac2;
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        
        int a=directindice(idirect,0);
        int b=directindice(idirect,1);
        int c=directindice(idirect,2);
        for (int n=0; n < bandnum; n++)
        {
            if (energyweight(n)<0.5){
                continue;
            }
            for (int m=0; m < bandnum; m++)
            {
                if (energyweight(m)>0.5){
                    continue;
                }
                double && emn=energyeigenval(m)-energyeigenval(n);
                for (int l=0;l<bandnum;l++){
                    if (l==n || l==m){
                        continue;
                    }

                    
                   

                    double && eln=energyeigenval(l)-energyeigenval(n);
                    double && eml=energyeigenval(m)-energyeigenval(l);
                    //去掉共振峰
                    if (abs(eln-eml)<etolerance || abs(emn+eln)<etolerance || abs(emn+eml)<etolerance || abs(eln)<etolerance || abs(emn)<etolerance || abs(eml)<etolerance){
                        continue;
                    }
                    complex<double> && rbc0=rnm(c)(m,n)*rnm(b)(n,l)+rnm(b)(m,n)*rnm(c)(n,l);
                    complex<double> && rbc1=rnm(c)(l,m)*rnm(b)(m,n)+rnm(b)(l,m)*rnm(c)(m,n);
                    // complex<double> && rab=rnm(a)(m,n)*rnm(b)(n,l)+rnm(b)(m,n)*rnm(a)(n,l);
                    // complex<double> && rca=rnm(c)(l,m)*rnm(a)(m,n)+rnm(a)(l,m)*rnm(c)(m,n);
                    // complex<double> && rdbc=(vnm(b)(m,m)-vnm(b)(n,n))*rnm(c)(l,n)+(vnm(c)(m,m)-vnm(c)(n,n))*rnm(b)(l,n);
                    complex<double> && rdbc0=(vnm(c)(m,m)-vnm(c)(n,n))*rnm(b)(m,n)+(vnm(b)(m,m)-vnm(b)(n,n))*rnm(c)(m,n);
                    fac1=0;
                    fac2=((-eln)*rnm(a)(l,m)*rbc0-(-eml)*rnm(a)(n,l)*rbc1)/emn/emn-complex<double>(0,1)*(rnm(a)(n,m)*rdbc0)/emn/emn;
                    for (int ihv=0;ihv<nhv;ihv++){
                        complex<double> && hv0=hv(ihv)+complex<double>(0,1)*ksi;
                        modu(idirect,ihv)+=fac1/(emn-2.0*hv0)-fac2/(emn-hv0);
                    }
                }
            }
        }
    }
    return max_occ*modu;

}



Eigen::MatrixXcd solveopt::get_chi_f(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){

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
    complex<double> fac11,fac12=0,fac2=0;
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
                    double &&Eln=energyeigenval(l)-energyeigenval(n);
                    double &&Eml=energyeigenval(m)-energyeigenval(l);
                    double &&Emn=energyeigenval(m)-energyeigenval(n);
                    // 消除发散项
                    if (abs(Eln-Eml) < etolerance){
                        continue;
                        
                    }
                    complex<double> && fac0=rnm(a)(n,m)*(rnm(b)(m,l)*rnm(c)(l,n)+rnm(c)(m,l)*rnm(b)(l,n))/(Eln-Eml)/2.0;
                    fac11 = fac0*((energyweight(m)-energyweight(l)));
                    fac12 = fac0*(energyweight(l)-energyweight(n));
                    fac2 = 2*(energyweight(n)-energyweight(m))*fac0;
                    // std::cout<<fac1<<std::endl;
                    for (int ihv = 0; ihv < nhv; ihv++)
                    {
                        complex<double> &&omega=hv(ihv)+complex<double>(0,1)*ksi;
                        // complex<double> &&fac2=(energyweight(m)-energyweight(l))/(Eml-omega)
                        //                 +(energyweight(l)-energyweight(n))/(Eln-omega)
                        //                 +2*(energyweight(n)-energyweight(m))/(Emn-2.0*omega);
                        
                        chi(idirect,ihv)+=(fac11/(Eml-omega)+fac12/(Eln-omega)+fac2/(Emn-2.0*omega));
                    }
                    
                }
                
            }
            
        }
        
    }
    
    return max_occ*chi;
    
}

Eigen::MatrixXcd solveopt::get_ita_f(Eigen::Vector3d kvector, Eigen::VectorXd hv, Eigen::MatrixX3i directindice)
{
    H1->runonekpoints(kvector);
    const auto &energyeigenval = H1->energy;
    int bandnum = H1->wcentnum;
    int ndirects = directindice.rows();
    int nhv = hv.size();
    const auto &rnm = H1->rnm;
    const auto &vnm = H1->vnm;
    const auto &energyweight = H1->energyweight;

    //当前k点ita存储变量
    Eigen::MatrixXcd ita(directindice.rows(), hv.size());
    ita.setZero();

    complex<double> fac1, fac2, fac3, fac12;
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        int a = directindice(idirect, 0);
        int b = directindice(idirect, 1);
        int c = directindice(idirect, 2);

        //可能有问题

        for (int n = 0; n < bandnum; n++)
        {
            for (int m = 0; m < bandnum; m++)
            {
                double &&Emn = energyeigenval(m) - energyeigenval(n);
                // 消除发散项
                    // if ( abs(Emn) < etolerance ){
                    //     continue;
                        
                    // }
                if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                {
                    //fac12 = -8.0 * complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * ((vnm(b)(m, m) - vnm(b)(n, n)) * rnm(c)(m, n) + (vnm(c)(m, m) - vnm(c)(n, n)) * rnm(b)(m, n)) / 2.0 / (Emn * Emn * (Emn - 2.0 * omega));
                    fac2 = -8.0 * complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * ((vnm(b)(m, m) - vnm(b)(n, n)) * rnm(c)(m, n) + (vnm(c)(m, m) - vnm(c)(n, n)) * rnm(b)(m, n)) / 2.0 / (Emn * Emn );
                    for (int ihv;ihv<nhv;ihv++){
                        complex<double> &&omega=hv(ihv)+complex<double>(0,1)*ksi;
                        ita(idirect, ihv) += fac2/(Emn - 2.0 * omega);
                    }
                    
                }

                for (int l = 0; l < bandnum; l++)
                {
                    // if ((energyweight(l) == energyweight(n)) and (energyweight(l) == energyweight(m)))
                    // {
                    //     continue;
                    // }
                    
                    double &&Eln = energyeigenval(l) - energyeigenval(n);
                    double &&Eml = energyeigenval(m) - energyeigenval(l);
                    // // 消除发散项
                    // if (abs(Eln-Eml) < etolerance ){
                    //     continue;
                        
                    // }
                    fac1 = 0;
                    fac2 = 0;
                    fac3 = 0;
                    
                        if (abs(energyweight(n) - energyweight(l)) > 0.0001)
                        {
                            // fac1 = Emn * (energyweight(n) - energyweight(l)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Eln * Eln * (Eln - omega));
                            fac1 = Emn * (energyweight(n) - energyweight(l)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Eln * Eln );
                        }
#ifndef newversion
                        if (abs(energyweight(l) - energyweight(m)) > 0.0001)
                        {
                            // fac2 = Emn * (energyweight(l) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Eml * Eml * (Eml - omega));
                            fac2 = Emn * (energyweight(l) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Eml * Eml );
                        }
#else
                        if (abs(energyweight(l) - energyweight(m)) > 0.0001)
                        {
                            // fac2 = -Emn * (energyweight(l) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Eml * Eml * (Eml - omega));
                            fac2 = -Emn * (energyweight(l) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Eml * Eml );
                        }
#endif

#ifndef newversion
                        if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                        {
                            // fac3 = -2 * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Emn * Emn * (Emn - 2.0 * omega)) * (Eln - Eml);
                            fac3 = -2 * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Emn * Emn ) * (Eln - Eml);
                        }
#else

                        if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                        {
                            // fac3 = 2 * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Emn * Emn * (Emn - 2.0 * omega)) * (-(Eln - Eml));
                            fac3 = 2 * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * (rnm(b)(m, l) * rnm(c)(l, n) + rnm(c)(m, l) * rnm(b)(l, n)) / 2.0 / (Emn * Emn ) * (-(Eln - Eml));
                        }
#endif
for (int ihv = 0; ihv < nhv; ihv++)
                    {
                        complex<double> &&omega = hv(ihv) + complex<double>(0, 1) * ksi;
                        ita(idirect, ihv) += (fac1/((Eln - omega)) + fac2/(Eml - omega) + fac3/(Emn - 2.0 * omega));
                    }
                }
            }
        }
    }
    return max_occ*ita;
}

Eigen::MatrixXcd solveopt::get_sigma_f(Eigen::Vector3d kvector, Eigen::VectorXd hv, Eigen::MatrixX3i directindice)
{
    H1->runonekpoints(kvector);
    const auto &energyeigenval = H1->energy;
    int bandnum = H1->wcentnum;
    int ndirects = directindice.rows();
    int nhv = hv.size();
    const auto &rnm = H1->rnm;
    const auto &vnm = H1->vnm;
    const auto &energyweight = H1->energyweight;

    //当前k点sigma存储变量
    Eigen::MatrixXcd sigma(directindice.rows(), hv.size());
    sigma.setZero();

    complex<double> fac1, fac2, fac3, fac12;
    for (int idirect = 0; idirect < ndirects; idirect++)
    {
        int a = directindice(idirect, 0);
        int b = directindice(idirect, 1);
        int c = directindice(idirect, 2);

        //可能有问题

        for (int n = 0; n < bandnum; n++)
        {
            for (int m = 0; m < bandnum; m++)
            {
                double &&Emn = energyeigenval(m) - energyeigenval(n);
                // 消除发散项
                    if ( abs(Emn) < etolerance ){
                        continue;
                        
                    }
                if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                {
#ifndef newversion
                    // fac12 = complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * ((vnm(b)(m, m) - vnm(b)(n, n)) * rnm(c)(m, n) + (vnm(c)(m, m) - vnm(c)(n, n)) * rnm(b)(m, n)) / 2.0 / (Emn * Emn * (Emn - 1.0 * omega)); //这里似乎应该是Emn-2*omega ，本来为Emn-omega //修改
                    fac12 = complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * rnm(a)(n, m) * ((vnm(b)(m, m) - vnm(b)(n, n)) * rnm(c)(m, n) + (vnm(c)(m, m) - vnm(c)(n, n)) * rnm(b)(m, n)) / 2.0 / (Emn * Emn );
#else
                    //fac12 = complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * (vnm(a)(n, n) - vnm(a)(m, m)) * (rnm(b)(m, n) * rnm(c)(n, m) + rnm(c)(m, n) * rnm(b)(n, m)) / 2.0 / (Emn * Emn * (Emn - 1.0 * omega)); //这里似乎应该是Emn-2*omega ，本来为Emn-omega //修改
                    fac12 = complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * (vnm(a)(n, n) - vnm(a)(m, m)) * (rnm(b)(m, n) * rnm(c)(n, m) + rnm(c)(m, n) * rnm(b)(n, m)) / 2.0 / (Emn * Emn );
#endif
                    for (int ihv = 0; ihv < nhv; ihv++)
                    {
                        complex<double> &&omega = hv(ihv) + complex<double>(0, 1) * ksi;
                        sigma(idirect, ihv) += fac12 / 2.0/(Emn - 1.0 * omega);
                    }
                }

                for (int l = 0; l < bandnum; l++)
                {

                    if ((energyweight(l) == energyweight(n)) and (energyweight(l) == energyweight(m)))
                    {
                        continue;
                    }
                    double &&Enl = energyeigenval(n) - energyeigenval(l);
                    double &&Elm = energyeigenval(l) - energyeigenval(m);
                    
                    fac1 = 0;
                    fac2 = 0;
                    fac3 = 0;
#ifndef newversion

                    if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                    {
                        //fac3 = (energyweight(n) - energyweight(m)) * (Enl * rnm(a)(l, m) * (rnm(b)(m, n) * rnm(c)(n, l) + rnm(c)(m, n) * rnm(b)(n, l)) / 2.0 - Elm * rnm(a)(n, l) * (rnm(b)(l, m) * rnm(c)(m, n) + rnm(c)(l, m) * rnm(b)(m, n)) / 2.0) / (Emn * Emn * (Emn - omega));
                        fac3 = (energyweight(n) - energyweight(m)) * (Enl * rnm(a)(l, m) * (rnm(b)(m, n) * rnm(c)(n, l) + rnm(c)(m, n) * rnm(b)(n, l)) / 2.0 - Elm * rnm(a)(n, l) * (rnm(b)(l, m) * rnm(c)(m, n) + rnm(c)(l, m) * rnm(b)(m, n)) / 2.0) / (Emn * Emn );
                    }
#else
                    if (abs(energyweight(n) - energyweight(m)) > 0.0001)
                    {
                        //fac3 = complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * (Enl * rnm(a)(l, m) * (rnm(b)(m, n) * rnm(c)(n, l) + rnm(c)(m, n) * rnm(b)(n, l)) / 2.0 - Elm * rnm(a)(n, l) * (rnm(b)(l, m) * rnm(c)(m, n) + rnm(c)(l, m) * rnm(b)(m, n)) / 2.0) / (Emn * Emn * (Emn - omega));
                        fac3 = complex<double>(0, 1) * (energyweight(n) - energyweight(m)) * (Enl * rnm(a)(l, m) * (rnm(b)(m, n) * rnm(c)(n, l) + rnm(c)(m, n) * rnm(b)(n, l)) / 2.0 - Elm * rnm(a)(n, l) * (rnm(b)(l, m) * rnm(c)(m, n) + rnm(c)(l, m) * rnm(b)(m, n)) / 2.0) / (Emn * Emn );
                    }
#endif
                    for (int ihv = 0; ihv < nhv; ihv++)
                    {
                        complex<double> &&omega = hv(ihv) + complex<double>(0, 1) * ksi;
                        sigma(idirect, ihv) += (fac3 / 2.0/(Emn - 1.0 * omega));
                    }
                }
            }
        }
    }
    return max_occ*sigma;
}


Eigen::MatrixXcd solveopt::get_shg_f(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){
    return get_sigma_f(kvector,hv,directindice)+get_ita_f(kvector,hv,directindice) +get_chi_f(kvector,hv,directindice);
}