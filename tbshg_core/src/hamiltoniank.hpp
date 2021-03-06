#pragma once

#include"Eigen/Dense"

#include<Eigen/Eigenvalues>
#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<pybind11/numpy.h>
#include<complex>

namespace py=pybind11; 
using std::complex;



class Hamiltoniank
{
private:
    /* data */
public:
    // Hamiltoniank(/* args */);
    // ~Hamiltoniank();
    Eigen::MatrixXcd Hk; //存储
    Eigen::MatrixXcd dHkdkx;
    Eigen::MatrixXcd dHkdky;
    Eigen::MatrixXcd dHkdkz;
    int overlapnum;
    int wcentnum;
    int clcdHdk=1; //是否计算动量矩阵
    double bandgapadd=0;
    py::array_t<complex<double>> Hr;//tight binding参数(相对位置，wc1，wc2)
    Eigen::MatrixX3d R;//相对位置 分数坐标
    Eigen::Matrix3d lat;//格矢量
    Eigen::MatrixX3d Rr;//相对位置 笛卡尔坐标
    Eigen::MatrixX3d wcent;//wannier函数中心位置
    Eigen::VectorXd wtR;//R权重
    Eigen::Vector3d nowk;//目前k点
    Eigen::VectorXd energy;//本征能
    Eigen::MatrixXcd wavef;//波函数
    Eigen::VectorXd energyweight;
    Eigen::Vector3d kvector_now{-100000000000,-100000000000,-1000000000000};//k点
    Eigen::Vector<Eigen::MatrixXcd,3> vnm;
    Eigen::Vector<Eigen::MatrixXcd,3> rnm;//（x or y or z,rnmx)
    Eigen::MatrixXcd getrnm(int i){return rnm(i);}
    Eigen::MatrixXcd getvnm(int i){return vnm(i);}
    void setvnm(int index,Eigen::MatrixXcd v){
        for (int i = 0; i < wcentnum; i++)
        {
            for (int j = 0; j < wcentnum; j++)
            {
                vnm(index)(i,j)=v(i,j);
            }
            
        }
        
        }
    void updaternm();
    int setup();
    int updateH(Eigen::Vector3d); //输入正交坐标系下的向量
    int solverH();
    void update_vnmrnm();
    void runonekpoints(Eigen::Vector3d); //输入正交坐标系下的向量

    std::string informations();
};

std::string Hamiltoniank::informations(){
    std::ostringstream oss;
    oss<<"Hamiltoniank:"<<std::endl;
    //int information
    oss<<"overlapnum:"<<overlapnum<<std::endl;
    oss<<"wcentnum:"<<wcentnum<<std::endl;
    oss<<"clcdHdk:"<<clcdHdk<<std::endl;
    oss<<"bandgapadd:"<<bandgapadd<<std::endl;
    // lattice informatin
    oss<<"lat:"<<lat<<std::endl;
    oss<<"R:"<<R<<std::endl;
    oss<<"Rr:"<<Rr<<std::endl;
    oss<<"wcent:"<<wcent<<std::endl;
    oss<<"wtR:"<<wtR<<std::endl;
    oss<<"nowk:"<<nowk<<std::endl;

    return oss.str();
}

int Hamiltoniank::setup(){
    Hk.resize(wcentnum,wcentnum);
    dHkdkx.resize(wcentnum,wcentnum);
    dHkdky.resize(wcentnum,wcentnum);
    dHkdkz.resize(wcentnum,wcentnum);
    energyweight.resize(wcentnum);
    R.resize(overlapnum,3);
    Rr.resize(overlapnum,3);
    wcent.resize(wcentnum,3);
    energy.resize(wcentnum);
    wavef.resize(wcentnum,wcentnum);
    for (int i = 0; i < 3; i++)
    {
        vnm(i).resize(wcentnum,wcentnum);
        rnm(i).resize(wcentnum,wcentnum);
    }
    
    return 1;
}

int Hamiltoniank::updateH(Eigen::Vector3d kv){
    nowk=kv;
    auto r=Hr.unchecked<3>();
    Hk.setZero();
    dHkdkx.setZero();
    dHkdky.setZero();
    dHkdkz.setZero();
    // std::cout<<"this"<<std::endl;
    for (int i = 0; i < wcentnum; i++)
    {
        for (int j = 0; j < wcentnum; j++)
        {
            for (int k = 0; k < overlapnum; k++)
            {
                
                complex<double> &&refHijk=r(k,i,j) * std::exp<double>( complex<double>(0,1) * (
                    kv(0)*(Rr(k,0))
                    +kv(1)*(Rr(k,1))
                    +kv(2)*(Rr(k,2))
                    ) ) / wtR(k);
                    
                Hk(i,j)+=refHijk;
                if (clcdHdk == 1){
                    dHkdkx(i,j)+= complex<double>(0,1) * refHijk * (Rr(k,0)+wcent(j,0)-wcent(i,0));
                    dHkdky(i,j)+= complex<double>(0,1) * refHijk * (Rr(k,1)+wcent(j,1)-wcent(i,1));
                    dHkdkz(i,j)+= complex<double>(0,1) * refHijk * (Rr(k,2)+wcent(j,2)-wcent(i,2));
                    // dHkdkx(i,j)+= -complex<double>(0,1) * refHijk * (Rr(k,0)+wcent(j,0)-wcent(i,0));
                    // dHkdky(i,j)+= -complex<double>(0,1) * refHijk * (Rr(k,1)+wcent(j,1)-wcent(i,1));
                    // dHkdkz(i,j)+= -complex<double>(0,1) * refHijk * (Rr(k,2)+wcent(j,2)-wcent(i,2));
                }
            }
            // if (i!=j){
            //     Hk(j,i)=std::conj(Hk(i,j));
            //     dHkdkx(j,i)=std::conj(dHkdkx(i,j));
            //     dHkdky(j,i)=std::conj(dHkdky(i,j));
            //     dHkdkz(j,i)=std::conj(dHkdkz(i,j));
            // }
        }
        
    }
    for (int i = 0; i < wcentnum; i++)
    {
        for (int j = 0; j < wcentnum; j++)
        {
            Hk(i,j)=complex<double>(round(real(Hk(i,j))*1000000)/1000000,round(imag(Hk(i,j))*1000000)/1000000);
            dHkdkx(i,j)=complex<double>(round(real(dHkdkx(i,j))*1000000)/1000000,round(imag(dHkdkx(i,j))*1000000)/1000000);
            dHkdky(i,j)=complex<double>(round(real(dHkdky(i,j))*1000000)/1000000,round(imag(dHkdky(i,j))*1000000)/1000000);
            dHkdkz(i,j)=complex<double>(round(real(dHkdkz(i,j))*1000000)/1000000,round(imag(dHkdkz(i,j))*1000000)/1000000);
        }
    }
    return 1;
}

int Hamiltoniank::solverH(){
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hk); //会自动排序好
    //std::cout<<es.eigenvalues()<<std::endl;
    energy=es.eigenvalues().real();
    for (int i = 0; i < wcentnum; i++)
    {   
        if (energyweight(i)<0.5){
            energy(i)+=bandgapadd;
        }
    }
    // std::cout<<energy<<std::endl;
    wavef=es.eigenvectors();//每个特征向量为列向量
    return 0;
}

void Hamiltoniank::updaternm(){
    for (int i = 0; i < wcentnum; i++)
    {
        for (int j = 0; j < wcentnum; j++)
        {
            
            for (int k = 0; k < 3; k++)
            {
                if (abs(energy(i)-energy(j)) <0.0001 ){
                    rnm(k)(i,j)=0;
                }
                else{
                    rnm(k)(i,j)=-complex<double>(0,1)*vnm(k)(i,j)/(energy(i)-energy(j));
                }
            }
            
        }
    }
}

void Hamiltoniank::update_vnmrnm(){
    vnm(0)=wavef.adjoint()*dHkdkx*wavef;
    vnm(1)=wavef.adjoint()*dHkdky*wavef;
    vnm(2)=wavef.adjoint()*dHkdkz*wavef;
    double change=0;
    for (int i = 0; i < wcentnum; i++)
    {
        for (int j = 0; j < wcentnum; j++)
        {
            
            for (int k = 0; k < 3; k++)
            {
                if (energyweight(i)<0.5 && energyweight(j)>0.5){
                    change = bandgapadd;
                }
                else if (energyweight(i)>0.5 && energyweight(j)<0.5){
                    change = -bandgapadd;
                }
                else{
                    change = 0;
                }
                if (i!=j){
                    vnm(k)(i,j)=vnm(k)(i,j)*(energy(i)-energy(j))/(energy(i)-energy(j)-change);
                }
            }
            
        }
    }
    updaternm();
    // std::cout<<rnm(0)(15,0)<<rnm(1)(14,15)<<vnm(1)(14,15)<<std::endl;
}



void Hamiltoniank::runonekpoints(Eigen::Vector3d kv){ 
    // std::cout<<"kv:"<<kv<<std::endl;
    
    //仅在k点不同时重新计算
    if((kv-kvector_now).norm()>0.00000001){
        // std::cout<<"kv:"<<kv<<std::endl;
        kvector_now=kv;
        updateH(kv);
        solverH();
        update_vnmrnm();
    }

}