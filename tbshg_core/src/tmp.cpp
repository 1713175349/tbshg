#include<solveopt.hpp>

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
                continue
            }
            for (int m=0; m < bandnum; m++)
            {
                if (energyweight(m)>0.5){
                    continue
                }
                double && emn=energyeigenval(m)-energyeigenval(n);
                for (int l=0;l<bandnum;l++){
                    if (l==n || l==m){
                        continue
                    }

                    
                   

                    double && eln=energyeigenval(l)-energyeigenval(n);
                    double && eml=energyeigenval(m)-energyeigenval(l);
                    //去掉共振峰
                    if (abs(eln-eml)<etolerance || abs(emn+eln)<etolerance || abs(emn+eml)<etolerance || abs(eln)<etolerance || abs(emn)<etolerance || abs(eml)<etolerance){
                        continue
                    }
                    complex<double> && rbc=rnm(c)(m,l)*rnm(b)(l,n)+rnm(b)(m,l)*rnm(c)(l,n);
                    complex<double> && rab=rnm(a)(m,n)*rnm(b)(n,l)+rnm(b)(m,n)*rnm(a)(n,l);
                    complex<double> && rca=rnm(c)(l,m)*rnm(a)(m,n)+rnm(a)(l,m)*rnm(c)(m,n);
                    fac1=2*rnm(a)(n,m)*rbc/(eln-eml);
                    fac2=-(rnm(c)(l,m)*rab/(-eln-emn)-rnm(b)(n,l)*rca/(-eml-emn));
                    for (int ihv=0;ihv<nhv;ihv++){
                        complex<double> && hv0=hv(ihv)+complex<double>(0,1)*ksi;
                        inter(idirect,ihv)+=fac1/(emn-2*hv0)+fac2/(emn-hv0);
                    }
                }
            }
        }
    }
    return inter;

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
                continue
            }
            for (int m=0; m < bandnum; m++)
            {
                if (energyweight(m)>0.5){
                    continue
                }
                double && emn=energyeigenval(m)-energyeigenval(n);
                for (int l=0;l<bandnum;l++){
                    if (l==n || l==m){
                        continue
                    }

                    
                   

                    double && eln=energyeigenval(l)-energyeigenval(n);
                    double && eml=energyeigenval(m)-energyeigenval(l);
                    //去掉共振峰
                    if (abs(eln-eml)<etolerance || abs(emn+eln)<etolerance || abs(emn+eml)<etolerance || abs(eln)<etolerance || abs(emn)<etolerance || abs(eml)<etolerance){
                        continue
                    }
                    complex<double> && rbc=rnm(c)(m,l)*rnm(b)(l,n)+rnm(b)(m,l)*rnm(c)(l,n);
                    complex<double> && rab=rnm(a)(m,n)*rnm(b)(n,l)+rnm(b)(m,n)*rnm(a)(n,l);
                    complex<double> && rca=rnm(c)(l,m)*rnm(a)(m,n)+rnm(a)(l,m)*rnm(c)(m,n);
                    complex<double> && rdbc=(vnm(b)(m,m)-vnm(b)(n,n))*rnm(c)(l,n)+(vnm(c)(m,m)-vnm(c)(n,n))*rnm(b)(l,n);
                    fac1=-complex<double>(0,8)*(rnm(a)(n,m)*rdbc)/emn/emn+2*rnm(a)(n,m)*rbc*(eml-eln)/emn/emn;
                    fac2=(eln*rnm(b)(n,l)*rca-eml*rnm(c)(l,m)*rab)/emn/emn;
                    for (int ihv=0;ihv<nhv;ihv++){
                        complex<double> && hv0=hv(ihv)+complex<double>(0,1)*ksi;
                        intra(idirect,ihv)+=fac1/(emn-2*hv0)+fac2/(emn-hv0);
                    }
                }
            }
        }
    }
    return intra;

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
                continue
            }
            for (int m=0; m < bandnum; m++)
            {
                if (energyweight(m)>0.5){
                    continue
                }
                double && emn=energyeigenval(m)-energyeigenval(n);
                for (int l=0;l<bandnum;l++){
                    if (l==n || l==m){
                        continue
                    }

                    
                   

                    double && eln=energyeigenval(l)-energyeigenval(n);
                    double && eml=energyeigenval(m)-energyeigenval(l);
                    //去掉共振峰
                    if (abs(eln-eml)<etolerance || abs(emn+eln)<etolerance || abs(emn+eml)<etolerance || abs(eln)<etolerance || abs(emn)<etolerance || abs(eml)<etolerance){
                        continue
                    }
                    complex<double> && rbc0=rnm(c)(m,n)*rnm(b)(n,l)+rnm(b)(m,n)*rnm(c)(n,l);
                    complex<double> && rbc1=rnm(c)(l,m)*rnm(b)(m,n)+rnm(b)(l,m)*rnm(c)(m,n);
                    // complex<double> && rab=rnm(a)(m,n)*rnm(b)(n,l)+rnm(b)(m,n)*rnm(a)(n,l);
                    // complex<double> && rca=rnm(c)(l,m)*rnm(a)(m,n)+rnm(a)(l,m)*rnm(c)(m,n);
                    // complex<double> && rdbc=(vnm(b)(m,m)-vnm(b)(n,n))*rnm(c)(l,n)+(vnm(c)(m,m)-vnm(c)(n,n))*rnm(b)(l,n);
                    complex<double> && rdbc0=(vnm(c)(m,m)-vnm(c)(n,n))*rnm(b)(m,n)+(vnm(b)(m,m)-vnm(b)(n,n))*rnm(c)(m,n);
                    fac1=0;
                    fac2=(-enl*rnm(a)(l,m)*rbc0-(-elm)*rnm(a)(n,l)*rbc1)/emn/emn-complex(0,1)*(rnm(a)(n,m)*rdbc0)/emn/emn;
                    for (int ihv=0;ihv<nhv;ihv++){
                        complex<double> && hv0=hv(ihv)+complex<double>(0,1)*ksi;
                        modu(idirect,ihv)+=fac1/(emn-2*hv0)-fac2/(emn-hv0);
                    }
                }
            }
        }
    }
    return modu;

}




