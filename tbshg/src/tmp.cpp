#include<solveopt.hpp>

Eigen::MatrixXcd &solveopt::get_sigma(Eigen::Vector3d kvector,Eigen::VectorXd hv,Eigen::MatrixX3i directindice){

    const auto & energyeigenval = H1->energy;
    int bandnum=H1->wcentnum;
    int ndirects=directindice.size();
    int nhv=hv.size();
    const auto & rnm=H1->rnm;
    const auto & vnm=H1->vnm;
    const auto & energyweight=H1->energyweight;

    //当前k点sigma存储变量
    Eigen::MatrixXcd sigma(directindice.size(),hv.size());
    sigma.Zero(directindice.size(),hv.size());

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