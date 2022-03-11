#include<pybind11/pybind11.h>
#include<Eigen/Dense>
#include<pybind11/numpy.h>
#include<pybind11/eigen.h>
#include<complex>
#include<iostream>
#include<memory>
#include"hamiltoniank.hpp"
#include"solveopt.hpp"
// #include"solveshg.hpp"
namespace py=pybind11;

// class ar1{
//     public:
//         py::array_t<std::complex<double>> a1;
//         std::complex<double> showa1(){
//             auto r=a1.unchecked<1>();
//             return r(1);
//         }
//         Eigen::MatrixXd a12;
// };

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>,true);

PYBIND11_MODULE(tbshg_core,m){


    // py::class_<ar1>(m,"ar1")
    //     .def(py::init())
    //     .def_readwrite("a1",&ar1::a1,py::return_value_policy::automatic)
    //     .def_readwrite("a12",&ar1::a12,py::return_value_policy::automatic)
    //     .def("showa1",&ar1::showa1);
    py::class_<Hamiltoniank,std::shared_ptr<Hamiltoniank>>(m,"Hamiltoniank")
        .def(py::init<>(/*[](){
            return std::make_shared<Hamiltoniank>();
        }*/))
        .def("updateH",&Hamiltoniank::updateH)
        .def("setup",&Hamiltoniank::setup)
        .def("solverH",&Hamiltoniank::solverH)
        .def("update_vnmrnm",&Hamiltoniank::update_vnmrnm)
        .def_readwrite("overlapnum",&Hamiltoniank::overlapnum)
        .def_readwrite("wcentnum",&Hamiltoniank::wcentnum)
        .def_readwrite("Hr",&Hamiltoniank::Hr,py::return_value_policy::reference)
        .def_readwrite("R",&Hamiltoniank::R)
        .def_readwrite("lat",&Hamiltoniank::lat)
        .def_readwrite("Rr",&Hamiltoniank::Rr)
        .def_readwrite("wtR",&Hamiltoniank::wtR)
        .def_readwrite("wcent",&Hamiltoniank::wcent)
        .def_readonly("nowk",&Hamiltoniank::nowk)
        .def_readwrite("energyweight",&Hamiltoniank::energyweight)
        .def_readwrite("energy",&Hamiltoniank::energy)
        .def("getvnm",&Hamiltoniank::getvnm)
        .def("setvnm",&Hamiltoniank::setvnm)
        .def("updaternm",&Hamiltoniank::updaternm)
        .def("getrnm",&Hamiltoniank::getrnm)
        .def("info",&Hamiltoniank::informations)
        .def_readwrite("clcdHdk",&Hamiltoniank::clcdHdk)
        .def_readwrite("wavefunction",&Hamiltoniank::wavef)
        .def_readwrite("Hk",&Hamiltoniank::Hk)
        .def_readwrite("dHkdkx",&Hamiltoniank::dHkdkx)
        .def_readwrite("dHkdky",&Hamiltoniank::dHkdky)
        .def_readwrite("dHkdkz",&Hamiltoniank::dHkdkz)
        .def_readwrite("bandgapadd",&Hamiltoniank::bandgapadd)
        .def("runonekpoint",&Hamiltoniank::runonekpoints);
    
    

    // py::class_<solveshg>(m,"solveshg")
    //     .def(py::init<>())
    //     .def("setup",&solveshg::setup)
    //     .def("updatek",&solveshg::updatek)
    //     .def_readwrite("H1",&solveshg::H1)
    //     .def_readwrite("kmesh",&solveshg::kmesh)
    //     .def_readwrite("kweight",&solveshg::kweight)
    //     .def_readwrite("directs",&solveshg::directindice)
    //     .def_readwrite("hv",&solveshg::hv)
    //     .def_readwrite("nkpts",&solveshg::nkpts)
    //     .def_readwrite("ndirects",&solveshg::ndirects)
    //     .def_readwrite("nhv",&solveshg::nhv)
    //     .def("getshg",&solveshg::getshg)
    //     .def("getchi",&solveshg::getchi)
    //     .def("getita",&solveshg::getita)
    //     .def("updatebyvaspH",&solveshg::updatebyvaspH)
    //     .def_readwrite("ksi",&solveshg::ksi)
    //     .def("getsigma",&solveshg::getsigma);

    py::class_<solveopt,std::shared_ptr<solveopt>>(m,"solveopt")
        .def(py::init<>())
        .def(py::init([](std::shared_ptr<Hamiltoniank> H1_,double ksi_=0.02){
            return std::make_shared<solveopt>(H1_,ksi_);
        })
        )
        .def("get_shg",&solveopt::get_shg)
        .def("get_chi",&solveopt::get_chi)
        .def("get_ita",&solveopt::get_ita)
        .def("get_sigma",&solveopt::get_sigma)
        .def("get_linechi",&solveopt::get_linechi)
        .def_property("ksi",&solveopt::get_ksi,&solveopt::set_ksi)
        .def_property("H1",&solveopt::get_H1,&solveopt::set_H1)
        .def("setup_mc",&solveopt::setup_mc)
        .def("get_shg_mc",&solveopt::get_shg_mc)
        .def("get_chi_mc",&solveopt::get_chi_mc)
        .def("get_ita_mc",&solveopt::get_ita_mc)
        .def("get_sigma_mc",&solveopt::get_sigma_mc);
}