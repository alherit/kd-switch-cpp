#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kds/kds.hpp"


namespace py = pybind11;



PYBIND11_MODULE(pykds, m) {
  m.doc() = "kd-switch: Python interface to c++ implementation";

  py::class_<KDSForest>(m, "KDSForest")
    .def(py::init<size_t,int,size_t,size_t,bool,const vector<FT> &>(), py::arg("ntrees"), py::arg("seed"), py::arg("dim"), py::arg("alpha_label"), py::arg("ctw"), py::arg("theta0"))
    .def("predict_proba", &KDSForest::predict_proba, "Returns the kd-switch predictive distribution for the given point. If frozen=true (default), it doesn't affect the structure.", py::arg("point"), py::arg("frozen") = true)
	.def("predict_log2_proba", &KDSForest::predict_log2_proba, "Returns the kd-switch predictive distribution for the given point in log2. If frozen=true (default), it doesn't affect the structure.", py::arg("point"), py::arg("frozen") = true)
	.def("update", &KDSForest::update, "Updates the kd-switch model. If frozen=false, predict must be called before with the same point. Otherwise use frozen=true (default).", py::arg("point"), py::arg("label"), py::arg("frozen") = true);

  

}
