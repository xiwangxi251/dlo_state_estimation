#include "trackdlo_core.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, module) {
    module.doc() = "ROS-free Python binding for the official TrackDLO C++ core";
    py::class_<trackdlo>(module, "TrackDLOCore")
        .def(
            py::init<int, double, double, double, double, double, double, int, double, double, double, double>(),
            py::arg("num_nodes"),
            py::arg("visibility_threshold"),
            py::arg("beta"),
            py::arg("lambda_"),
            py::arg("alpha"),
            py::arg("k_vis"),
            py::arg("mu"),
            py::arg("max_iter"),
            py::arg("tol"),
            py::arg("beta_pre_proc"),
            py::arg("lambda_pre_proc"),
            py::arg("lle_weight"))
        .def("initialize_nodes", &trackdlo::initialize_nodes)
        .def("initialize_geodesic_coord", &trackdlo::initialize_geodesic_coord)
        .def("set_sigma2", &trackdlo::set_sigma2)
        .def("set_adaptive_alpha", &trackdlo::set_adaptive_alpha,
             py::arg("visible_alpha"), py::arg("occluded_alpha"))
        .def("get_sigma2", &trackdlo::get_sigma2)
        .def("get_last_nonconverged", &trackdlo::get_last_nonconverged)
        .def("get_tracking_result", &trackdlo::get_tracking_result)
        .def("get_guide_nodes", &trackdlo::get_guide_nodes)
        .def("get_correspondence_pairs", &trackdlo::get_correspondence_pairs)
        .def(
            "tracking_step",
            &trackdlo::tracking_step,
            py::arg("points"),
            py::arg("visible_nodes"),
            py::arg("visible_nodes_extended"),
            py::arg("projection_matrix"),
            py::arg("image_rows"),
            py::arg("image_cols"));
}
