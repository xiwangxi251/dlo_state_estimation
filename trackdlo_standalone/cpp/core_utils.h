#pragma once

#include "trackdlo_core.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Preserve the official source with ROS logging compiled out. The Python layer
// exposes timings and errors without requiring roscpp.
#define ROS_INFO_STREAM(message) do { } while (0)
#define ROS_INFO(message) do { } while (0)
#define ROS_ERROR(message) do { std::cerr << message << std::endl; } while (0)

double pt2pt_dis_sq(MatrixXd pt1, MatrixXd pt2);
double pt2pt_dis(MatrixXd pt1, MatrixXd pt2);
std::vector<MatrixXd> line_sphere_intersection(
    MatrixXd point_A,
    MatrixXd point_B,
    MatrixXd sphere_center,
    double radius);
