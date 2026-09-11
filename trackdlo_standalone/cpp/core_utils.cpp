#include "core_utils.h"

double pt2pt_dis_sq(MatrixXd pt1, MatrixXd pt2) {
    return (pt1 - pt2).rowwise().squaredNorm().sum();
}

double pt2pt_dis(MatrixXd pt1, MatrixXd pt2) {
    return (pt1 - pt2).rowwise().norm().sum();
}

static bool is_between(MatrixXd point, MatrixXd start, MatrixXd end) {
    for (int axis = 0; axis < 3; ++axis) {
        const double low = std::min(start(0, axis), end(0, axis)) - 0.0001;
        const double high = std::max(start(0, axis), end(0, axis)) + 0.0001;
        if (point(0, axis) < low || point(0, axis) > high) {
            return false;
        }
    }
    return true;
}

std::vector<MatrixXd> line_sphere_intersection(
    MatrixXd point_A,
    MatrixXd point_B,
    MatrixXd sphere_center,
    double radius) {
    std::vector<MatrixXd> intersections;
    const double a = pt2pt_dis_sq(point_A, point_B);
    if (a <= 1e-16) {
        return intersections;
    }
    const double b = 2.0 * (
        (point_B(0, 0) - point_A(0, 0)) * (point_A(0, 0) - sphere_center(0, 0)) +
        (point_B(0, 1) - point_A(0, 1)) * (point_A(0, 1) - sphere_center(0, 1)) +
        (point_B(0, 2) - point_A(0, 2)) * (point_A(0, 2) - sphere_center(0, 2)));
    const double c = pt2pt_dis_sq(point_A, sphere_center) - std::pow(radius, 2);
    const double delta = std::pow(b, 2) - 4.0 * a * c;
    if (delta < 0.0) {
        return intersections;
    }
    const double sqrt_delta = std::sqrt(delta);
    const double roots[] = {(-b + sqrt_delta) / (2.0 * a), (-b - sqrt_delta) / (2.0 * a)};
    const int root_count = delta == 0.0 ? 1 : 2;
    for (int index = 0; index < root_count; ++index) {
        MatrixXd point = point_A + roots[index] * (point_B - point_A);
        if (is_between(point, point_A, point_B)) {
            intersections.push_back(point);
        }
    }
    return intersections;
}
