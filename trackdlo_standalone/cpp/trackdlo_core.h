#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <vector>

using Eigen::MatrixXd;

class trackdlo {
public:
    trackdlo();
    explicit trackdlo(int num_of_nodes);
    trackdlo(
        int num_of_nodes,
        double visibility_threshold,
        double beta,
        double lambda,
        double alpha,
        double k_vis,
        double mu,
        int max_iter,
        double tol,
        double beta_pre_proc,
        double lambda_pre_proc,
        double lle_weight);

    double get_sigma2();
    bool get_last_nonconverged();
    MatrixXd get_tracking_result();
    MatrixXd get_guide_nodes();
    std::vector<MatrixXd> get_correspondence_pairs();
    void initialize_geodesic_coord(std::vector<double> geodesic_coord);
    void initialize_nodes(MatrixXd Y_init);
    void set_sigma2(double sigma2);
    void set_adaptive_alpha(double visible_alpha, double occluded_alpha);

    bool cpd_lle(
        MatrixXd X_orig,
        MatrixXd& Y,
        double& sigma2,
        double beta,
        double lambda,
        double lle_weight,
        double mu,
        int max_iter = 30,
        double tol = 0.0001,
        bool include_lle = true,
        std::vector<MatrixXd> correspondence_priors = {},
        double alpha = 0,
        std::vector<int> visible_nodes = {},
        double k_vis = 0,
        double visibility_threshold = 0.01,
        std::vector<double> node_alpha = {});

    void tracking_step(
        MatrixXd X_orig,
        std::vector<int> visible_nodes,
        std::vector<int> visible_nodes_extended,
        MatrixXd proj_matrix,
        int img_rows,
        int img_cols);

private:
    MatrixXd Y_;
    MatrixXd guide_nodes_;
    double sigma2_;
    double beta_;
    double beta_pre_proc_;
    double lambda_;
    double lambda_pre_proc_;
    double alpha_;
    double k_vis_;
    double mu_;
    int max_iter_;
    double tol_;
    double lle_weight_;
    std::vector<double> geodesic_coord_;
    std::vector<MatrixXd> correspondence_priors_;
    double visibility_threshold_;
    // True when the most recent tracking_step completed with a finite
    // iterate but hit max_iter before satisfying the convergence tolerance.
    bool last_nonconverged_;
    bool cpd_last_nonconverged_;
    bool adaptive_alpha_enabled_ = false;
    double adaptive_visible_alpha_ = 0.0;
    double adaptive_occluded_alpha_ = 0.0;

    std::vector<int> get_nearest_indices(int k, int M, int idx);
    MatrixXd calc_LLE_weights(int k, MatrixXd X);
    std::vector<MatrixXd> traverse_geodesic(
        std::vector<double> geodesic_coord,
        const MatrixXd guide_nodes,
        const std::vector<int> visible_nodes,
        int alignment);
    std::vector<MatrixXd> traverse_euclidean(
        std::vector<double> geodesic_coord,
        const MatrixXd guide_nodes,
        const std::vector<int> visible_nodes,
        int alignment,
        int alignment_node_idx = -1);
};
