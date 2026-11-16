## constants
FEAT_X = [
    "n_nodes",
    "n_edges",
    "n_articulation_points",
    "n_bridges",
    "diameter",
    "radius",
    "degeneracy",
    "k_core_size",
    "maximum_degree",
    "degree_variance",
    "global_clustering",
    "degree_assortativity",
    "degree_entropy",
    "joint_degree_entropy",
    "degree_skewness",
    "normalized_laplacian_second_moment",
    "normalized_laplacian_third_moment",
    "random_walk_triangle_weight",
    "random_walk_fourth_moment",
    "adjacency_fourth_moment_per_node",
    "degree_kurtosis"
]

FEAT_Z = [
    "lag1_autocorr",
    "coef_variation",
    "fano_factor",
    "norm_succ_diff",
    "hurst_exponent",
    "trend_coeff",
    "count_skewness",
]

FEAT_XU = [
    "n_edges",
    "degree_entropy",
    "random_walk_triangle_weight",
]

FEAT_ZU = [
    "norm_succ_diff",
    "coef_variation",
    "trend_coeff",
]

TARGET = "target"
FRONTIER_METRICS = ["vr", "mv", "ms", "ei"]
CONSENSUS_METRICS = ["rho", "rbo", "dcr", "ci"]

