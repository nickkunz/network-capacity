## libraries
import logging
import igraph as ig
import numpy as np
import scipy.stats as stats
from scipy.sparse import csr_matrix, diags

## modules
from src.data.helpers import _force_finite

## logging
logger = logging.getLogger(__name__)

## compute graph invariant vector
class GraphInvariants:
    """
    Desc:
        Computes true graph-wise global invariants for igraph.Graph objects without 
        aggregating individual node or edge properties. Only includes pure classical 
        global features. Guarantees returned values are finite (no inf, -inf, or nan).

    Args:
        graph (igraph.Graph): An igraph object representing the graph.

    Returns:
        dict: Each method returns a dictionary containing the extracted graph-level 
        features. All values are guaranteed to be finite.

    Raises:
        TypeError: If the input is not an igraph.Graph object.
    """
    ## init input
    def __init__(self, graph):
        if not isinstance(graph, ig.Graph):
            raise TypeError("Input graph must be an igraph.Graph object")
        self.graph = graph
        self._degree = None

    ## compute simple (linear-time) graph invariants
    def simple(self) -> dict:
        graph = self.graph
        features = {}

        ## number of nodes and edges (cardinality of graph)
        features['n_nodes'] = graph.vcount()
        features['n_edges'] = graph.ecount()

        ## articulation points (vertex cut set of size 1)
        articulation_points = graph.articulation_points()
        features['n_articulation_points'] = len(articulation_points)

        ## bridges (edges whose removal increases number of components)
        bridges = graph.bridges()
        features['n_bridges'] = len(bridges)

        ## simple pure graph invariants
        return features

    ## compute cohesion (cubic-time) graph invariants
    def cohesion(self) -> dict:
        graph = self.graph
        features = {}

        ## operate on the largest connected component for canonical diameter/radius on disconnected graphs
        H = graph.components().giant() if graph.vcount() > 0 else graph
        if H.vcount() < 2:
            return {
                'diameter': 0.0,
                'radius': 0.0,
                'degeneracy': 0.0,
                'k_core_size': graph.vcount()
            }

        ## exact diameter and radius from one eccentricity pass
        try:
            ecc = np.array(H.eccentricity())
            finite_ecc = ecc[np.isfinite(ecc)]
            features['diameter'] = _force_finite(float(finite_ecc.max()), 0.0) if finite_ecc.size > 0 else 0.0
            features['radius'] = _force_finite(float(finite_ecc.min()), 0.0) if finite_ecc.size > 0 else 0.0
        except ig.InternalError:
            features['diameter'] = 0.0
            features['radius'] = 0.0

        ## degeneracy (largest core number)
        core_nums = graph.coreness()
        if core_nums:
            d = max(core_nums)
            features['degeneracy'] = d
            features['k_core_size'] = core_nums.count(d)
        else:
            features['degeneracy'] = 0.0
            features['k_core_size'] = 0.0

        return features

    ## compute extremal graph invariants
    def extremal(self) -> dict:
        features = {}
        
        ## maximum degree (extremum of degree sequence)
        if self._degree is None:
            self._degree = self.graph.degree()
        degrees = self._degree
        features['maximum_degree'] = max(degrees) if degrees else 0.0

        ## extremal pure graph invariant
        return features

    ## compute statistical graph invariants
    def statistical(self) -> dict:
        graph = self.graph
        features = {}

        ## degree sequence variance
        if self._degree is None:
            self._degree = self.graph.degree()
        degrees = self._degree
        features['degree_variance'] = np.var(degrees) if degrees else 0.0

        ## global clustering coefficient (fraction of triangles among connected triples)
        clustering = graph.transitivity_undirected()
        features['global_clustering'] = _force_finite(clustering, 0.0)

        ## degree assortativity (correlation of connected node degrees)
        assortativity = graph.assortativity_degree(directed = False)
        features['degree_assortativity'] = _force_finite(assortativity, 0.0)

        ## shannon entropy of degree sequence
        if len(degrees) > 0:
            degree_counts = np.bincount(degrees)
            degree_probs = degree_counts[degree_counts > 0] / len(degrees)
            features['degree_entropy'] = -np.sum(degree_probs * np.log(degree_probs + 1e-16))
        else:
            features['degree_entropy'] = 0.0

        ## joint degree entropy (unordered degree pairs across undirected edges)
        if graph.ecount() > 0:
            deg = np.array(degrees)
            ## make each edge’s degree pair unordered by sorting the pair
            sources, targets = np.array(graph.get_edgelist(), dtype=int).T
            deg_pairs = np.sort(np.column_stack((deg[sources], deg[targets])), axis=1)
            pairs, counts = np.unique(deg_pairs, axis = 0, return_counts = True)
            probs = counts / counts.sum()
            features['joint_degree_entropy'] = -np.sum(probs * np.log(probs + 1e-16))
        else:
            features['joint_degree_entropy'] = 0.0

        ## degree skewness (fisher–pearson, bias-corrected)
        if len(degrees) < 3 or len(set(degrees)) <= 1:
            features['degree_skewness'] = 0.0
        else:
            features['degree_skewness'] = _force_finite(stats.skew(degrees, bias = False), 0.0)

        ## degree kurtosis (fisher, bias-corrected)
        if len(degrees) < 4 or len(set(degrees)) <= 1:
            features['degree_kurtosis'] = 0.0
        else:
            features['degree_kurtosis'] = _force_finite(stats.kurtosis(degrees, fisher = True, bias = False), 0.0)

        return features

    ## compute spectral invariants (block-streamed trace identities, no eigendecomposition)
    def spectral(self, block_size: int = 1024) -> dict:
        graph = self.graph
        features = {}
        n_nodes = graph.vcount()
        
        if n_nodes < 2:
            return {
                'normalized_laplacian_second_moment': 0.0,
                'normalized_laplacian_third_moment': 0.0,
                'random_walk_triangle_weight': 0.0,
                'random_walk_fourth_moment': 0.0,
                'adjacency_fourth_moment_per_node': 0.0,
            }
        
        ## precompute degrees and adjacency
        degrees = np.array(graph.degree(), dtype = float)
        n_edges = graph.ecount()
        edges = graph.get_edgelist()

        if n_edges == 0:
            return {
                'normalized_laplacian_second_moment': 0.0,
                'normalized_laplacian_third_moment': 0.0,
                'random_walk_triangle_weight': 0.0,
                'random_walk_fourth_moment': 0.0,
                'adjacency_fourth_moment_per_node': 0.0,
            }
        
        ## adjacency as csr for row-block products
        A = csr_matrix(graph.get_adjacency_sparse(), dtype = float)

        ## second moment from edge-wise degree products
        edge_array = np.array(edges, dtype = int)
        sum_inv = np.sum(
            1.0 / (degrees[edge_array[:, 0]] * degrees[edge_array[:, 1]])
        )
        n_active = int(np.count_nonzero(degrees))
        features['normalized_laplacian_second_moment'] = _force_finite(
            (n_active + 2.0 * sum_inv) / n_nodes, 0.0
        )

        ## symmetric normalization: tr(p^k) = tr(a_hat^k) exactly
        d_inv_sqrt_values = np.zeros_like(degrees)
        np.divide(1.0, np.sqrt(degrees), out = d_inv_sqrt_values, where = degrees > 0)
        d_inv_sqrt = diags(d_inv_sqrt_values)
        a_hat = (d_inv_sqrt @ A @ d_inv_sqrt).tocsr()

        ## block traces for symmetric m: tr(m^3) = <m^2, m> and tr(m^4) = ||m^2||^2
        trace_ah3 = 0.0
        trace_ah4 = 0.0
        for start in range(0, n_nodes, block_size):
            stop = min(start + block_size, n_nodes)
            block = a_hat[start:stop] @ a_hat
            trace_ah3 += float(block.multiply(a_hat[start:stop]).sum())
            trace_ah4 += float(block.multiply(block).sum())

        features['normalized_laplacian_third_moment'] = _force_finite(
            (n_active + 6.0 * sum_inv - trace_ah3) / n_nodes, 0.0
        )
        features['random_walk_triangle_weight'] = _force_finite(
            trace_ah3 / n_nodes, 0.0
        )
        features['random_walk_fourth_moment'] = _force_finite(
            trace_ah4 / n_nodes, 0.0
        )

        ## adjacency fourth moment via the same block identity
        trace_a4 = 0.0
        for start in range(0, n_nodes, block_size):
            stop = min(start + block_size, n_nodes)
            block = A[start:stop] @ A
            trace_a4 += float(block.multiply(block).sum())
        features['adjacency_fourth_moment_per_node'] = _force_finite(
            trace_a4 / n_nodes, 0.0
        )

        return features

    ## compute all invariants and ensure all are finite
    def all(self, analytical: bool = False):

        ## compute invariants analytically for fully connected bipartite graphs
        ## (much faster than numerical computation for large dense bipartite graphs)
        if analytical:
            bipartite, types = self.graph.is_bipartite(return_types = True)
            if not bipartite or types is None:
                raise ValueError("Analytical mode requires a bipartite graph")
            
            ## count nodes in each partition
            n1 = sum(types)
            n2 = len(types) - n1
            
            ## quality check: ensure it's fully connected bipartite
            if self.graph.ecount() != n1 * n2:
                raise ValueError("Analytical mode expects a fully connected bipartite graph")
            return BipartiteInvariants(n1, n2).all()
        
        ## compute all invariants numerically
        else:
            features = dict()
            features.update(self.simple())
            features.update(self.cohesion())
            features.update(self.extremal())
            statistical = self.statistical()
            degree_kurtosis = statistical.pop('degree_kurtosis')
            features.update(statistical)
            features.update(self.spectral())
            features['degree_kurtosis'] = degree_kurtosis
            
        ## final results check - should never trigger with proper implementation
        for key, value in features.items():
            if not np.isfinite(value):
                raise ValueError(f"Feature '{key}' is non-finite ({value}).")

        return features


## compute graph invariants for fully connected bipartite graph
class BipartiteInvariants:
    """
    Desc:
        Computes analytic graph-wise invariants for a complete bipartite graph
        K_{m,n}. This uses direct mathematical formulas for efficiency and does 
        not require constructing a graph object. It computes the same set of 
        invariants as the GraphInvariants class.

    Args:
        m (int): The number of vertices in the first partition.
        n (int): The number of vertices in the second partition.
    
    Returns:
        dict: Each method returns a dictionary containing the graph-level features.
    """
    
    ## init input
    def __init__(self, m: int, n: int):
        if not isinstance(m, int) or not isinstance(n, int):
            raise TypeError("Inputs m and n must be integers.")
        if m < 0 or n < 0:
            raise ValueError("Inputs m and n must be non-negative.")
        self.m = m
        self.n = n
        self.is_trivial = (self.m == 0 or self.n == 0)
        self.is_star = (self.m == 1 and self.n > 1) or (self.n == 1 and self.m > 1)
        self.is_edge = (self.m == 1 and self.n == 1)
    
    ## compute simple bipartite invariants
    def simple(self) -> dict:
        if self.is_trivial:
            return {'n_nodes': self.m + self.n, 'n_edges': 0, 'n_articulation_points': 0, 'n_bridges': 0}
        
        ## articulation points: 1 if it's a star graph (k_1,n with n>1), otherwise 0.
        n_articulation = 1 if self.is_star else 0
        
        ## bridges: n if it's a star graph (k_1,n), 1 if it's a single edge (k_1,1), otherwise 0.
        n_bridges = self.n if self.m == 1 else (self.m if self.n == 1 else 0)

        return {
            'n_nodes': self.m + self.n, 
            'n_edges': self.m * self.n,
            'n_articulation_points': n_articulation,
            'n_bridges': n_bridges
        }

    ## compute cohesion bipartite invariants
    def cohesion(self) -> dict:
        if self.is_trivial:
            return {'diameter': 0, 'radius': 0, 'degeneracy': 0, 'k_core_size': self.m + self.n}
        
        ## diameter: 1 for a single edge (k_1,1), 2 for star graphs and other k_m,n.
        diameter = 1 if self.is_edge else 2
        
        ## radius: 1 for single edge or star graphs, 2 otherwise.
        radius = 1 if self.is_edge or self.is_star else 2
        
        degeneracy = min(self.m, self.n)
        
        ## k-core size: the k-core (for k=min(m,n)) is the entire graph.
        k_core_size = self.m + self.n

        return {
            'diameter': diameter, 
            'radius': radius, 
            'degeneracy': degeneracy,
            'k_core_size': k_core_size
        }

    ## compute extremal bipartite invariants
    def extremal(self) -> dict:
        if self.is_trivial:
            return {'maximum_degree': 0}
        return {'maximum_degree': max(self.m, self.n)}

    ## compute statistical bipartite invariants
    def statistical(self) -> dict:
        if self.is_trivial:
            return {
                'degree_variance': 0.0,
                'global_clustering': 0.0,
                'degree_assortativity': 0.0,
                'degree_entropy': 0.0,
                'joint_degree_entropy': 0.0,
                'degree_skewness': 0.0,
                'degree_kurtosis': 0.0
            }
        
        N = self.m + self.n
        M = self.m * self.n
        
        ## degree variance
        mean_k = (2 * M) / N
        deg_var = ((self.m * (self.n ** 2) + self.n * (self.m ** 2)) / N) - (mean_k ** 2)
        
        ## degree assortativity
        assortativity = -1.0 if self.m != self.n else 0.0
        
        ## degree entropy
        p_m = self.m / N  ## probability of having degree n
        p_n = self.n / N  ## probability of having degree m
        degree_entropy = - (p_m * np.log(p_m + 1e-16) + p_n * np.log(p_n + 1e-16)) if self.m != self.n else 0.0
        
        ## joint degree entropy: all edges connect a degree-m node to a degree-n node.
        ## there is only one type of edge, so probability is 1. log(1) = 0.
        joint_degree_entropy = 0.0
        
        ## degree skewness
        if self.m == self.n:
            skewness = 0.0
            kurtosis = 0.0
        else:
            biased_skewness = abs(self.m - self.n) / np.sqrt(self.m * self.n)
            skewness = (
                np.sqrt(N * (N - 1)) / (N - 2) * biased_skewness
                if N >= 3 else 0.0
            )
            biased_kurtosis = (N ** 2) / (self.m * self.n) - 6.0
            kurtosis = (
                (N - 1) / ((N - 2) * (N - 3))
                * ((N + 1) * biased_kurtosis + 6.0)
                if N >= 4 else 0.0
            )

        return {
            'degree_variance': _force_finite(float(deg_var)),
            'global_clustering': 0.0,
            'degree_assortativity': _force_finite(assortativity),
            'degree_entropy': _force_finite(float(degree_entropy)),
            'joint_degree_entropy': _force_finite(float(joint_degree_entropy)),
            'degree_skewness': _force_finite(float(skewness)),
            'degree_kurtosis': _force_finite(float(kurtosis))
        }

    ## compute spectral invariants (trace-polynomial, no eigendecomposition)
    def spectral(self) -> dict:

        ## trivial cases: no edges
        if self.is_trivial:
            return {
                'normalized_laplacian_second_moment': 0.0,
                'normalized_laplacian_third_moment': 0.0,
                'random_walk_triangle_weight': 0.0,
                'random_walk_fourth_moment': 0.0,
                'adjacency_fourth_moment_per_node': 0.0,
            }
        
        ## non-trivial case: K_{m,n}
        m, n = self.m, self.n
        N = m + n

        ## normalized laplacian second moment: tr(L^2)/N = [4 + (N-2)*1]/N = (N+2)/N
        nl2 = 1.0 + 2.0 / N

        ## normalized laplacian third moment: tr(L^3)/N = [8 + (N-2)*1]/N = (N+6)/N
        nl3 = 1.0 + 6.0 / N

        ## random-walk triangle weight: tr(P^3)/N = 0 for bipartite graphs
        rw3 = 0.0

        ## random-walk fourth moment: tr(P^4)/N
        rw4 = 2.0 / N if m > 0 and n > 0 else 0.0

        ## adjacency fourth moment per node: tr(A^4)/N
        a4 = 2.0 * (m**2) * (n**2) / N

        return {
            'normalized_laplacian_second_moment': _force_finite(float(nl2), 0.0),
            'normalized_laplacian_third_moment': _force_finite(float(nl3), 0.0),
            'random_walk_triangle_weight': _force_finite(float(rw3), 0.0),
            'random_walk_fourth_moment': _force_finite(float(rw4), 0.0),
            'adjacency_fourth_moment_per_node': _force_finite(float(a4), 0.0),
        }

    ## compute all bipartite invariants
    def all(self) -> dict:
        if self.is_trivial:
            return {
                'n_nodes': self.m + self.n,
                'n_edges': 0,
                'n_articulation_points': 0,
                'n_bridges': 0,
                'diameter': 0,
                'radius': 0,
                'degeneracy': 0,
                'k_core_size': self.m + self.n,
                'maximum_degree': 0,
                'degree_variance': 0.0,
                'global_clustering': 0.0,
                'degree_assortativity': 0.0,
                'degree_entropy': 0.0,
                'joint_degree_entropy': 0.0,
                'degree_skewness': 0.0,
                'normalized_laplacian_second_moment': 0.0,
                'normalized_laplacian_third_moment': 0.0,
                'random_walk_triangle_weight': 0.0,
                'random_walk_fourth_moment': 0.0,
                'adjacency_fourth_moment_per_node': 0.0,
                'degree_kurtosis': 0.0,
            }
        features = {}
        features.update(self.simple())
        features.update(self.cohesion())
        features.update(self.extremal())
        statistical = self.statistical()
        degree_kurtosis = statistical.pop('degree_kurtosis')
        features.update(statistical)
        features.update(self.spectral())
        features['degree_kurtosis'] = degree_kurtosis

        for k, v in features.items():
            if not np.isfinite(v):
                raise ValueError(f"Feature '{k}' is non-finite ({v}).")
        return features

"""
Graph Invariants
----------------
These statistics characterize the structural properties of an undirected
graph. They are grouped into simple, cohesion, extremal, statistical, and
spectral categories. All values are guaranteed finite.
-----------
n_nodes : int
    Number of vertices in the graph.

n_edges : int
    Number of edges in the graph.

n_articulation_points : int
    Number of cut vertices whose removal disconnects the graph.

n_bridges : int
    Number of edges whose removal increases the number of connected
    components.

diameter : float
    Longest shortest-path distance among all vertex pairs in the largest
    connected component.

radius : float
    Minimum eccentricity among vertices in the largest connected component.

degeneracy : float
    Largest core number in the k-core decomposition of the graph.

k_core_size : float
    Number of vertices belonging to the maximum k-core.

maximum_degree : float
    Largest vertex degree in the graph.

degree_variance : float
    Variance of the degree sequence, measuring heterogeneity in vertex
    connectivity.

global_clustering : float
    Fraction of closed triangles among connected triples, quantifying
    local triadic closure.

degree_assortativity : float
    Pearson correlation of degrees across connected vertex pairs. Positive
    values indicate assortative mixing, negative values disassortative.

degree_entropy : float
    Shannon entropy of the degree distribution, capturing diversity of
    vertex connectivity patterns.

joint_degree_entropy : float
    Shannon entropy of unordered degree pairs across edges, measuring
    diversity of edge-level connectivity patterns.

degree_skewness : float
    Fisher-Pearson skewness of the degree sequence, indicating asymmetry
    in the degree distribution.

degree_kurtosis : float
    Excess kurtosis of the degree sequence, indicating heavy-tailedness
    relative to a normal distribution.

normalized_laplacian_second_moment : float
    Second spectral moment of the normalized Laplacian, computed via
    trace of its square divided by the number of nodes.

normalized_laplacian_third_moment : float
    Third spectral moment of the normalized Laplacian, computed via
    trace of its cube divided by the number of nodes.

random_walk_triangle_weight : float
    Trace of the third power of the random-walk transition matrix divided
    by the number of nodes, measuring triangle density weighted by degree.

random_walk_fourth_moment : float
    Trace of the fourth power of the random-walk transition matrix divided
    by the number of nodes, capturing higher-order return probabilities.

adjacency_fourth_moment_per_node : float
    Trace of the fourth power of the adjacency matrix divided by the
    number of nodes, counting closed walks of length four per vertex.
"""
