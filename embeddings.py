import numpy as np
import gudhi as gd
import gudhi.representations as tda
from dotenv import load_dotenv
import psycopg2

load_dotenv(".env")

VERBOSE = os.getenv("VERBOSE").lower.strip(" ") in ["t", "true", "1", "y", "yes"]


# persistence vector
##############################


def pdvec_alpha(data, Nint, epsmax=1):
    """make persistence vector with the top Nint persistences using alpha complex"""
    digs, ints0, ints1 = gettree_int_alpha(data, edgelen=edgelen)
    out1 = _pdproc(ints1, Nint)
    out0 = _pdproc(ints0, Nint)
    return (out0, out1)


def pdvec(data, Nint, epsmax=1):
    """make persistence vector with the top Nint persistences using rips complex"""
    digs, ints0, ints1 = gettree_int(data, edgelen=edgelen)
    out1 = _pdproc(ints1, Nint)
    out0 = _pdproc(ints0, Nint)
    return (out0, out1)


def _pdproc(dim1, Nint):  # sort proc the
    """internal helper function for persistence vectorcalculation"""
    out = np.zeros((len(dim1) + Nint,))
    k = 0
    for d in dim1:
        out[k] = d[1] - d[0]
        k = k + 1
    out = np.isfinite(out) * np.nan_to_num(out)
    out = np.nan_to_num(out)
    out.sort()
    out = np.flip(out)
    return out[0:Nint]


# /persistence vector
###############################

# betti vector
###############################
def betti_vector_alpha(data, Ntda, edgelen=1):
    """make betti vector isong Ntda intervals  using sparse rips complex"""
    epsvec = np.linspace(0, edgelen, Ntda)
    digs, ints0, ints1 = gettree_int_alpha(data, edgelen=edgelen)
    h0 = _betti_vector(digs, epsvec, 0)
    h1 = _betti_vector(digs, epsvec, 1)
    return (h0, h1)


def betti_vector(data, Ntda, edgelen=1):
    """make betti vector isong Ntda intervals using alpha complex"""
    epsvec = np.linspace(0, edgelen, Ntda)
    digs, ints0, intsd1 = gettree_int(data, edgelen=edgelen)
    h0 = _betti_vector(digs, epsvec, 0)
    h1 = _betti_vector(digs, epsvec, 1)
    return (h0, h1)


def _betti_vector(dgs, epsspace, dim):
    ints = []
    for d in dgs:
        if d[0] == dim:
            ints.append(d[1])
    bvec = []
    for e in epsspace:
        betti = 0
        for i in ints:
            betti = betti + int(_input(e, i))
        bvec.append(betti)
    return bvec


def _input(x, tup):
    return x >= tup[0] and x <= tup[1]


# /betti vector
####################################


# get simplex treesssssss
###############################################
def gettree_int(data, edgelen=100, maxdim=2):
    """calculates simplex tree and persistence, returns persistence diagram and
    the intervals in H_0 and H_1"""
    rips = gd.RipsComplex(points=data, max_edge_length=edgelen, sparse=0.3)
    simplex_tree = rips.create_simplex_tree(max_dimension=maxdim)
    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    dim1 = simplex_tree.persistence_intervals_in_dimension(1)
    dim0 = simplex_tree.persistence_intervals_in_dimension(0)
    simplex_tree = []
    return diag, dim0, dim1


def gettree_int_alpha(data, edgelen=100, maxdim=1):
    """calculates simplex tree and persistence of alpha complex, returns
    persistence diagram and the intervals in H_0 and H_1"""
    rips = gd.AlphaComplex(points=data)  # ,sparse=0.3)
    simplex_tree = rips.create_simplex_tree(max_alpha_square=edgelen)
    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    dim1 = simplex_tree.persistence_intervals_in_dimension(1)
    dim0 = simplex_tree.persistence_intervals_in_dimension(0)
    simplex_tree = []
    return diag, dim0, dim1


# /get simplex trees
###############################################


def mkbothvec(data, Ntda, edgelen, Nint):
    """makes betti vector and persistence vector, using computational
    redundance that would be wasted if the functions called separately"""
    epsvec = np.linspace(0, edgelen, Ntda)
    digs, ints0, ints1 = gettree_int(data, edgelen=edgelen)
    out1 = _pdproc(ints1, Nint)
    out0 = _pdproc(ints0, Nint)
    h0 = _betti_vector(digs, epsvec, 0)
    h1 = _betti_vector(digs, epsvec, 1)
    return (h0, h1, out0, out1)


def mkbothvec_alpha(data, Ntda, edgelen, Nint):
    """makes betti vector and persistence vector, using computational
    redundance that would be wasted if the functions called separately, using alpha complex"""
    epsvec = np.linspace(0, edgelen, Ntda)
    digs, ints0, ints1 = gettree_int_alpha(data, edgelen=edgelen)
    out1 = _pdproc(ints1, Nint)
    out0 = _pdproc(ints0, Nint)
    h0 = _betti_vector(digs, epsvec, 0)
    h1 = _betti_vector(digs, epsvec, 1)
    return (h0, h1, out0, out1)


# sliding window functions and dimensional reduction
####################################################
def slidend(data, n):
    """sliding window (data,n)  of enumerable  vector/list/array "data", using window size n"""
    N = len(data)
    cloud = []
    for j in range(N - n):
        c = []
        for i in range(n):
            c.append(data[j + i])
        cloud.append(c)
    return np.matrix(cloud)


def dimred(dat, Nsize, Svals=False, getV=False):
    """perform PCA on the data, each element being a row, projecting onto
    dimension Nsize, optionally returning the singular  values, the right
    singular vectors V"""
    # data elements are each row
    n = np.shape(dat)[1]
    for j in range(n):
        dat[:, j] = dat[:, j] - dat[:, j].mean()
    [U, S, V] = np.linalg.svd(dat, full_matrices=False)
    if VERBOSE:
        print("VERBOSE", S[0:10])
    op = np.matrix(V[0:Nsize, :])
    if not Svals and not getV:
        return dat * op.T
    elif getV and Svals:
        return (dat * op.T, S, V)
    elif not getV:
        return (dat * op.T, S)
    elif not Svals:
        return (dat * op.T, V)


def dimred3(dat):
    """convenience function dimensionally reduce input data, each row being an
    element in some vector space, to dimension 3 using PCA calcualted by the
    SVD"""
    return dimred(dat, 3)


# representation method
#########################################


class sw_rep_embedding:
    """computes a vector of kernelvalues with a set of reference
    persistence diagrams and observed data using the sliced wasserstein
    kernel"""

    def __init__(
        self,
        kernels_points=False,
        simplex=gd.AlphaComplex,
        dimension=1,
        bandwidth=0.1,
        num_directions=10,
        Nrefs=17,
        order=False,
    ):
        self.Nrefs = Nrefs
        self.dimension = dimension
        if kernels_points:
            self.kernels_points = kernels_points  # must be list of lists or
        else:
            self.kernels_points = self.load_laplacian_geodesics(order=order)
        self.simplex = simplex
        self.kernels = self.compute_reference_kerns(
            bandwidth=bandwidth, num_directions=num_directions
        )

    def compute_reference_kerns(self, bandwidth=0.1, num_directions=10):
        """prepare the sliced wasserstein kernels with their respective
        reference PDs, using alpha compex"""
        SW = []
        for frame in self.kernels_points:
            ac_ref = self.simplex(points=frame).create_simplex_tree(
                max_alpha_square=np.sqrt(2)
            )
            ac_ref.persistence()
            SWk = gd.representations.SlicedWassersteinKernel(
                bandwidth=0.10, num_directions=10
            )
            SWk.fit([ac_ref.persistence_intervals_in_dimension(self.dimension)])
            SW.append(SWk)
        return SW

    def load_laplacian_geodesics(self, order=False):
        """load saved reference elements"""
        Xl = np.loadtxt("datasample/referenceX.txt")
        Yl = np.loadtxt("datasample/referenceY.txt")
        X_refs = []
        Npt = 1000
        NN = 300
        for j in range(NN):
            ou = np.concatenate(
                (
                    np.reshape(Xl[0:Npt, j], (Npt, 1)),
                    np.reshape(Yl[0:Npt, j], (Npt, 1)),
                ),
                1,
            )
            X_refs.append(ou)
        X_refs = np.array(X_refs)
        if order:  # takes custom ordering
            if type(order) == list:
                X_refs[order[0 : self.Nrefs]]
            else:
                argz = list(np.loadtxt("orthorder.dat").astype(int))
                return X_refs[argz[0 : self.Nrefs]]
        else:
            return X_refs[0 : self.Nrefs]

    def embed(self, pcloud):
        """calculate alpha complex of pcloud and embed it in a vector space
        formed by computing the sliced wasserstein kernel between its
        persistence diagram and a set of precomputed refererences"""
        features = []
        ac = self.simplex(points=pcloud).create_simplex_tree(
            max_alpha_square=np.sqrt(2)
        )
        ac.persistence()

        for kernel_function in self.kernels:
            features.append(
                kernel_function.transform(
                    [ac.persistence_intervals_in_dimension(self.dimension)]
                )[0][0]
            )
        return features


# /representation method
#########################################
