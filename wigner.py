import numpy as np

pi = np.pi
sqrt = np.sqrt


def get_matrix(dim):
    '''
    Parameters:
        dim <int> dimension of matrix
    Return:
        m <array((dim, <float>), (dim, <float>))>
    '''
    m = np.random.standard_normal((dim, dim))
    m = m + m.T
    return m


def get_ensemble(num, dim):
    '''
    Parameters:
        num <int>  size of ensemble
        dim <int> dimension of matrix
    Return:
        ensemble <list(num, <array((dim, <float>), (dim, <float>))>)>
    '''
    ensemble = map(lambda i: get_matrix(dim), range(num))
    return ensemble


def get_center_eigenvalues_differences(ensemble, dim):
    '''
    Parameters:
        ensemble <list(num, <array((dim, <float>), (dim, <float>))>)>
        dim <int> dimension of matrix
    Return:
        center_list <list(num, <float>)>
    '''
    num_matrices = len(ensemble)
    center_list = np.array([])
    for m in ensemble:
        vals = np.sort(np.linalg.eigvals(m))
        center_list = np.append(center_list, [vals[dim//2] - vals[dim//2-1]])

    return center_list

get_wigner = lambda l: (np.pi*l/2)*np.exp(-np.pi*l**2./4.)


def get_semicircle_law(l, sigma=1):
    '''
    Parameters:
        dim <int> dimension of matrix
    Return:
        m <array((dim, <float>), (dim, <float>))>
    '''
    if abs(l) <= 2.*sigma:
        return sqrt(4.*sigma**2. - l**2.)/(pi*2*sigma**2.)
    else:
        return 0