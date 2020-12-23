from numpy import asarray_chkfinite
from numpy.linalg import norm
from math import log10

from scipy.linalg.misc import LinAlgError
from scipy.linalg.lapack import get_lapack_funcs

def cho_invert(c, lower=False, overwrite_c=False):
    """Compute the inverse to the Cholesky decomposition of a matrix. The input
    matrix is assumed to have come from cho_factor.
    """
    c1 = asarray_chkfinite(c)
    
    if len(c1.shape) != 2 or c1.shape[0] != c1.shape[1]:
        raise ValueError('expected square matrix')
    overwrite_c = overwrite_c or _datanotshared(c1, c)   
    potri, = get_lapack_funcs(('potri',), (c1,)) 
    c, info = potri(c1, lower=lower, overwrite_c=overwrite_c)
    if info >0:
        raise LinAlgError("%d-th element of the factor is zero" % info)
    if info <0:
        raise LinAlgError("illegal value in %d-th argument of internal potri"% -info)
    return c
            
def _datanotshared(a1,a):
    if a1 is a:
        return False
    else:
        #try comparing data pointers
        try:
            return a1.__array_interface__['data'][0] != a.__array_interface__['data'][0]
        except:
            return True
        
def cond_num(a, aInv):
    result = norm(a,1)*norm(aInv,1)
    if result != 0.0:
        result = log10(result)
    return result