# cython: language_level=3
"""
Cython-optimized DTW alignment for score-performance alignment.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fabs

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float32_t, ndim=2] dtw_align(
    float[:,:] score, 
    float[:,:] perf, 
    float ds, 
    float lmbda
):
    """
    DTW alignment algorithm - Cython optimized.
    
    Args:
        score: array of shape (N, 129) where [:, :128] are note features and [:, 128] is timing
        perf: array of shape (M, 128) note features
        ds: time step size
        lmbda: regularization parameter for tempo deviation
    
    Returns:
        L: array of shape (N, M) cost matrix
    """
    cdef int N = score.shape[0]
    cdef int M = perf.shape[0]
    
    cdef float[:] score_timing = score[:, 128]
    
    # Compute cumulative sum for prior calculation
    cdef float cumsum = 0.0
    cdef int idx
    for idx in range(N):
        cumsum += score_timing[idx]
    
    cdef float prior = (ds * M) / cumsum  # slope = rise/run
    
    # Allocate cost matrix
    cdef np.ndarray[np.float32_t, ndim=2] npL = np.full((N, M), np.inf, dtype=np.float32)
    cdef float[:,:] L = npL
    
    # Allocate local cost matrix
    cdef np.ndarray[np.float32_t, ndim=2] npC = np.empty((N, M), dtype=np.float32)
    cdef float[:,:] local_cost = npC
    
    cdef float cost, tmp, sj, instantaneous_tempo, incremental_cost, R
    cdef int j, k, m, i
    
    # Precompute local cost of aligning score[j] with perf[k]
    for j in range(N):
        for k in range(M):
            local_cost[j, k] = 0
            for i in range(128):
                tmp = score[j, i] - perf[k, i]
                if tmp > 0:
                    local_cost[j, k] += tmp
                else:
                    local_cost[j, k] -= tmp
    
    # Base case j = 0
    sj = score_timing[0]
    incremental_cost = 0
    for k in range(M):
        instantaneous_tempo = (k * ds) / sj
        tmp = instantaneous_tempo - prior
        R = lmbda * tmp * tmp
        incremental_cost += local_cost[0, k]
        L[0, k] = incremental_cost * ds + R
    
    L[0, 0] = 0  # base case
    
    # Fill DP table
    for j in range(1, N):
        sj = score_timing[j]
        for k in range(M):
            incremental_cost = 0
            
            for m in range(k, -1, -1):  # reversed range
                instantaneous_tempo = ((k - m) * ds) / sj
                tmp = instantaneous_tempo - prior
                R = lmbda * tmp * tmp
                
                cost = L[j - 1, m] + incremental_cost * ds + R
                
                if cost < L[j, k]:
                    L[j, k] = cost
                
                incremental_cost += local_cost[j, m]
    
    return npL


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple dtw_traceback(
    float[:,:] score,
    float[:,:] perf,
    float[:,:] L,
    float ds,
    float lmbda
):
    """
    Traceback through DTW cost matrix to find optimal alignment.
    
    Args:
        score: array of shape (N, 129)
        perf: array of shape (M, 128)
        L: cost matrix from dtw_align
        ds: time step size
        lmbda: regularization parameter
    
    Returns:
        A: list of (score_idx, perf_idx) alignment pairs
        C: list of costs
    """
    cdef int N = score.shape[0]
    cdef int M = perf.shape[0]
    
    cdef float[:] score_timing = score[:, 128]
    
    # Compute cumulative sum for prior calculation
    cdef float cumsum = 0.0
    cdef int idx
    for idx in range(N):
        cumsum += score_timing[idx]
    
    cdef float prior = (ds * M) / cumsum
    
    cdef list A = []
    cdef list C = []
    cdef int k = M - 1
    cdef int j, m, i
    cdef float sj, incremental_cost, instantaneous_tempo, tmp, R, expected_cost
    cdef float tol = 1e-4
    cdef bint found
    
    for j in range(N - 1, 0, -1):  # reversed range(1, N)
        sj = score_timing[j]
        incremental_cost = 0
        found = False
        
        for m in range(k, -1, -1):  # reversed range(0, k+1)
            instantaneous_tempo = ((k - m) * ds) / sj
            
            tmp = instantaneous_tempo - prior
            R = lmbda * tmp * tmp
            
            expected_cost = L[j - 1, m] + incremental_cost * ds + R
            
            # Check if costs match (with tolerance)
            if fabs(L[j, k] - expected_cost) < tol:
                A.append((j, k))
                C.append(L[j, k])
                k = m
                found = True
                break
            
            # Update incremental cost
            for i in range(128):
                tmp = score[j, i] - perf[m, i]
                if tmp > 0:
                    incremental_cost += tmp
                else:
                    incremental_cost -= tmp
        
        if not found:
            # Fallback: just move to previous frame
            A.append((j, k))
            C.append(L[j, k])
    
    A.append((0, k))
    C.append(L[0, k])
    
    # Reverse lists
    A = A[::-1]
    C = C[::-1]
    
    return A, C


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float32_t, ndim=2] compute_local_costs(
    float[:,:] score,
    float[:,:] perf
):
    """
    Precompute local costs between all score and performance frames.
    Uses L1 distance over 128 piano keys.
    """
    cdef int N = score.shape[0]
    cdef int M = perf.shape[0]
    cdef int i, j, k
    cdef float tmp, cost
    
    cdef np.ndarray[np.float32_t, ndim=2] local_cost = np.empty((N, M), dtype=np.float32)
    
    for j in range(N):
        for k in range(M):
            cost = 0
            for i in range(128):
                tmp = score[j, i] - perf[k, i]
                if tmp > 0:
                    cost += tmp
                else:
                    cost -= tmp
            local_cost[j, k] = cost
    
    return local_cost
