# cython: language_level=3

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def match_perf_to_score(object p_tuples,
                        object s_tuples,
                        double p_min,
                        double p_max,
                        object mapped_times,
                        double thres):
    cdef Py_ssize_t i, j, n_perf, n_score, num_candidates
    cdef double p_time, mapped_time, s_time, dist, best_dist
    cdef long p_note, best_index, k
    cdef object score_indices_by_pitch = {}
    cdef object candidate_indices
    cdef object s_tuple
    cdef bytearray available
    cdef list matched_indices

    n_perf = len(p_tuples)
    n_score = len(s_tuples)
    matched_indices = [-1] * n_perf
    available = bytearray(b"\x01") * n_score

    for i in range(n_score):
        s_tuple = s_tuples[i]
        p_note = <long>s_tuple[2]
        candidate_indices = score_indices_by_pitch.get(p_note)
        if candidate_indices is None:
            score_indices_by_pitch[p_note] = [i]
        else:
            candidate_indices.append(i)

    for i in range(n_perf):
        p_time = <double>p_tuples[i][0]
        if p_time < p_min or p_time > p_max:
            continue

        p_note = <long>p_tuples[i][2]
        candidate_indices = score_indices_by_pitch.get(p_note)
        if candidate_indices is None:
            continue

        mapped_time = <double>mapped_times[i]
        best_dist = thres + 1.0
        best_index = -1
        num_candidates = len(candidate_indices)

        for j in range(num_candidates):
            k = <long>candidate_indices[j]
            if not available[k]:
                continue

            s_time = <double>s_tuples[k][0]
            dist = mapped_time - s_time
            if dist < 0:
                dist = -dist

            if dist <= thres and dist <= best_dist:
                best_dist = dist
                best_index = k

        if best_index >= 0:
            matched_indices[i] = best_index
            available[best_index] = 0

    return matched_indices
