# cython: language_level=3

from libc.math cimport fabs


def match_perf_to_score(list p_tuples, list s_tuples, double p_min, double p_max,
                        list mapped_times, double thres):
    """
    Cython mirror of align_tokens2's matching loop.

    Returns a list with one entry per performance tuple: the matched score tuple
    index in the original `s_tuples`, or -1 when no match is found.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t score_pos
    cdef Py_ssize_t best_index
    cdef Py_ssize_t n_perf = len(p_tuples)
    cdef double p_time
    cdef double s_time
    cdef double mapped_time
    cdef double dist
    cdef double best_dist
    cdef object p_tuple
    cdef object s_tuple
    cdef object best_match
    cdef object p_note
    cdef object s_note
    cdef list matches = [-1] * n_perf
    cdef list s_tuples_copy = s_tuples.copy()

    for i in range(n_perf):
        p_tuple = p_tuples[i]
        p_time = float(p_tuple[0])
        p_note = p_tuple[2]
        best_dist = float("inf")
        best_match = None
        best_index = -1

        if p_min <= p_time <= p_max:
            mapped_time = float(mapped_times[i])

            for s_tuple in s_tuples_copy:
                s_time = float(s_tuple[0])
                s_note = s_tuple[2]

                if p_note != s_note:
                    continue

                dist = fabs(mapped_time - s_time)
                if dist <= thres and dist <= best_dist:
                    best_dist = dist
                    best_match = s_tuple
                    best_index = -1
                    for score_pos in range(len(s_tuples)):
                        if s_tuples[score_pos] == s_tuple:
                            best_index = score_pos
                            break

        if best_index != -1:
            matches[i] = best_index
            s_tuples_copy.remove(best_match)

    return matches
