import math

import statistics
from numpy import ndarray


def __check_vectors(a, b):
    if isinstance(a, ndarray) and len(a.shape) > 1:
        a = a.flatten()
    if isinstance(b, ndarray) and len(b.shape) > 1:
        b = b.flatten()
    if len(a) != len(b):
        raise ValueError("The dimensions of the arguments mismatch")
    return a, b


def cosine_similarity(a, b):
    a, b = __check_vectors(a, b)
    vector_dimension = len(a)
    
    length_a = math.sqrt(sum([math.pow(a[i], 2) for i in range(vector_dimension)]))
    length_b = math.sqrt(sum([math.pow(b[i], 2) for i in range(vector_dimension)]))
    cosine_similarity_value = sum([a[i] * b[i] for i in range(vector_dimension)]) / (length_a * length_b)
    return cosine_similarity_value


def euclidean_distance(a, b):
    a, b = __check_vectors(a, b)
    return math.sqrt(sum([math.pow(a[i] - b[i], 2) for i in range(len(a))]))


def sift_similarity_function(matches):
    if isinstance(matches, list) or isinstance(matches, tuple):
        # return 1.0 / sum(matches) if sum(matches) != 0 else sys.maxint
        return statistics.mean(matches)
    raise ValueError()
