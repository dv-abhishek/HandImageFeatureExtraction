import math

from numpy import ndarray


def cosine_similarity(a, b):
    if isinstance(a, ndarray) and len(a.shape) > 1:
        a = a.flatten()
    if isinstance(b, ndarray) and len(b.shape) > 1:
        b = a.flatten()
    if len(a) != len(b):
        raise ValueError("The dimensions of the arguments mismatch")
    cosine_similarity_value = 0
    length_a, length_b = 0, 0
    
    for i in range(len(a)):
        cosine_similarity_value += a[i] * b[i]
        length_a += math.pow(a[i], 2)
        length_b += math.pow(b[i], 2)
    length_a = math.sqrt(length_a)
    length_b = math.sqrt(length_b)
    cosine_similarity_value /= (length_a * length_b)
    return cosine_similarity_value
