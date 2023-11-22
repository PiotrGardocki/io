#!/bin/env python3

import random
import math

# a)
def add_vectors(vec1, vec2):
    return [sum(i) for i in zip(vec1, vec2)]

def multiply_vectors(vec1, vec2):
    return [t[0] * t[1] for t in zip(vec1, vec2)]

# b)
def scalar_product(vec1, vec2):
    return sum(multiply_vectors(vec1, vec2))

# c)
def euclidian_len(vec):
    return math.sqrt(sum(i * i for i in vec))

# d)
def random_vector(len=50, min=1, max=100):
    return [random.randint(min, max) for _ in range(len)]

# e)
def print_vector_stats(vec):
    print(f"min: {min(vec)}")
    print(f"max: {max(vec)}")
    avg = sum(vec) / len(vec)
    print(f"avg: {avg}")
    deviation = math.sqrt(sum(pow(i - avg, 2) for i in vec) / len(vec))
    print(f"standard deviation: {deviation}")

# f)
def normalize_vector(vec):
    min_v = min(vec)
    max_v = max(vec)
    max_index = vec.index(max_v)
    normalized = [(i - min_v) / (max_v - min_v) for i in vec]
    print(f"New vector: {normalized}")
    print(f"max = {max_v} was at index {max_index}, now {normalized[max_index]}")

# g)
def standarize_vector(vec):
    avg = sum(vec) / len(vec)
    deviation = math.sqrt(sum(pow(i - avg, 2) for i in vec) / len(vec))
    standarized = [(i - avg) / deviation for i in vec]
    vec = standarized
    avg = sum(vec) / len(vec)
    deviation = math.sqrt(sum(pow(i - avg, 2) for i in vec) / len(vec))
    print(f"New vector: {vec}")
    print(f"new avg: {avg}, new deviation: {deviation}")

vec1 = [1, 2, 3]
vec2 = [4, 5, 6]

print("a)")
print(add_vectors(vec1, vec2))
print(multiply_vectors(vec1, vec2))
print("b)")
print(scalar_product(vec1, vec2))
print("c)")
print(euclidian_len(vec1))
print(euclidian_len(vec2))
print("d)")
rand_vec = random_vector()
print(rand_vec)
print("e)")
print_vector_stats(rand_vec)
print("f)")
normalize_vector(rand_vec)
print("g)")
standarize_vector(rand_vec)
