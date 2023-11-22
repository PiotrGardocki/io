#!/bin/env python3

# a)
def prime(n):
    i = 2

    while i * i <= n:
        if n % i == 0:
            return False
        i += 1

    return True

# b)
def select_primes(x):
    return [i for i in x if prime(i)]

print("a)")
print("Wynik:", prime(3))
print("Wynik:", prime(4))
print("Wynik:", prime(49))
print("Wynik:", prime(23))
print("Wynik:", prime(22))
print("b)")
print(select_primes([3, 4, 49, 23, 22]))
