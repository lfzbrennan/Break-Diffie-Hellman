import torch
from torch.utils.data import IterableDataset
import random

import sympy
from sympy.ntheory.primetest import isprime


class DiscreteLogDataset(IterableDataset):
    def __init__(self, bits=16):
        self.g = 2
        self.bits = bits

    def __iter__(self):
        return self

    # format long as binary
    def to_bit_string(self, s):
        return format(s, f"0{self.bits}b")

    def get_safe_prime(self):
        while True:
            q = sympy.randprime(0, 2 ** (self.bits - 1))
            if q % 12 == 5 and isprime(2 * q + 1):
                return 2 * q + 1

    def __next__(self):

        g = self.g

        # get random prime N
        N = self.get_safe_prime()

        # get random "key" x
        x = random.randrange(0, N)

        # calculate exponent
        g_x = pow(g, x, N)

        # convert to bit format
        N = self.to_bit_string(N)
        g = self.to_bit_string(g)
        x = self.to_bit_string(x)

        g_x = self.to_bit_string(g_x)

        # cipher is N plus exponent, label is x
        cipher = N + g_x

        # convert to LongTensor
        cipher = torch.LongTensor(self.tokenize(cipher))
        x = torch.LongTensor(self.tokenize(g_x))

        return cipher, x

    # convert bit string to list of [0, 1] ints
    def tokenize(self, bit_string):
        return [int(c) for c in bit_string]
