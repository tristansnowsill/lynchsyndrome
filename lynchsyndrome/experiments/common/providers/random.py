

import numpy.random
from injector import Module, SingletonScope


class RandomProvider(Module):
    def configure(self, binder):
        binder.bind(numpy.random.Generator, numpy.random.default_rng(), scope=SingletonScope)
