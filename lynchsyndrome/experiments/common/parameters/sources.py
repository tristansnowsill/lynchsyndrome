from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Mapping, Optional, Union

import numpy.random
import pandas
from injector import inject


class ParameterSource(ABC):
    """Abstract base class for parameter sources
    
    This abstract base class specifies that ``ParameterSource`` concrete classes
    must implement the iterator interface (i.e., `__iter__` and `__next__`
    methods) and the `__next__` method should return an object which implements
    the ``collections.abc.Mapping`` interface. ``ParameterSource`` also
    overloads the addition operator so that multiple parameter sources can be
    easily combined.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()
    
    @abstractmethod
    def __next__(self):
        raise NotImplementedError()
    
    @abstractmethod
    def keys(self) -> Iterable[str]:
        raise NotImplementedError()
    
    def __add__(self, other):
        return CompositeParameterSource([self, other])


class CachedParameterSource(ParameterSource):
    """Parameter source cache
    
    Particularly helpful to avoid traversing the binary trees that result from
    creating ``CompositeParameterSource`` using the overloaded addition
    option.
    """

    def __init__(self, source: ParameterSource):
        self._source = source
        self._params = None
        self._cache = dict()
        self._keys = list(self._source.keys())
    
    def __iter__(self):
        return self
    
    def __next__(self):
        logging.debug('CachedParameterSource.__next__ called')
        self._params = next(self._source)
        self._cache = dict()
        return self
    
    def keys(self) -> Iterable[str]:
        return self._keys
    
    def __contains__(self, key):
        logging.debug('CachedParameterSource.__contains__ called with key %s', key)
        return key in self._keys

    def __getitem__(self, key):
        logging.debug('CachedParameterSource.__getitem__ called with key %s', key)
        if key in self._cache:
            return self._cache[key]
        else:
            val = self._params[key]
            self._cache[key] = val
            return val


class CompositeParameterSource(ParameterSource):
    """Parameter source created by combining existing ``ParameterSource``
    
    ``CompositeParameterSource`` uses a hashtable to determine which
    ``ParameterSource`` to dispatch the `__getitem__` call to. If the key does
    not exist in the hashtable it will search through the input sources in
    sequence until it finds the relevant parameter or raises a ``KeyError``.
    """
    def __init__(self, sources: List[ParameterSource]):
        super().__init__()
        self._sources = sources
        self._params = list()
        self._cache = dict()
        self._keys = sum((list(s.keys()) for s in sources), list())

    def __iter__(self):
        return self
    
    def __next__(self):
        logging.debug('CompositeParameterSource.__next__ called')
        self._params = [next(s) for s in self._sources]
        return self
    
    def keys(self) -> Iterable[str]:
        return self._keys
    
    def __contains__(self, key):
        logging.debug('CompositeParameterSource.__contains__ called with key %s', key)
        return key in self._keys
    
    def __getitem__(self, key):
        logging.debug('CompositeParameterSource.__getitem__ called with key %s', key)
        if key in self._cache:
            logging.debug('According to cache, key is in %i', self._cache[key])
            return self._params[self._cache[key]][key]
        else:
            logging.debug('Key is not present in cache')
            for i in range(len(self._params)):
                logging.debug('Checking in _sources[%d]', i)
                if key in self._sources[i]:
                    self._cache[key] = i
                    return self._params[i][key]
        raise KeyError()
    

class ConstantParameterSource(ParameterSource):

    """A source for parameters which are not subject to uncertainty
    
    This source will always produce the same parameter set, so it is not
    appropriate for parameters which are not known with certainty. It may be
    appropriate for quantities such as discount rates.

    :param params: a Python ``dict`` of the parameters to be included in the parameter source
    """
    def __init__(self, params):
        super().__init__()
        self._params: Mapping[str, Any] = params
    
    def keys(self) -> Iterable[str]:
        return self._params.keys()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self._params
    
    def __contains__(self, key):
        return key in self._params


class LazyCallableParameterSet:
    
    def __init__(self, rng: numpy.random.Generator, callables):
        self._rng = rng
        self._callables = callables
        self._cache = dict()
    
    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        elif key in self._callables:
            self._cache[key] = self._callables[key](self._rng)
            return self._cache[key]
        else:
            return KeyError()


class CallableParameterSource(ParameterSource):

    @inject
    def __init__(self, rng: numpy.random.Generator, callables: dict):
        super().__init__()
        self._rng = rng
        self._callables = callables
    
    def keys(self) -> Iterable[str]:
        return self._callables.keys()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return LazyCallableParameterSet(self._rng, self._callables)
    
    def __contains__(self, key):
        return key in self._callables


class DependentParameterSource(ParameterSource):

    """Allows for parameters to be statistically dependent
    
    Supply a single callable and an index (list of parameter names). The
    callable is called once to generate an array (or some other iterable data
    structure), and this is combined with the index.
    """
    @inject
    def __init__(self, rng: numpy.random.Generator, index: List[str], callable):
        super().__init__()
        self._rng = rng
        self._index = index
        self._callable = callable
    
    def __iter__(self):
        return self
    
    def __next__(self):
        res = self._callable(self._rng)
        return dict(zip(self._index, res))
    
    def keys(self) -> Iterable[str]:
        return self._index
    
    def __contains__(self, key):
        logging.debug('DependentParameterSource.__contains__ called with key %s', key)
        if key in self._index:
            logging.debug('Key found')
            return True
        else:
            logging.debug('Key not found')
            return False


class ParameterSamplesSource(ParameterSource):

    """A source of parameters which is defined by a finite set of samples
    
    A typical use case for this is that samples from a posterior distribution
    for parameters has been obtained by numerical Bayesian methods, e.g.,
    Bayesian MCMC.

    The default behaviour of ``ParameterSamplesSource`` is to randomly pick
    from the samples without replacement until all samples have been exhausted
    and then to restart and pick again in a different order. This ensures that
    if multiple ``ParameterSamplesSource`` are in use with the same number of
    samples they will not become correlated, but each will cover the parameter
    space efficiently.

    The input `samples` must implement `__getitem__` to select the data for a
    particular parameter, and that data must itself implement `__getitem__` to
    select the value of the parameter from a particular iteration. Suitable
    structures include a `dict` of `list` objects and a ``pandas.DataFrame``.
    """
    @inject
    def __init__(self, rng: numpy.random.Generator, samples, n_samples: int):
        super().__init__()
        self._rng = rng
        self._n_samples = n_samples
        self._samples = samples
        self._order = rng.choice(n_samples, n_samples, replace=False)
        self._pos = -1

    def __iter__(self):
        return self
    
    def __next__(self):
        self._pos += 1
        if self._pos == self._n_samples:
            self._order = self._rng.choice(self._n_samples, self._n_samples, replace=False)
            self._pos = 0
        return self
    
    def keys(self) -> Iterable[str]:
        return self._samples.keys()
    
    def __contains__(self, key):
        logging.debug('ParameterSamplesSource.__contains__ called with key %s', key)
        if key in self._samples:
            logging.debug('Key found')
        else:
            logging.debug('Key not found')
        return key in self._samples
    
    def __getitem__(self, key):
        if self._pos >= 0:
            return self._samples[key][self._pos]
        else:
            raise RuntimeError('attempting to access a parameter from ParameterSamplesSource when __next__ has never been called')


class OverrideParameterSource(ParameterSource):
    """A ParameterSource to specifically support overriding parameter values
    
    Although :py:class:`~lynchsyndrome.experiments.common.parameters.sources.CompositeParameterSource`
    currently returns the parameter value from the first source which reports
    containing the parameter (using a depth-first search), this behaviour is
    not guaranteed.
    """

    def __init__(self, base: ParameterSource, overrides: ParameterSource):
        super().__init__()
        self._base = base
        self._overrides = overrides
        self._keys = list()
        for k in base.keys():
            self._keys.append(k)
        for k in overrides.keys():
            self._keys.append(k)

    def __iter__(self):
        return self
    
    def __next__(self):
        logging.debug('OverridesParameterSource.__next__ called')
        self._base = next(self._base)
        self._overrides = next(self._overrides)
        return self
    
    def keys(self) -> Iterable[str]:
        return self._keys
    
    def __contains__(self, key):
        logging.debug('OverridesParameterSource.__contains__ called with key %s', key)
        return key in self._overrides or key in self._base
    
    def __getitem__(self, key):
        logging.debug('OverridesParameterSource.__getitem__ called with key %s', key)
        if key in self._overrides:
            return self._overrides[key]
        elif key in self._base:
            return self._base[key]
        else:
            raise KeyError()


# Some utility functions

def from_cmdstan_csv(filenames_or_buffers: List, rng: numpy.random.Generator, prefix: Optional[str] = '') -> ParameterSamplesSource:
    df = pandas.concat(
        (pandas.read_csv(f, comment='#') for f in filenames_or_buffers),
        axis=0
    )
    df.reset_index(inplace=True, drop=True)
    n = len(df)
    cmdstan_cols = [
        'lp__','accept_stat__','stepsize__','treedepth__','n_leapfrog__','divergent__','energy__',
        'log_p__','log_g__'
    ]
    df.drop(columns=cmdstan_cols, inplace=True, errors='ignore')
    df = df.add_prefix(prefix)
    return ParameterSamplesSource(rng, df, n)
