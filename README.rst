===========================================================
Lynch Syndrome Whole Disease Model for Economic Evaluations
===========================================================


.. image:: https://img.shields.io/pypi/v/lynchsyndrome.svg
        :target: https://pypi.python.org/pypi/lynchsyndrome

.. image:: https://img.shields.io/travis/tristansnowsill/lynchsyndrome.svg
        :target: https://travis-ci.org/tristansnowsill/lynchsyndrome

.. image:: https://codecov.io/gh/tristansnowsill/lynchsyndrome/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/tristansnowsill/lynchsyndrome

.. image:: https://readthedocs.org/projects/lynchsyndrome/badge/?version=latest
        :target: https://lynchsyndrome.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




TODO - Add short description


* Free software: MIT license
* Documentation: https://lynchsyndrome.readthedocs.io.

Running existing experiments
----------------------------

Existing experiments are found in the ``lynchsyndrome.experiments`` package. You can run them
from the console with code such as:

.. code-block:: shell
    
    $ python3 -m lynchsyndrome.experiments.NIHR129713

Using this package for your own experiments
-------------------------------------------

You can create your own submodules inside ``lynchsyndrome.experiments``. There is a repository
of suitable parameters and experiment setup code in ``lynchsyndrome.experiments.common``. Note
that the contents of ``lynchsyndrome.experiments.common`` are subject to change and expected
to evolve with programming best practice and with improving subject knowledge.

Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `pyOpenSci/cookiecutter-pyopensci`_ project template, based off `audreyr/cookiecutter-pypackage`_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`pyOpenSci/cookiecutter-pyopensci`: https://github.com/pyOpenSci/cookiecutter-pyopensci
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
