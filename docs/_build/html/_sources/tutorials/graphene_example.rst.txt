Graphene Density Matrix Tutorial
================================

This tutorial demonstrates how to calculate the density matrix of graphene using three different solvers provided by ``tb_solve``:

1. **Diagonalization** (Standard method)
2. **Fermi Operator Expansion** (Linear scaling, finite temperature)
3. **Density Matrix Purification** (Linear scaling, zero temperature)

It uses `PythTB <https://pythtb.org/>`_ to generate the Hamiltonian.

.. literalinclude:: graphene_density_matrix.py
   :language: python
   :linenos:
   :caption: graphene_density_matrix.py

