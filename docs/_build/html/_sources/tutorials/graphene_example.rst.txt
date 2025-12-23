Graphene Density Matrix Tutorial
================================

This tutorial demonstrates how to calculate the density matrix of graphene using three different solvers provided by ``tb_solve``:

1. **Diagonalization** (Standard method)
2. **Fermi Operator Expansion** (Linear scaling, finite temperature)
3. **Density Matrix Purification** (Linear scaling, zero temperature)

`PythTB <https://pythtb.org/>`_ is used to generate the Hamiltonian, but these solvers are general to any tight-binding model.

.. literalinclude:: ../../examples/tutorial_graphene_density_matrix.py
   :language: python
   :linenos:
   :caption: tutorial_graphene_density_matrix.py

