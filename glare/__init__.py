"""
GLARE: GPU-accelerated, differentiable surface mass balance framework.

Two interchangeable physics cores compute surface mass balance on mountain glaciers
and ice caps over a shared, agnostic :class:`~glare.grid.Grid`:

* the **temperature-index** core (:class:`~glare.model.ImprovedTemperatureIndex` over
  :class:`~glare.grid.TIMGrid`) -- enhanced positive-degree-day + terrain-aware solar
  radiation melt, and
* the **snowpack-enthalpy** core (:class:`~glare.enthalpy.EnthalpyModel` over
  :class:`~glare.grid.EnthalpyGrid`) -- a physically explicit depth-averaged
  thermodynamic reservoir.

Both are co-equal: parameters live on the grid as ``Constant``s, output state on
``grid.state``, and the adjoint writes gradients directly into the grid's
``Field.grad`` / ``Constant.grad`` buffers.  The two paths share the same surface --
``model.forward()``, ``model.adjoint(...)``, ``model.grid.state.<field>``.
"""

__version__ = "0.1.0"
__author__ = "Doug Brinkerhoff"

from .helpers import PanCarraBase
from .grid import Grid, TIMGrid, EnthalpyGrid
from .model import ImprovedTemperatureIndex
from .avalanche import AvalancheOperator, partition_forward, partition_adjoint
from .operators import GRAD_PARAM_NAMES
from .enthalpy import (EnthalpyModel, State, Fluxes, step, run_column,
                       run_column_adjoint, generate_temp_deviations, scalar_params)

__all__ = ["PanCarraBase",
           "Grid", "TIMGrid", "EnthalpyGrid",
           "ImprovedTemperatureIndex",
           "AvalancheOperator", "partition_forward", "partition_adjoint",
           "EnthalpyModel", "State", "Fluxes", "step", "run_column",
           "run_column_adjoint", "generate_temp_deviations", "scalar_params",
           "GRAD_PARAM_NAMES"]
