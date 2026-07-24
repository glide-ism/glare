"""
GLARE: GPU-accelerated enhanced temperature index surface mass balance model.

A GPU-accelerated model for computing surface mass balance on mountain glaciers
and ice caps, using enhanced temperature index methods and terrain-aware solar
radiation calculations.
"""

__version__ = "0.1.0"
__author__ = "Doug Brinkerhoff"

from .helpers import PanCarraBase
from .avalanche import AvalancheOperator, partition_forward, partition_adjoint
from .enthalpy import (EnthalpyModel, Parameters, State, Fluxes, step,
                       run_column, run_column_adjoint, generate_temp_deviations,
                       Gradients)

__all__ = ["PanCarraBase", "AvalancheOperator",
           "partition_forward", "partition_adjoint",
           "EnthalpyModel", "Parameters", "State", "Fluxes", "step",
           "run_column", "run_column_adjoint", "generate_temp_deviations",
           "Gradients"]
