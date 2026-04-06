"""
GLARE: GPU-accelerated enhanced temperature index surface mass balance model.

A GPU-accelerated model for computing surface mass balance on mountain glaciers
and ice caps, using enhanced temperature index methods and terrain-aware solar
radiation calculations.
"""

__version__ = "0.1.0"
__author__ = "Doug Brinkerhoff"

from .solar_potential import SolarPotential
from .helpers import PanCarraBase

__all__ = ["SolarPotential","PanCarraBase"]
