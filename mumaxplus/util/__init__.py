"""Utilities for mumaxplus."""

from .constants import GAMMALL, GAMMA0, MU0, KB, QE, MUB, HBAR
from .config import twodomain, antivortex, blochskyrmion, neelskyrmion, vortex
from .config import gaussian_spherical_OoP, gaussian_spherical_IP, gaussian_uniforn_IP
from .formulary import magnetostatic_energy_density, Km, exchange_length, l_ex, wall_width, helical_length, magnetic_hardness
from .show import show_field, show_layer, show_magnet_geometry, show_field_3D
from .shape import *

__all__ = [
    "GAMMALL", "GAMMA0", "MU0", "KB", "QE", "MUB", "HBAR",

    "twodomain", "vortex", "antivortex", "neelskyrmion", "blochskyrmion",
    "gaussian_spherical_OoP", "gaussian_spherical_IP", "gaussian_uniforn_IP",

    "magnetostatic_energy_density", "Km", "exchange_length", "l_ex",
    "wall_width", "helical_length", "magnetic_hardness",
    "show_field",
    "show_layer",
    "show_magnet_geometry",
    "show_field_3D"
]
