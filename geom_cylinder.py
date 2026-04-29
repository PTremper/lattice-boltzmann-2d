"""Obstacle geometry utilities for the Lattice Boltzmann simulation."""

import numpy as np
from numpy.typing import NDArray


def cylinder(
    nx: int,
    ny: int,
    r: int,
    center_x: int,
    center_y: int,
    y_offset: int,
) -> NDArray[np.integer]:
    """Create a circular obstacle geometry for the 2D Lattice Boltzmann simulation.

    Args:
    ----
        nx (int): Number of grid points in the x direction.
        ny (int): Number of grid points in the y direction.
        r (int): Radius of the cylinder.
        center_x (int): X-coordinate of the cylinder's center.
        center_y (int): Y-coordinate of the cylinder's center.
        y_offset (int): Vertical offset of the cylinder from the center.

    Returns:
    -------
        NDArray[np.integer]: A 2D array representing the obstacle geometry.

    """
    geometry = np.zeros((nx, ny), dtype=int)

    # Circular obstacle slightly offset from center
    for x in range(nx):
        for y in range(ny):
            if (x - center_x) ** 2 + (y - (center_y + y_offset)) ** 2 < r**2:
                geometry[x, y] = 1
    return geometry
