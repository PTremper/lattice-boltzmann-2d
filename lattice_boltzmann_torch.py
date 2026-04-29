"""2D Lattice Boltzmann Method solver using the D2Q9 model.

This module implements a basic single-relaxation-time (BGK) LBM scheme
for simulating fluid flow on a 2D grid.

The module is written in PyTorch for GPU acceleration.
"""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

# ---------------------------------------------------------------------------
# D2Q9 Lattice Boltzmann Method (LBM) implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Direction:
    """Represents a single discrete velocity direction in the D2Q9 lattice.

    Attributes
    ----------
    cx : int
        Velocity component in x-direction.
    cy : int
        Velocity component in y-direction.
    w : float
        Lattice weight associated with this direction.

    """

    cx: int
    cy: int
    w: float


class LatticeBoltzmann2D:
    """2D Lattice Boltzmann Method solver in PyTorch using the D2Q9 model.

    This class implements a basic single-relaxation-time (BGK) LBM scheme
    for simulating fluid flow on a 2D grid.

    Parameters
    ----------
    geometry : NDArray[np.integer] of shape (nx, ny)
        2D array defining the simulation domain.
        Non-zero values indicate solid nodes (obstacles), zero indicates fluid.
    u0 : float, optional
        Physical Input. Inlet velocity in lattice units, by default 0.06.
    nu : float, optional
        Physical Input. Kinematic viscosity in lattice units, by default 0.02 (--> tau ~ 0.56).
        stable range 0.02 ~ 0.15 (air ~ oil)

    tau : float, optional
        Relaxation time (controls viscosity). Overwrites the kinematic viscosity!
        tau = 3 * nu + 0.5 --> stable range 0.55 ~ 1.0.

    rho0 : float, optional
        Initial density, by default 1.0.

    device : torch.device | str, optional
        Device to use for computations, by default "cpu".
        Set to "cuda" to use GPU acceleration.
        Check `torch.cuda.is_available()` before setting to "cuda".
    dtype : torch.dtype, optional
        Data type to use for computations, by default torch.float64.
        Note that torch.float32 may be faster but less accurate.

    Notes
    -----
    - The initial configuration of u0 = 0.06 and nu = 0.02 lead to
      a Reynolds number of ~120, which is sufficient for vortex shedding.

    - Distribution function `f` has shape (nx, ny, 9)
    - Velocity field `u` has shape (nx, ny, 2)
    - Density field `rho` has shape (nx, ny)

    """

    def __init__(
        self,
        geometry: NDArray[np.integer],
        u0: float = 0.06,
        nu: float = 0.02,
        tau: float | None = None,
        rho0: float = 1.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize the Lattice Boltzmann simulation."""
        self.device = torch.device(device)
        self.dtype = dtype

        print(f"Running on {self.device} with dtype {self.dtype}")

        # -------------------------------------------------------------------
        # Define D2Q9 lattice directions and weights
        # The ordering here MUST remain consistent across c, weights, and f
        # -------------------------------------------------------------------
        # fmt: off
        DIRECTIONS = [  # noqa: N806 - upper case is conventional for constants
            Direction( 0,  0, 4 / 9),   # 0: rest
            Direction( 1,  0, 1 / 9),   # 1: east
            Direction( 0,  1, 1 / 9),   # 2: north
            Direction(-1,  0, 1 / 9),   # 3: west
            Direction( 0, -1, 1 / 9),   # 4: south
            Direction( 1,  1, 1 / 36),  # 5: northeast
            Direction(-1,  1, 1 / 36),  # 6: northwest
            Direction(-1, -1, 1 / 36),  # 7: southwest
            Direction( 1, -1, 1 / 36),  # 8: southeast
        ]
        # fmt: on

        # Discrete velocity vectors c_i (shape: (9, 2))
        self.c: torch.Tensor = torch.tensor(
            [(d.cx, d.cy) for d in DIRECTIONS],
            device=self.device,
            dtype=self.dtype,
        )

        # Corresponding lattice weights w_i (shape: (9,))
        self.weights: torch.Tensor = torch.tensor(
            [d.w for d in DIRECTIONS],
            device=self.device,
            dtype=self.dtype,
        )

        # Opposite direction indices (used for bounce-back boundary conditions)
        # Example: east (1) <-> west (3), north (2) <-> south (4), etc.
        self.opp: torch.Tensor = torch.tensor(
            [0, 3, 4, 1, 2, 7, 8, 5, 6],
            device=self.device,
            dtype=torch.int64,
        )

        # -------------------------------------------------------------------
        # Geometry and domain setup, physical parameters
        # -------------------------------------------------------------------
        self.solid: torch.Tensor = torch.as_tensor(
            geometry.astype(bool),
            device=self.device,
            dtype=torch.bool,
        )
        self.nx, self.ny = self.solid.shape

        # setting a characteristic length scale based on domain size
        self.length_scale = self._estimate_cylinder_diameter(geometry)
        print(f"Check: Characteristic Length: {self.length_scale:.2f}")

        # Store inlet velocity u0 for Zou/He inlet
        self.u0 = u0

        # Calculate relaxation parameter tau from kinematic viscosity unless tau is given
        if tau is None:
            self.nu: float = nu
            self.tau: float = 0.5 + 3 * self.nu
            print(
                f"Kinematic viscosity nu = {self.nu:.2f}.\n"
                f"Calculating relaxation parameter tau = {self.tau:.2f}",
            )
        else:
            self.tau: float = tau
            self.nu: float = (self.tau - 0.5) / 3
            print(
                f"Relaxation parameter tau = {self.tau:.2f} given.\n"
                f"Calculating kinematic viscosity: {self.nu:.2f}.",
            )

        # safety checks for simulation stability
        # small Mach number ma << 1 to be clearly subsonic
        # (characteristic velocity divided by speed of sound in lattice units)
        print(f"Stability check: Mach number {(u0 / 0.577):.2f} << 1")
        # relaxation parameter tau > 0.6
        print(f"Stability check: Relaxation parameter tau = {self.tau:.2f} > 0.5")

        # -------------------------------------------------------------------
        # Initialize macroscopic fields
        # -------------------------------------------------------------------

        self.rho: torch.Tensor = rho0 * torch.ones(
            (self.nx, self.ny),
            device=self.device,
            dtype=self.dtype,
        )
        self.u: torch.Tensor = torch.zeros(
            (self.nx, self.ny, 2),
            device=self.device,
            dtype=self.dtype,
        )

        # Distribution function f (shape: (nx, ny, 9))
        self.f: torch.Tensor = torch.zeros(
            (self.nx, self.ny, 9),
            device=self.device,
            dtype=self.dtype,
        )

        # initialize f in equilibrium
        self.f = self._equilibrium()

    # -----------------------------------------------------------------------
    # Helper functions
    # -----------------------------------------------------------------------

    def _estimate_cylinder_diameter(self, geometry: NDArray[np.integer]) -> float:
        """Estimate the diameter of a cylinder from the geometry mask."""
        area = geometry.astype(bool).sum()  # number of solid cells
        # return the diameter
        return 2.0 * np.sqrt(area / np.pi)

    def estimate_reynolds_number(self) -> float:
        """Estimate the Reynolds number based on the effective mean velocity and length scale."""
        u_eff = (self.u * (self.u > 0))[..., 0].mean().item()
        # return the Reynolds number
        return u_eff * self.length_scale / self.nu

    # -----------------------------------------------------------------------
    # Core LBM steps
    # -----------------------------------------------------------------------

    def _streaming(self) -> None:
        """Streaming step: propagate distribution functions along lattice directions.

        Each population f_i is shifted by its corresponding velocity c_i.
        Periodic boundary conditions are implicitly applied via np.roll.
        --> Implicitly establishes a periodic boundary.
        --> This is rectified when boundary conditions are applied.
        """
        for i in range(9):
            self.f[..., i] = torch.roll(self.f[..., i], int(self.c[i, 0]), dims=0)
            self.f[..., i] = torch.roll(self.f[..., i], int(self.c[i, 1]), dims=1)

    def _inlet(self) -> None:
        """Left boundary: impose inflow velocity u0, calculate rho (Zou/He)."""
        # left boundary
        x = 0

        # initialize inlet velocity components
        ux, uy = self.u0, 0

        # add bias to break symmetry: simulates a not perfectly aligned inflow angle
        uy += 1e-3

        # known populations: f0, f2, f4, f3, f6, f7
        rho = (
            self.f[x, :, 0]
            + self.f[x, :, 2]
            + self.f[x, :, 4]
            + 2 * (self.f[x, :, 3] + self.f[x, :, 6] + self.f[x, :, 7])
        ) / (1 - ux)

        # reconstruct unknown populations
        self.f[x, :, 1] = self.f[x, :, 3] + (2 / 3) * rho * ux

        self.f[x, :, 5] = (
            self.f[x, :, 7]
            + 0.5 * (self.f[x, :, 4] - self.f[x, :, 2])
            + (1 / 6) * rho * ux
            + 0.5 * rho * uy
        )

        self.f[x, :, 8] = (
            self.f[x, :, 6]
            + 0.5 * (self.f[x, :, 2] - self.f[x, :, 4])
            + (1 / 6) * rho * ux
            - 0.5 * rho * uy
        )

    def _outlet(self) -> None:
        r"""Right boundary: zero-gradient (copy from inside).

        approximates \del f / \del x = 0
        """
        x = self.nx - 1

        self.f[x, :, :] = self.f[x - 1, :, :]

    def _boundary(self) -> None:
        """Apply bounce-back boundary conditions on obstacles and top/bottom edges.

        - Populations hitting an obstacle are reflected back along
        the opposite direction in both velocity components (bounce back).
        - Populations hitting the top/bottom edges are reflected back along
        the opposite direction only in the y-direction (free slip).
        """
        # free slip boundary conditions on top and bottom edges
        # bottom y = 0
        self.f[:, 0, [2, 5, 6]] = self.f[:, 0, [4, 7, 8]]

        # top y = ny-1
        self.f[:, -1, [4, 7, 8]] = self.f[:, -1, [2, 5, 6]]

        # bounce back at obstacle: set velocity to opposite direction
        self.f[self.solid, :] = self.f[self.solid, :][..., self.opp]

    def _macroscopic(self) -> None:
        """Compute macroscopic quantities (density and velocity).

        - Density: rho = sum_i f_i
        - Momentum: j = sum_i f_i * c_i
        - Velocity: u = j / rho
        """
        # Density (shape: (nx, ny))
        self.rho = self.f.sum(dim=2)

        # Momentum (shape: (nx, ny, 2))
        j = (self.f.unsqueeze(-1) * self.c).sum(dim=2)

        # Velocity (shape: (nx, ny, 2))
        self.u = j / self.rho.unsqueeze(-1)

    def _collision(self) -> None:
        """Collision step (BGK approximation).

        Relax distribution function toward equilibrium:
            f <- f - (f - f_eq) / tau

        """
        feq = self._equilibrium()

        # Relaxation step
        self.f = self.f - (self.f - feq) / self.tau

    def _equilibrium(self) -> torch.Tensor:
        """Compute the equilibrium distribution function for the populations f.

        Equilibrium distribution:
            f_eq = w_i * rho * (1 + 3 cu + 4.5 cu^2 - 1.5 |u|^2)
        """
        # Dot product c_i · u (shape: (nx, ny, 9))
        cu = torch.einsum("ia,xya->xyi", self.c.to(self.dtype), self.u)

        # Squared velocity magnitude |u|^2 (shape: (nx, ny))
        usqr = (self.u**2).sum(dim=-1)

        # Equilibrium distribution
        feq = torch.empty_like(self.f, device=self.device, dtype=self.dtype)

        for i in range(9):
            feq[..., i] = (
                self.weights[i]
                * self.rho
                * (1 + 3 * cu[..., i] + 4.5 * cu[..., i] ** 2 - 1.5 * usqr)
            )

        return feq

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def step(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Perform one full LBM iteration.

        streaming → inlet + outlet + boundary → macroscopic → collision.

        Returns
        -------
        u : NDArray[np.floating]
            Velocity field of shape (nx, ny, 2)
        rho : NDArray[np.floating]
            Density field of shape (nx, ny)

        """
        self._streaming()

        self._inlet()
        self._outlet()
        self._boundary()

        self._macroscopic()

        self._collision()

        return self.get_fields_as_numpy()

    def simulate(self, steps: int = 1_000) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Run the simulation for a given number of time steps.

        Parameters
        ----------
        steps : int, optional
            Number of iterations to perform, by default 1_000.

        Returns
        -------
        u : NDArray[np.floating]
            Final velocity field.
        rho : NDArray[np.floating]
            Final density field.

        """
        for step in (pbar := tqdm(range(steps))):
            self.step()

            # update progress bar with u and Reynolds number. Purely visual.
            if step % 100 == 0:
                u, _ = self.get_fields_as_numpy()
                u_eff = (u * (u > 0))[..., 0].mean()
                re_eff = self.estimate_reynolds_number()
                pbar.set_description(f"u_eff = {u_eff:.4f}, re_eff = {re_eff:.2f}")

        return self.get_fields_as_numpy()

    def get_fields_as_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the velocity and density fields as NumPy arrays."""
        if self.device.type == "cuda":
            return self.u.cpu().numpy(), self.rho.cpu().numpy()
        return self.u.numpy(), self.rho.numpy()


if __name__ == "__main__":
    # Example geometry: obstacle in center
    nx, ny = 400, 100
    geometry = np.zeros((nx, ny), dtype=int)

    # Circular obstacle, slightly off center to break symmetry
    cx, cy, r = nx // 5, ny // 2 + 1, 20
    for x in range(nx):
        for y in range(ny):
            if (x - cx) ** 2 + (y - cy) ** 2 < r**2:
                geometry[x, y] = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lbm = LatticeBoltzmann2D(geometry, device=device, dtype=torch.float32)

    # use ~9_000 steps to reach steady state of vortex shedding
    u, _ = lbm.simulate(steps=1_000)

    # Visualize velocity magnitude
    import matplotlib.pyplot as plt

    vel_mag = np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2)

    vmin, vmax = 0, np.percentile(vel_mag, 99)

    plt.figure(figsize=(20, 5))
    plt.imshow(vel_mag.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Velocity magnitude")
    plt.title("LBM Flow Field")
    plt.show()
