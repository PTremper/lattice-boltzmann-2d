"""Run a Lattice Boltzmann simulation and save the results."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from geom_cylinder import cylinder

BASE_PATH = Path("outputs")
DATA_PATH = BASE_PATH / "data"


# fmt: off
def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Set up arguments
    parser.add_argument("backend", type=str, default="numpy", choices=["numpy", "torch"],
        help="select backend: numpy or torch. Default is numpy.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="select device: cpu or cuda (only when using torch backend). Default is cpu.")
    parser.add_argument("--steps", type=int, default=1000,
        help="Number of simulation steps to run.")
    parser.add_argument("--burn_in", type=int, default=0,
        help="Number of burn-in steps before saving output.")

    # Simulation arguments
    parser.add_argument("--nx", type=int, default=400,
        help="Set domain size in x-direction")
    parser.add_argument("--ny", type=int, default=100,
        help="Set domain size in y-direction")
    parser.add_argument("--u0", type=float, default=0.06,
        help="Inlet velocity in x-direction (flow direction) in lattice units")
    parser.add_argument("--nu", type=float, default=0.02,
        help="Kinematic viscosity of the fluid in lattice units")
    parser.add_argument("--tau", type=float, default=None,
        help="Relaxation parameter. By default none and calculated from nu.")

    # Save output arguments
    parser.add_argument("--save_every", type=int, default=10,
        help="Save output every N steps")
    parser.add_argument("--output_filename_u", type=str, default="u.npy",
        help="Output filename for velocity field u")
    parser.add_argument("--output_filename_rho", type=str, default=None,
        help="Output filename for density field rho. Will not save if set to None. Default: None.")
    parser.add_argument("--output_filename_geometry", type=str, default=None,
        help="Output filename for geometry. Will not save if set to None. Default: None.")

    return parser.parse_args()
# fmt: on


def main(data_path: Path) -> None:
    """Run a Lattice Boltzmann simulation and save the results."""
    args = _parse_args()

    data_path.mkdir(parents=True, exist_ok=True)
    save_path_u: Path = data_path / args.output_filename_u
    save_path_rho: Path | None = (
        (data_path / args.output_filename_rho) if args.output_filename_rho is not None else None
    )
    save_path_geometry: Path | None = (
        (data_path / args.output_filename_geometry)
        if args.output_filename_geometry is not None
        else None
    )
    # Example geometry: obstacle in center
    nx, ny = args.nx, args.ny

    geometry = cylinder(
        nx=nx,
        ny=ny,
        r=20,
        center_x=nx // 5,
        center_y=ny // 2,
        y_offset=1,
    )

    if save_path_geometry is not None:
        np.save(save_path_geometry, geometry)

    # load LBM implementation based on backend
    if args.backend == "numpy":
        from lattice_boltzmann_numpy import LatticeBoltzmann2D as LBMNumPy  # noqa: PLC0415

        lbm = LBMNumPy(
            geometry,
            u0=args.u0,
            nu=args.nu,
            tau=args.tau,
        )

    elif args.backend == "torch":
        import torch  # noqa: PLC0415

        from lattice_boltzmann_torch import LatticeBoltzmann2D as LBMTorch  # noqa: PLC0415

        lbm = LBMTorch(
            geometry,
            u0=args.u0,
            nu=args.nu,
            tau=args.tau,
            device=args.device,
            dtype=torch.float32,
        )

    else:
        print(
            f"Unsupported backend: {args.backend}. Please set to either 'numpy' or 'torch'.",
        )
        return

    # run simulation loop manually instead of u, rho = lbm.simulate(steps=...)
    fluid = geometry == 0
    u_list = []

    if save_path_rho is not None:
        rho_list = []

    steps = args.steps
    burn_in = args.burn_in  # for animation to see how the simulation evolves
    save_every = args.save_every

    print(
        f"Generating LBM simulation data. Backend: {args.backend}, Device: {args.device}",
        f"Geometry: {geometry.shape}, steps: {steps}, burn_in: {burn_in}, save_every: {save_every}",
        "Monitoring field mean velocity 'u_eff' and Reynolds number 're_eff'",
        sep="\n",
    )

    for step in (pbar := tqdm(range(steps))):
        lbm.step()
        if step >= burn_in and step % save_every == 0:
            u, rho = lbm.get_fields_as_numpy()
            u[~fluid] = 0
            u_list.append(u.copy())

            if save_path_rho is not None:
                rho[~fluid] = 0
                rho_list.append(rho.copy())

            # update progress bar with mean velocity and its Reynolds number
            u_eff = (u * (u > 0))[..., 0].mean()
            re_eff = lbm.estimate_reynolds_number()
            pbar.set_description(f"u_eff = {u_eff:.4f}, Re = {re_eff:.2f}")

    # stack u_total along time axis, shape (time, x, y, 2)
    u_total = np.stack(u_list, axis=0)
    np.save(save_path_u, u_total)
    print(f"Velocity data saved to {save_path_u} as numpy array of shape {u_total.shape}")

    if save_path_rho is not None:
        # stack rho_total along time axis, shape (time, x, y)
        rho_total = np.stack(rho_list, axis=0)
        np.save(save_path_rho, rho_total)
        print(f"Density data saved to {save_path_rho} as numpy array of shape {rho_total.shape}")


if __name__ == "__main__":
    main(data_path=DATA_PATH)
