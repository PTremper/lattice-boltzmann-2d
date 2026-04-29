"""Run a Lattice Boltzmann simulation to generate velocity and vorticity images."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geom_cylinder import cylinder

BASE_PATH = Path("outputs")
IMAGE_PATH = BASE_PATH / "images"


# fmt: off
def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("backend", type=str, default="numpy", choices=["numpy", "torch"],
        help="select backend: numpy or torch. Default is numpy.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="select device: cpu or cuda (only when using torch backend). Default is cpu.")
    parser.add_argument("--steps", type=int, default=1000,
        help="Number of simulation steps to run")
    parser.add_argument("--output_filename", type=str, default="lbm_velocity_vorticity.png",
        help="Output path including filename for image.")
    parser.add_argument("--u0", type=float, default=0.06,
        help="Inlet velocity in x-direction (flow direction) in lattice units")
    parser.add_argument("--nu",type=float,default=0.02,
        help="Kinematic viscosity of the fluid in lattice units")
    parser.add_argument("--tau",type=float,
        help="Relaxation parameter. By default none and calculated from nu.")
    parser.add_argument("--nx", type=int, default=400,
        help="Set domain size in x-direction")
    parser.add_argument("--ny", type=int, default=100,
        help="Set domain size in y-direction")
    parser.add_argument("--velocity_streamplot", action="store_true",
        help="Overlay a streamplot over the velocity plot.")

    return parser.parse_args()
# fmt: on


def main(image_path: Path) -> None:
    """Run a Lattice Boltzmann simulation to generate velocity and vorticity images."""
    args = _parse_args()

    image_path.mkdir(parents=True, exist_ok=True)
    save_path = image_path / args.output_filename

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

    u, _ = lbm.simulate(steps=args.steps)  # use ~9_000 steps to reach steady state

    ux = u[..., 0]
    uy = u[..., 1]
    vel = np.sqrt(ux**2 + uy**2)

    # --- mask obstacle ---
    vel_masked = vel.copy()
    vel_masked[geometry] = np.nan

    ux_masked = ux.copy()
    uy_masked = uy.copy()
    ux_masked[geometry] = np.nan
    uy_masked[geometry] = np.nan

    # --- compute vorticity (central differences) ---
    dudy = (ux_masked[:, 2:] - ux_masked[:, :-2]) / 2
    dvdx = (uy_masked[2:, :] - uy_masked[:-2, :]) / 2

    # match shapes
    omega = dvdx[:, 1:-1] - dudy[1:-1, :]

    # pad back to full size
    omega_full = np.full_like(vel, np.nan)
    omega_full[1:-1, 1:-1] = omega
    omega_full[geometry] = np.nan

    # --- color limits ---

    vmin, vmax = 0, np.nanpercentile(vel_masked, 99)
    wmin, wmax = -np.nanpercentile(np.abs(omega_full), 99), np.nanpercentile(np.abs(omega_full), 99)

    # --- plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(20, 8))

    # =========================
    # Velocity magnitude
    # =========================
    im0 = axes[0].imshow(
        vel_masked.T,
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    # --- streamlines ---
    if args.velocity_streamplot:
        nx, ny = vel.shape
        x = np.arange(nx)
        y = np.arange(ny)

        axes[0].streamplot(
            x,
            y,
            ux_masked.T,
            uy_masked.T,
            color="white",
            density=1.2,
            linewidth=0.5,
        )
    # --- end streamlines ---

    axes[0].set_title("Velocity magnitude")
    plt.colorbar(im0, ax=axes[0])

    # overlay obstacle as black
    axes[0].imshow(
        geometry.T,
        origin="lower",
        cmap="gray_r",
        alpha=geometry.T.astype(float),
    )

    # =========================
    # Vorticity
    # =========================
    im1 = axes[1].imshow(
        omega_full.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=wmin,
        vmax=wmax,
    )
    axes[1].set_title("Vorticity")
    plt.colorbar(im1, ax=axes[1])

    # overlay obstacle as black
    axes[1].imshow(
        geometry.T,
        origin="lower",
        cmap="gray_r",
        alpha=geometry.T.astype(float),
    )

    # clean axes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main(image_path=IMAGE_PATH)
