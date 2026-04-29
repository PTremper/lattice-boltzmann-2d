"""Calculate total velocity and vorticity and generate an mp4 video."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tqdm import tqdm

BASE_PATH = Path("outputs")
VIDEO_PATH = BASE_PATH / "videos"
DATA_PATH = BASE_PATH / "data"


# fmt: off
def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_filename", type=str, default="lbm_velocity_vorticity.mp4",
        help="Output filename for video.")
    parser.add_argument("--input_filename_u", type=str, default="u.npy",
        help="Input filename for velocity field u.")
    parser.add_argument("--t_start", type=int, default=0,
        help="Time slice to start the video from.")
    parser.add_argument("--t_end", type=int,
        help="Time slice to end the video at.")

    return parser.parse_args()
# fmt: on


def main() -> None:
    """Calculate total velocity and vorticity and generate an mp4 video."""
    args = _parse_args()

    VIDEO_PATH.mkdir(parents=True, exist_ok=True)
    save_path = VIDEO_PATH / args.output_filename
    load_path = DATA_PATH / args.input_filename_u

    # =========================
    # Load data
    # =========================
    u = np.load(load_path)  # (T, nx, ny, 2)
    if len(u.shape) != 4 or u.shape[3] != 2:  # noqa: PLR2004 (accept explicit numbers)
        msg = f"Expected shape (T, nx, ny, 2), got {u.shape}"
        raise ValueError(msg)

    t_total = u.shape[0]

    # =========================
    # Derived fields
    # =========================
    ux = u[..., 0]
    uy = u[..., 1]
    vel = np.sqrt(ux**2 + uy**2)

    # vorticity (2D curl)
    omega = (
        np.roll(uy, -1, axis=1)
        - np.roll(uy, 1, axis=1)
        - np.roll(ux, -1, axis=0)
        + np.roll(ux, 1, axis=0)
    )

    # =========================
    # Figure layout
    # =========================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))

    vmin, vmax = 0, np.percentile(vel, 99)
    wmin, wmax = -np.percentile(np.abs(omega), 99), np.percentile(np.abs(omega), 99)

    im1 = ax1.imshow(vel[0].T, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    im2 = ax2.imshow(omega[0].T, origin="lower", vmin=wmin, vmax=wmax, cmap="seismic")

    ax1.set_title("Velocity magnitude")
    ax2.set_title("Vorticity")

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    # =========================
    # Animation and save mp4
    # =========================

    t_start = args.t_start
    t_end = args.t_end if (args.t_end is not None and args.t_end < t_total) else t_total

    writer = animation.FFMpegWriter(fps=30)

    with writer.saving(fig, save_path, dpi=200):
        for t in tqdm(range(t_start, t_end), desc="Writing video"):
            # update fields
            im1.set_data(vel[t].T)
            im2.set_data(omega[t].T)

            ax1.set_title(f"Velocity - frame={t}")
            ax2.set_title(f"Vorticity - frame={t}")

            writer.grab_frame()

    plt.close(fig)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
