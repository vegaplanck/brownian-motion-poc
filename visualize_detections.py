"""Visualiza as particulas detectadas pelo trackpy em um frame do video."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pims
import trackpy as tp


# Ajuste estes valores para uma execucao sem argumentos.
VIDEO_PATH = Path("videos/WIN_20241127_17_26_09_Pro.mp4")
FRAME_INDEX = 0
FRAME_START = None
FRAME_END = None
FRAME_STEP = 1
DIAMETER = 15
MINMASS = 1500.0


def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Converte um frame RGB para escala de cinza, se necessario."""
    if frame.ndim == 3:
        return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mostra um frame com as deteccoes feitas pelo trackpy."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO_PATH,
        help="Caminho do video (padrao: %(default)s)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Indice de um unico frame a visualizar",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=FRAME_START,
        help="Primeiro frame do intervalo (inclusivo)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=FRAME_END,
        help="Ultimo frame do intervalo (inclusivo)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=FRAME_STEP,
        help="Intervalo entre frames consecutivos (padrao: %(default)s)",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=DIAMETER,
        help="Diametro impar esperado das particulas (padrao: %(default)s)",
    )
    parser.add_argument(
        "--minmass",
        type=float,
        default=MINMASS,
        help="Massa minima para aceitar uma deteccao (padrao: %(default)s)",
    )
    return parser.parse_args()


def select_frame_indices(args: argparse.Namespace) -> list[int]:
    """Monta a selecao de frames a partir dos argumentos informados."""
    has_interval = args.start_frame is not None or args.end_frame is not None
    if args.frame is not None and has_interval:
        raise ValueError("Use --frame ou --start-frame/--end-frame, nao ambos.")

    if args.frame is not None:
        return [args.frame]
    if not has_interval:
        return [FRAME_INDEX]
    if args.start_frame is None or args.end_frame is None:
        raise ValueError("Informe --start-frame e --end-frame juntos.")
    if args.step <= 0:
        raise ValueError("--step deve ser maior que zero.")
    if args.start_frame > args.end_frame:
        raise ValueError("--start-frame deve ser menor ou igual a --end-frame.")

    return list(range(args.start_frame, args.end_frame + 1, args.step))


def plot_frame(frame: np.ndarray, detections, frame_index: int, args) -> None:
    """Exibe um frame com as deteccoes encontradas pelo trackpy."""
    print(f"Frame: {frame_index}")
    print(f"Particulas detectadas: {len(detections)}")
    print(detections[["x", "y", "mass", "signal"]].to_string(index=False))

    plt.figure(figsize=(8, 6))
    tp.annotate(detections, frame)
    plt.title(
        f"Deteccoes do trackpy | frame {frame_index} | "
        f"diameter={args.diameter}, minmass={args.minmass}"
    )
    plt.tight_layout()
    plt.show()
    plt.close()


def main() -> None:
    args = parse_args()
    frame_indices = select_frame_indices(args)

    if args.diameter < 3 or args.diameter % 2 == 0:
        raise ValueError("--diameter deve ser um numero impar maior ou igual a 3.")
    if any(frame_index < 0 for frame_index in frame_indices):
        raise ValueError("Os indices dos frames devem ser maiores ou iguais a zero.")
    if not args.video.exists():
        raise FileNotFoundError(f"Video nao encontrado: {args.video}")

    raw_frames = pims.open(str(args.video))
    invalid_frames = [
        frame_index for frame_index in frame_indices if frame_index >= len(raw_frames)
    ]
    if invalid_frames:
        raise IndexError(
            f"O video possui {len(raw_frames)} frames; "
            f"frame(s) invalidos: {invalid_frames}."
        )

    for frame_index in frame_indices:
        frame = convert_to_grayscale(raw_frames[frame_index])
        detections = tp.locate(
            frame,
            diameter=args.diameter,
            minmass=args.minmass,
        )
        plot_frame(frame, detections, frame_index, args)


if __name__ == "__main__":
    main()
