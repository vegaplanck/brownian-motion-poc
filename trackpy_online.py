"""Reproduz um video e mostra deteccoes do trackpy em tempo real."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle
import numpy as np
import pims
import trackpy as tp


VIDEO_PATH = Path("videos/WIN_20241127_17_26_09_Pro.mp4")
DIAMETER = 15
MINMASS = 1500.0
DEFAULT_FPS = 30.0


def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Converte um frame RGB para escala de cinza, se necessario."""
    if frame.ndim == 3:
        return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exibe deteccoes do trackpy durante a reproducao do video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO_PATH,
        help="Caminho do video (padrao: %(default)s)",
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
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS para reproducao; por padrao usa o FPS do video",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Primeiro frame do trecho (inclusivo)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Ultimo frame do trecho (inclusivo)",
    )
    parser.add_argument(
        "--start-second",
        type=float,
        default=None,
        help="Segundo inicial do trecho (inclusivo)",
    )
    parser.add_argument(
        "--end-second",
        type=float,
        default=None,
        help="Segundo final do trecho (inclusivo)",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Salva a animacao anotada neste arquivo MP4",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Gera o video sem abrir a janela de visualizacao",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.diameter < 3 or args.diameter % 2 == 0:
        raise ValueError("--diameter deve ser um numero impar maior ou igual a 3.")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps deve ser maior que zero.")
    if not args.video.exists():
        raise FileNotFoundError(f"Video nao encontrado: {args.video}")
    if args.no_show and args.save_video is None:
        raise ValueError("--no-show exige --save-video.")
    frame_selection = args.start_frame is not None or args.end_frame is not None
    second_selection = args.start_second is not None or args.end_second is not None
    if frame_selection and second_selection:
        raise ValueError("Use intervalo em frames ou em segundos, nao ambos.")
    if frame_selection and (
        args.start_frame is None or args.end_frame is None
    ):
        raise ValueError("Informe --start-frame e --end-frame juntos.")
    if second_selection and (
        args.start_second is None or args.end_second is None
    ):
        raise ValueError("Informe --start-second e --end-second juntos.")
    if args.start_second is not None and (
        args.start_second < 0 or args.end_second < args.start_second
    ):
        raise ValueError("O intervalo em segundos deve ser valido e nao negativo.")
    if args.start_frame is not None and (
        args.start_frame < 0 or args.end_frame < args.start_frame
    ):
        raise ValueError("O intervalo em frames deve ser valido e nao negativo.")

    frames = pims.open(str(args.video))
    video_fps = getattr(frames, "frame_rate", None) or DEFAULT_FPS
    playback_fps = args.fps or video_fps
    interval_ms = 1000 / playback_fps

    if frame_selection:
        start_frame = args.start_frame
        end_frame = args.end_frame
    elif second_selection:
        start_frame = round(args.start_second * video_fps)
        end_frame = round(args.end_second * video_fps)
    else:
        start_frame = 0
        end_frame = len(frames) - 1

    if end_frame >= len(frames):
        raise ValueError(
            f"O video possui {len(frames)} frames; o frame final solicitado "
            f"({end_frame}) nao existe."
        )
    selected_frames = range(start_frame, end_frame + 1)

    first_frame = convert_to_grayscale(frames[start_frame])
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(first_frame, cmap="gray", vmin=0, vmax=255)
    axis.set_axis_off()
    axis.set_title(
        f"trackpy online | diameter={args.diameter}, minmass={args.minmass}"
    )
    circles = []

    def update(frame_index: int):
        nonlocal circles
        frame = convert_to_grayscale(frames[frame_index])
        detections = tp.locate(
            frame,
            diameter=args.diameter,
            minmass=args.minmass,
        )
        image.set_data(frame)

        for circle in circles:
            circle.remove()
        circles = [
            Circle(
                (detection.x, detection.y),
                radius=args.diameter / 2,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
            )
            for detection in detections.itertuples()
        ]
        for circle in circles:
            axis.add_patch(circle)

        axis.set_title(
            f"trackpy online | frame={frame_index} | "
            f"particulas={len(detections)} | "
            f"diameter={args.diameter}, minmass={args.minmass}"
        )
        return [image, *circles]

    animation = FuncAnimation(
        figure,
        update,
        frames=selected_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    # Mantem a animacao viva ate a janela ser fechada.
    figure._trackpy_animation = animation
    plt.tight_layout()

    if args.save_video is not None:
        args.save_video.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(fps=playback_fps, metadata={"artist": "trackpy"})
        print(f"Salvando video anotado em: {args.save_video}")
        animation.save(str(args.save_video), writer=writer, dpi=100)
        print("Video salvo com sucesso.")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
