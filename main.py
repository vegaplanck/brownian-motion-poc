import matplotlib.pyplot as plt
import numpy as np
import pims
import trackpy as tp


# Função para converter qualquer frame do PIMS/PyAV em escala de cinza 2D
def convert_to_grayscale(frame):
    if frame.ndim == 3:
        # Média ponderada dos canais RGB (padrão ITU-R 601-2)
        return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
    return frame


def main():
    # 1. Carregar o vídeo com PIMS
    video_path = "videos/WIN_20241127_17_26_09_Pro.mp4"  # Altere para o caminho correto do seu arquivo
    raw_frames = pims.open(video_path)

    # Aplica a conversão de escala de cinza em cada frame carregado
    frames = [convert_to_grayscale(f) for f in raw_frames]

    # 2. Parâmetros de rastreamento
    DIAMETER = 15  # Número ímpar (medido ~14px)
    MINMASS = 1500.0  # Ajuste conforme o brilho das suas partículas
    SEARCH_RANGE = 15
    MEMORY = 3
    MIN_FRAMES = 10

    # 3. Validação visual no primeiro frame
    print("Validando detecção no Frame 0...")
    f_teste = tp.locate(frames[0], diameter=DIAMETER, minmass=MINMASS)

    plt.figure(figsize=(8, 6))
    tp.annotate(f_teste, frames[0])
    plt.title("Validação do Reconhecimento (Frame 0)")
    plt.show()

    # 4. Processamento em Lote
    # Importante: processes=1 evita o erro do gerador do PyAV em leituras paralelas
    print("Processando frames em lote (modo sequencial)...")
    features = tp.batch(
        frames, diameter=DIAMETER, minmass=MINMASS, processes=1
    )

    # 5. Conectar Posições ao Longo do Tempo (Linking)
    print("Conectando trajetórias...")
    trajectories = tp.link(features, search_range=SEARCH_RANGE, memory=MEMORY)
    trajectories = tp.filter_stubs(trajectories, threshold=MIN_FRAMES)

    # 6. Salvar e exibir o CSV
    output_path = "posicoes_particulas.csv"
    trajectories[["frame", "particle", "x", "y"]].to_csv(
        output_path, index=False
    )

    print("\n--- Primeiras posições extraídas ---")
    print(trajectories[["frame", "particle", "x", "y"]].head())
    print(f"\nSucesso! O arquivo '{output_path}' foi criado com as posições.")


if __name__ == "__main__":
    main()