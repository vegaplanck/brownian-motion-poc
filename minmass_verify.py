import pims
import trackpy as tp

# Carregue o frame
raw_frames = pims.open("videos/WIN_20241127_17_26_09_Pro.mp4")
frame0 = raw_frames[2700]
if frame0.ndim == 3:
    frame0 = frame0[:, :, 0]

# Detecta todos os picos sem filtro rigoroso para ver a massa da partícula brilhante
df = tp.locate(frame0, diameter=15, minmass=100)

# Ordena pelas partículas com MAIOR massa (mais brilhantes)
print("Partículas mais brilhantes encontradas:")
print(df[["x", "y", "mass", "signal"]].sort_values(by="mass", ascending=False).head(5))