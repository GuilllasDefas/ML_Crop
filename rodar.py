import pickle
import numpy as np
import matplotlib.pyplot as plt

# Carregar o cache
with open("models/bbox_cache.pkl", "rb") as f:
    cache = pickle.load(f)

bbox_data = cache["bboxes"]

# Rodar diagnóstico
bboxes = np.array(bbox_data)

widths  = bboxes[:, 2] - bboxes[:, 0]
heights = bboxes[:, 3] - bboxes[:, 1]
areas   = widths * heights

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(widths, heights, alpha=0.4)
axes[0].set_xlabel("Largura normalizada")
axes[0].set_ylabel("Altura normalizada")
axes[0].set_title("Distribuição de tamanhos dos crops")

axes[1].scatter(bboxes[:, 0], bboxes[:, 1], alpha=0.4)
axes[1].set_xlabel("x1 (início esquerda)")
axes[1].set_ylabel("y1 (início topo)")
axes[1].set_title("Posição dos crops")

axes[2].hist(areas, bins=30)
axes[2].set_xlabel("Área normalizada")
axes[2].set_title("Distribuição de área dos crops")

plt.tight_layout()
plt.savefig("diagnostico_dataset.png", dpi=150)
plt.show()

print(f"Total de amostras: {len(bboxes)}")
print(f"\nÁrea  — média: {areas.mean():.3f} | std: {areas.std():.3f} | min: {areas.min():.3f} | max: {areas.max():.3f}")
print(f"x1    — média: {bboxes[:,0].mean():.3f} | std: {bboxes[:,0].std():.3f}")
print(f"y1    — média: {bboxes[:,1].mean():.3f} | std: {bboxes[:,1].std():.3f}")
print(f"x2    — média: {bboxes[:,2].mean():.3f} | std: {bboxes[:,2].std():.3f}")
print(f"y2    — média: {bboxes[:,3].mean():.3f} | std: {bboxes[:,3].std():.3f}")

# Quanto do dataset está em cada pico?
pico_menor = (areas < 0.28).sum()
pico_maior = (areas >= 0.28).sum()
print(f"Crops pequenos (<0.28): {pico_menor} ({pico_menor/len(areas)*100:.1f}%)")
print(f"Crops grandes (>=0.28): {pico_maior} ({pico_maior/len(areas)*100:.1f}%)")