import cv2
import saliency

import matplotlib.pyplot as plt

image = cv2.imread("images/waffles.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

saliency_map = saliency.detect_saliency(image)
boundary = saliency.boundary_mask(saliency_map, threshold=0.5)
seeds= saliency.boundary_seeds(boundary)
nearest_seeds = saliency.jump_flood_algorithm(seeds)
dist_field = saliency.normalize_and_blur(nearest_seeds)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(saliency_map, cmap="gray")
plt.title("Saliency Map")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(dist_field, cmap="magma")
plt.title("Saliency Distance Field")
plt.axis("off")

plt.show()