import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def detect_saliency(image):
    # uses Frequency-tuned Saliency to identify borders
    color = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean_color = np.mean(np.mean(color, axis=0), axis=0)
    # compute Euclidean distance
    diff = np.linalg.norm(color - mean_color, axis=2)
    # normalize [0,1]
    salient = (diff - diff.min()) / (diff.max() - diff.min())
    return salient

def boundary_mask(salient, threshold):
    # makes salient mask based on given threshold (is this considered an important region)
    foreground = (salient >= threshold).astype(np.uint8)
    # mask of neighbors
    kernel = np.ones((3, 3), dtype=np.uint8)
    # slowly shrinks foreground
    eroded = cv2.erode(foreground, kernel, iterations=1)
    boundary = foreground - eroded
    return boundary

def boundary_seeds(boundary):
    h,w = boundary.shape
    seeds = np.zeros((h, w, 2), dtype = int)

    # no initial best seed
    seeds[:, :, 0] = -1 # y coord
    seeds[:, :, 1] = -1 # x coord

    y, x = np.where(boundary == 1)
    for (j, i) in zip(y, x):
        seeds[j, i] = (j, i)
    return seeds

def jump_flood_algorithm(seeds):
    # basic logic to start with a pixel and progressively 
    # makes smaller jumps each time to check where the 
    # nearest boundary pixel is

    h,w,_ = seeds.shape

    max_size = max(h,w)
    jump = 1
    while jump * 2 <= max_size:
        jump *= 2
    
    curr_seeds = seeds.copy()

    # neighboring jumps in all 8 directions
    directions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, -1)]

    while jump >= 1:
        next = curr_seeds.copy()
        for y in range(h):
            for x in range(w):
                best = curr_seeds[y, x]
                if (best[0] == -1):
                    min_dist = np.inf
                else:
                    min_dist = np.linalg.norm([best[0] - y, best[1] - x])

                for dy, dx in directions:
                    y_neighbor = y + (dy * jump)
                    x_neighbor = x + (dx * jump)

                    # ensures neighbors are not out-of-bounds
                    if not (h > y_neighbor >= 0 and w > x_neighbor >= 0):
                        continue

                    neighbor = curr_seeds[y_neighbor, x_neighbor]

                    if neighbor[0] == -1:
                        continue

                    dist = np.linalg.norm([neighbor[0] - y, neighbor[1] - x])
                    if dist < min_dist:
                        min_dist = dist
                        best = neighbor
                next[y, x] = best
        curr_seeds = next
        jump = jump // 2
    return curr_seeds

def normalize_and_blur(seeds):
    h,w,_ = seeds.shape
    abs_dist = np.zeros((h, w), dtype=float)

    for y in range(h):
        for x in range(w):
            j, i = seeds[y, x]
            if j == -1:
                abs_dist[y, x] = np.inf
            else:
                abs_dist[y, x] = np.linalg.norm([j - y, i - x])
    
    # prevent division by inf
    fixed = abs_dist[np.isfinite(abs_dist)]
    if fixed.size == 0:
        max_val = 1
    else:
        max_val = fixed.max()
    # creates bool mask
    abs_dist[~np.isfinite(abs_dist)] = max_val
    norm_dist = abs_dist / abs_dist.max()
    
    blurred = gaussian_filter(norm_dist, sigma=3.0)
    return blurred

def saliency_distance_field(image):
    saliency_map = detect_saliency(image)
    salient_boundary = boundary_mask(saliency_map, threshold=0.5)
    seeds = boundary_seeds(salient_boundary)
    best_seeds = jump_flood_algorithm(seeds)
    dist_field = normalize_and_blur(best_seeds)

    return dist_field