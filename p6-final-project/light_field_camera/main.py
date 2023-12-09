import os
import cv2
import math
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm


def load_images(dataset):
    # Load if cached
    cache_path = f'datasets/{dataset}.npy'
    if os.path.exists(cache_path):
        return np.load(cache_path)

    # Load all images into np array
    paths = list(glob(f'datasets/{dataset}/*.png'))
    size = int(math.sqrt(len(paths)))
    assert size ** 2 == len(paths), f'Invalid number of images for {size}x{size} microlens array'

    array = [[None for _ in range(size)] for _ in range(size)]
    for path in tqdm(paths, desc='Load images'):
        name = os.path.basename(path)
        row, col = map(int, name.split('_')[1:3])
        array[row][col] = cv2.imread(path)
    result = np.array(array)
    np.save(cache_path, result)
    return result


def shift(target, dx, dy):
    result = np.zeros_like(target)
    h, w, _ = target.shape
    cropped = target[max(0, -dy):min(h, h - dy),
                     max(0, -dx):min(w, w - dx)]
    result[max(0, dy):min(h, h + dy),
           max(0, dx):min(w, w + dx)] = cropped
    return result


def refocus(imgs, depth, aperture=float('inf')):
    nrows, ncols, height, width, _ = imgs.shape

    # Refocus the image at the given depth
    result = []
    center = np.array([nrows // 2, ncols // 2])
    for i in range(nrows):
        for j in range(ncols):
            # Only include images inside the aperture
            if abs(i - center[0]) > aperture or abs(j - center[1]) > aperture:
                continue

            # Shift according to offset from the center
            dx = int((j - center[1]) * depth)
            dy = -int((i - center[0]) * depth)
            result.append(shift(imgs[i, j, :, :, :], dx, dy))
    return np.mean(np.array(result), axis=0)


###############
#   Refocus   #
###############
def depth_sweep():
    folder = f'results/{DATASET}/refocus_{DEPTH_RANGE[0]:.1f}_{DEPTH_RANGE[1]:.1f}'
    if not os.path.exists(folder):
        os.makedirs(folder)

    imgs = load_images(DATASET)
    frames = []
    depths = np.linspace(DEPTH_RANGE[0], DEPTH_RANGE[1], NUM_FRAMES)
    for i, d in enumerate(tqdm(depths)):
        result = refocus(imgs, d)
        frames.append(np.flip(result, axis=-1).astype('uint8'))
        cv2.imwrite(f'{folder}/{d:.3f}.png', result)

    frames = [frames[0]] * PADDING \
             + frames \
             + [frames[-1]] * PADDING \
             + list(reversed(frames))
    imageio.mimsave(os.path.join(folder, 'final.gif'), frames, loop=0, duration=int(1000 / FPS))


#################
#   Aperture    #
#################
def aperture_sweep():
    folder = f'results/{DATASET}/aperture_{APERTURE_RANGE[0]:.1f}_{APERTURE_RANGE[1]:.1f}_{DEPTH:.1f}'
    if not os.path.exists(folder):
        os.makedirs(folder)

    imgs = load_images(DATASET)
    frames = []
    apertures = np.linspace(APERTURE_RANGE[0], APERTURE_RANGE[1], NUM_FRAMES)
    for i, a in enumerate(tqdm(apertures)):
        result = refocus(imgs, DEPTH, aperture=a)
        frames.append(np.flip(result, axis=-1).astype('uint8'))
        cv2.imwrite(f'{folder}/{a:.3f}.png', result)

    frames = [frames[0]] * PADDING \
             + frames \
             + [frames[-1]] * PADDING \
             + list(reversed(frames))
    imageio.mimsave(os.path.join(folder, 'final.gif'), frames, loop=0, duration=int(1000 / FPS))


if __name__ == '__main__':
    DATASET = 'lego'
    NUM_FRAMES = 30
    PADDING = 5
    FPS = 15

    DEPTH_RANGE = (-1.2, 8)
    APERTURE_RANGE = (0, 7)
    DEPTH = 8

    depth_sweep()
    aperture_sweep()
