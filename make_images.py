"""Converts a directory of .npy files into a directory of images.

One subdirectory is created for each .npy file."""
from __future__ import division, print_function
import numpy as np
import cv2 as cv
from time import time
import os


npy_dir = 'NPYs'
out_dir = 'images'

os.mkdir(out_dir)
for npy in os.listdir(npy_dir):
    print("Working on %s ..." % npy, end='')
    time_npy_start = time()
    subset = os.path.splitext(npy)[0]
    os.mkdir(os.path.join(out_dir, subset))

    flat_images = np.load(os.path.join(npy_dir, npy))
    assert flat_images.shape[1] == 28**2
    images = flat_images.reshape(-1, 28, 28)

    for k, image in enumerate(images):
        cv.imwrite(os.path.join(out_dir, subset, subset + '_%s.png' % k), image)
    print('Done (in %s s).' % (time() - time_npy_start))

