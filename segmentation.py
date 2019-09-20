import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage import io
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--polygon', required=True, help='Path to the polygon file')
args = vars(ap.parse_args())


x = [map(eval,l.strip().split('\t')) for l in open(args['polygon'])]
x = [tuple(i) for i in x]
img = np.zeros((484,484), 'uint8')

# fill polygon
for i in x:
    poly = np.array(i)
    rr, cc = polygon(poly[:,0], poly[:,1], img.shape)
    img[rr,cc] = 255

img = 255 - img

io.imsave('Y-%s.png' %(args['polygon'].split(".")[0]), img)