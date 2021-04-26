from scipy.ndimage import map_coordinates, interpolation

from package_functions import main_algo
import numpy as np
from PIL import Image
from datetime import datetime

im_moving = Image.open('circle.png')
im_static = Image.open('appleholes.png')
im_moving = np.array(im_moving.convert('L')) / 255
im_static = np.array(im_static.convert('L')) / 255

startTime = datetime.now()
maps, mu, reg = main_algo.segment_image_descent(im_moving, im_static, iteration=20)
print(datetime.now() - startTime)