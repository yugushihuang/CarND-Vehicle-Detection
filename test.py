import cv2
import os
import numpy as np
from math import *
import glob
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import itertools

from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from training_functions import *
np.random.seed(0xdeadbeef)

# Parameters
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

# Display Example Image
cars = glob.glob('data/vehicles/**/*.png')
notcars = glob.glob('data/non-vehicles/**/*.png')
car = mpimg.imread(cars[1])
notcar = mpimg.imread(notcars[0])

plt.subplot(1,2,1)
plt.title('Car')
plt.imshow(car)
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(notcar)
plt.title('Not Car')
plt.xticks([])
plt.yticks([])
plt.show()

# HOG Image
img = (car*255).astype(np.uint8)
from skimage.feature import hog
from skimage import data, color, exposure

hog_feat, hog_image = hog(img[:,:,0], orientations=orient, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

# Preview
plt.subplot(1,2,1)
plt.title('RGB')
plt.imshow(img, cmap='gray')

plt.subplot(1,2,2)
plt.imshow(hog_image, cmap='gray')
plt.title('HoG - Y Channel - YUV')
plt.show()

# Create Windows
from detection_functions import create_windows
image = mpimg.imread('test_images/test1.jpg')
pyramid = [((64, 64),  [400, 500]),
           ((96, 96),  [400, 500]),
           ((128, 128),[450, 600]),
              ]
image_size = image.shape[:2]
windows = create_windows(pyramid, image_size)
print(len(list(itertools.chain(*windows))))
for p1, p2 in itertools.chain(*windows):
    cv2.rectangle(image, p1, p2, (15,15,200), 4)
plt.imshow(image)
plt.show()


