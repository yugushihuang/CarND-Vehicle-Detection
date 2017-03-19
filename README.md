# Self Driving Car Nanodegree - Vehicle Detection and Tracking

## README

The code I used for doing this project can be found in `main.py` and `training.py`. The main file imports functions defined in other files -- `training_functions.py`, `detection_functions.py` and `tracking_functions.py`.

### Usage

```
  MODEL             SKLearn model file created using training.py
  INPUT             Input video file
  OUTPUT            Output video filename
```

To run the program on a video:

`python main.py`

Trained SKlearn classifier model files are stored in the `models` folder. The best one so far is `models/clf.pkl`

---
[//]: # (Image References)
[image1]: ./figures/car_not_car.png
[image2]: ./figures/HOG_example.png
[image3]: ./figures/sliding_windows.png
[image4]: ./figures/sliding_window_output.png
[image5]: ./figures/img50.jpg
[image6]: ./figures/example_output.jpg
[video1]: ./output/project_video_out.mp4

## Histogram of Oriented Gradients (HOG)

### 1. Loading the training data

The code for this step is contained in the second step of the python file `training.py` as well as lines 13-134 of `training_functions.py`.

First, the `glob` package is used to read in all the files in the data-set. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

### 2. HOG Parameters

Different colorspaces and HoG parameters were explored. The final configuration was chosen according to the one which gave the best test-set accuracy from the classifier (described later). Here is an example using the Y channel of the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

*Note: OpenCV's `HOGDescriptor` class was used in place of `skimage.hog` as it was found to be significantly faster. The method `init_hog` in `training_functions.py` initializes the descriptor with the parameters and then it is re-used for the rest of the process.*


### 3. Training the classifier

A `LinearSVC` was used as the classifier for this project. The training process can be seen in step 3 of `training.py`. The features are extracted and concatenated using functions in `training_functions.py`. Since the training data consists of PNG files, and because of how `mpimg.imread` loads PNG files, the image data is scaled up to 0-255 before being passed into the feature extractor.

The features include HOG features, spatial features and color histograms. The classifier is set up as a pipeline that includes a scaler as shown below:

```python
clf = Pipeline([('scaling', StandardScaler()),
                ('classification', LinearSVC(loss='hinge')),
               ])
```
This keeps the scaling factors embedded in the model object when saved to a file. One round of hard negative mining was also used to enhance the results. This was done by making copies of images that were identified as false positives and adding them to the training data set. This improved the performance of the classifier. The model stored in the file `models/clf.pkl` obtained a test accuracy of 98.85% where the test-set was 20% of the total data.

## Sliding Window Search

### 1. Scales of Windows and Overlap

The function `create_windows` in lines 141-147 of `detection_functions.py` was used to generate the list of windows to search. The input to the function specifies both the window size, along with the 'y' range of the image that the window is to be applied to. Different scales were explored and the following set was eventually selected based on classification performance:

| Window size | Y-range     |
|-------------|-------------|
| (64, 64)    | [400, 500]  |
| (96, 96)    | [400, 500]  |
| (128, 128)  | [450, 600] |

![alt text][image3]

#### 2. Image Pipeline

The final image used the Y-channel of a YUV-image along with the sliding window approach shown above to detect windows of interest. These windows were used to create a heatmap which was then thresholded to remove false positives. This strategy is of course more effective in a video stream where there the heatmap is stacked over multiple frames. `scipy.ndimage.measurements.label` was then used to define clusters on this heatmap to be labeled as possible vehicles. This is shown in lines 48-56 of the  `process_image` function in `main.py`. The heatmap is created using the `update_heatmap` function in `detection_functions.py` in lines 171-186.

In the video pipeline (described in the next section), the output of the `label` function was used as measurements for a Kalman Filter which estimated the corners of the bounding box.

![alt text][image4]
---

## Video Implementation

### 1. Video Output

#### Project Video

The output video can be found at:  [project_video_out.mp4](./output/project_video_out.mp4)

### 2. Kalman Filter and Thresholding

#### Heatmap and Clustering

The pipeline worked by first creating a heatmap using all the windows marked as positive by the classifier. The first step to reducing false positives was to use classifier's decision function rather than the `predict` method. By setting a threshold on the decision function as shown below, many spurious detections were avoided:

```python
dec = clf.decision_function(test_features)
prediction = int(dec > 0.75)
```

The heatmap was then added up over 25 frames, and any pixels below a threshold of 10 were zeroed out. This excluded any detections that didn't show up consistently in 10 out of the last 25 frames.

#### Vehicle Tracking

The process of tracking the vehicle, given a set of bounding boxes from the thresholded heatmap is performed by the `VehicleTracker` class in `tracking.py`. After the clusters were extracted from the heatmap using `scipy.ndimage.measurements.label`, the bounding boxes were passed in as measurements to a Kalman Filter.

The tracking process is mainly done by the `track` method of the `VehicleTracker` class. The first step in the process is to perform non-maximum suppression of the bounding boxes to remove duplicates.

The following rules were used to identify possible candidates for tracking. These are implemented in lines 182-208 of `tracking_functions.py`.

1. If there are no candidates being tracked, assume all the non-max suppressed bounding boxes as possible tracking candidates.

2. If there are candidates currently being tracked, compute the overlap between the measured bounding boxes as well as the distance between the centroids of these boxes and the candidates.

3. The measured bounding boxes are assigned to a candidates based on the overlap as well as the distance between the centroids.

Once each measurement has been assigned to an existing (or new) candidate, the Kalman Filter is used to update the state of each candidate. The filter uses the x and y positions of the top-left and bottom-right corners of the bounding box as the state along with their second and third derivatives.

#### Cleanup

The `cleanup` method in the `VehicleTracker` class removes stray vehicle candidates that haven't haven't been seen in a few frames or those that have gone beyond a pre-defined area of interest. This is implemented in lines 238-241 of `tracking.py`.

Each candidate also has an "age" value that is initially set to zero. Once the age hits `-50`, the variable is locked, and the  candidate is assumed to be valid vehicle detection and permanently tracked even if it is not seen for a few frames. The Kalman Filter estimates the velocity of the bounding box and predicts the position whenever the measurements are lacking. A permanently tracked vehicle is only deleted once it goes outside a certain region-of-interest (horizon). This helps maintain the tracking even when vehicles are obscured.

The is decremented by one any time the candidate has at least one measurement assigned to it. Conversely, the age is increased if no measurements are assigned to a candidate in a frame (assuming it's age > -50). Once the age reaches `5`, the candidate is considered a false positive and removed.

There is also a threshold on the age at which a candidate is drawn on the image. An orange bounding box indicates that the covariance estimated by the Kalman filter has increased beyond a certain limit. This happens when a vehicle candidate is obscured behind another and hasn't been detected for a few frames.

---

### Discussion

The pipeline still detects a few false positives. One thing that can help make this more reliable would be a way to detect the horizon and automatically mask out only those places where a vehicle can show up.

Using convolutional neural networks for the initial segmentation of the image might be much faster than a Support Vector Machine classifier such as the one used here. This might enable real-time vehicle detection and tracking.

