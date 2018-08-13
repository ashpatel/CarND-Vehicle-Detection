## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[train_ex]: ./examples/training_ex.png
[color_car]: ./examples/color_car.png
[color_notcar]: ./examples/color_notcar.png
[find_cars]: ./examples/find_cars.png
[scale]: ./examples/scale.png
[hog4]: ./examples/hog4.png
[hog8]: ./examples/hog8.png
[hog16]: ./examples/hog16.png
[pipe1]: ./examples/pipe1.png
[pipe2]: ./examples/pipe2.png
[pipe3]: ./examples/pipe3.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

The project is implemented in the Jupyter Notebook file `VehicleDetection.ipynb`. The Code cells are referenced in this writeup are identified with a comment in the first line of each cell.


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In "# Code Cell 2", defined the function `readTrainingData()` that generates the list of all the car and not-car images to be used for training.

In "# Code Cell 5" I displayed some random selection of car and not-car images.
You can see examples of these below.


![alt text][train_ex]

Further in that same cell, I then explored different color spaces. For a randomly selected car and nor-car image I did a 3D plot of the 3 channels for a RGB, HSV, LUV and YCrCb color spaces. You can see an example of that below.

![alt text][color_car]
![alt text][color_notcar]

In "# Code Cell 3" is the code that extracts the HOG features using `skimage.hog()`. In "# Code Cell 7" I visualized the HOG features using various Color Spaces (RGB, HSV, LUV, YCrCb and YYV) and pixels_per_cell values.
You can see an example of `pixels_per_cell` with values 4,8 and 16.

![alt text][hog4]
![alt text][hog8]
![alt text][hog16]


#### 2. Explain how you settled on your final choice of HOG parameters.

I settled on 8 as the pixels_per_cell, as that gave better HOG features that looked like the outlines of cars and were better differentiated vs the not-car HOG features. Better for the LinearSVM to work against. This is also the first point where the outlines were clear, while with higher values there was a loss of too much data.

I also played around with the value of orient and settled on 11.

Using a combination of the above two visualizations and trying various random images, I settled on YCrCb as the color space that gave the better seperation in the 3D Plots for cars vs not-cars and also better HOG features.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In "#Code Cell 8" I trained the SVM using `sklearn.svm.LinearSVC`. I extracted all the features using `extract_features()` which extraced HOG features and combined them with color hist features. As discussed above, the following parameters were settled upon;

```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the sliding window search provided in the lessson as `find_cars()`, this implemented a more efficent HOG sub-sampling method where you only had to extract HOG features once per search. This is implemented in "# Code Cell 11". The example out of `find_cars` is a set of bounding boxes as shown below;

![alt text][find_cars]

In "# Code Cell 12" I ran `find_cars()` with different scales to be able to search for different car sizes. An example of all the boxes generated with that approach is below;

![alt text][scale]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In "# Code cell 15" in `process_image()` I implemented the pipeline. This used `find_cars()` with various scales, and then heatmaps and threshold to find the bounding boxes.

In "# Code cell 16" I ran the pipeline against some test images and below are the outputs;

![alt text][pipe1]
![alt text][pipe2]
![alt text][pipe3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./test_videos_output/project_video.mp4)

When debugging I created a debug video that showed the heatmap for each frame, and then the combined heatmap tresholded, so that I could see the pipeline in action. These two frames were added to the output video.

Here's a [link to my debug video result](./test_videos_output/project_video_debug.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In '# Code Cell 13' is where I implemented the functions (from the lessons) then took the boxes from `find_cars`, overlayed them as a heatmap map, and then with thresholding and labeling using `scipy.ndimage.measurements.label` ended up with a single bounding box.

In "# Code Cell 15" I implemented a `FrameHistory` class to keep track of state while processing the video. In this object I saved the last heatmaps from the previous frames, and then combined them and used a higher treshold to remove false positives. This removed most of the false positives.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My final output removed most of the false positives, but towards the end there were a few that showed up in a few frames. Currently using a sum or average of heatmaps of frames doesn't deal well with false positives that are intermittent, but over a few frames could still yield a bounding box. One optimization to the algorithm to try would be to look for boxes that don't persist in multiple concurrent frames.
