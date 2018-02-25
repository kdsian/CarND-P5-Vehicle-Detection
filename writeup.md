## Writeup

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

[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
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

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 1~12 code cell of the IPython notebook (the file called `P5-Vehicle-Detection.ipynb`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

<center><img src="./example_images/data_look.png"></center>

Color space을 변경하는 과정을 수행했습니다.

우선 주어진 test 이미지 12개에 대해 RGB 영상의 histogram 을 plot 해보았습니다.

<center><img src="./example_images/Histogram_Color.png"></center>

ColorSpace1 의 car 이미지와 ColorSpace6의 배경 이미지가 크게 차이가 없는 것을 볼 수 있습니다. 

따라서 더 좋은 결과를 위해서 color sapce 을 변경 할 필요가 있다고 판단 했습니다.

어떤 color sapce로 변경하면 좋을지 판단하기 위해 여러 가지 plot 을 하여 car의 특징이 잘 categorization 되어 있는 color space 을 눈으로 찾아보았습니다.

<center><img src="./example_images/color_scope_car.png"></center>

<center><img src="./example_images/color_scope_noncar.png"></center>


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

<center><img src="./example_images/sliding_windows.jpg"></center>


#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

 linear SVM 에 입력 데이터를 넣기 전에 feature에 대해 normalization 작업을 수행했습니다.

<center><img src="./example_images/Normalized_Features.png"></center>


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Hog 알고리즘을 sub-sampling window을 적용함으로써 sliding window search를 수행했습니다,
<center><img src="./example_images/multi-sub.jpg"></center>




#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

<center><img src="./example_images/box.png"></center>
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 1 frames and their corresponding heatmaps:

<center><img src="./example_images/threshold.png"></center>

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
<center><img src="./examples/labels_map.png"></center>


### Here the resulting bounding boxes are drawn onto the 6 frame in the series:

<center><img src="./example_images/result_test1.jpg"></center>
<center><img src="./example_images/result_test2.jpg"></center>
<center><img src="./example_images/result_test3.jpg"></center>
<center><img src="./example_images/result_test4.jpg"></center>
<center><img src="./example_images/result_test5.jpg"></center>
<center><img src="./example_images/result_test6.jpg"></center>



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

흰 차량을 잘 찾지 못하는 것을 영상에서 볼 수 있었는데, 성능 향상을 위해서 좀 더 고려해보아야할 것 같습니다.

