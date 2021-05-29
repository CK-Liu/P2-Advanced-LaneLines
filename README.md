## Advanced Lane Finding
---
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
## The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Navigating repository

* The main code is in `P2.ipynb`
* Camera calibration code is in `camera_cal.ipynb`
* Test images and the output of image pipeline are in `./test_images`
* Output of video pipeline are in `./output_images`. The `project_video_output.mp4` works really well, but `challenge_output.mp4` and `harder_challenge_output.mp4` not work well which will be explained in 'discussion' section.

[//]: # (Image References)

[image1]: ./tmp/camera_cal.jpg "Undistorted"
[image2]: ./test_images/undistorted2.jpg "Road Transformed"
[image3]: ./test_images/binary2.jpg "Binary Example"
[image4]: ./test_images/tracked2.jpg "Warp Example"
[image5]: ./test_images/fit_line.jpg "Fit Visual"
[image6]: ./test_images/unwarped.jpg "unwarp"
[image7]: ./test_images/output.jpg "Output"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is located in `./camera_cal.ipynb`. The calculated camera matrix and distortion coefficients is stored in `./camera_cal/calibration_pickle.p`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  

`corners` are detected by `cv2.findChessboardCorners` and will be appended to `imgpoints`. I then used the output `objpoints` and `imgpoints` to compute the camera matrix and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

At the last cell of `camera_cal.ipynb`, I used `cv2.undistort()` to correct all of the six test images in `./test_images` folder. The results are saved in `./test_images/undistorted*.jpg`. Here is an example of a distortion corrected image:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The main code including both 'image pipeline' and 'video pipeline' is `P2.ipynb`.I used a combination of color thresholding (HSV and HLS), gradients thresholding (both horizontal and vertical Sobel filter) to generate a binary image.  Here's an example of my output for this step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The key step for perspective transform is to properly select source (`src`) points and destination (`dst`) points. After perspective transform, the original binary image will be transformed into bird view image.  The Python code to select source points and destination points:

```python
img_size = (img.shape[1], img.shape[0])
bot_width = .76
mid_width = .08
height_pct = .62
bottom_trim = .935

src = np.float32([ [img_size[0]*(.5-mid_width/2), img_size[1]*height_pct], [img_size[0]*(.5+mid_width/2), img_size[1]*height_pct],\
                [img_size[0]*(.5+bot_width/2), img_size[1]*bottom_trim], [img_size[0]*(.5-bot_width/2), img_size[1]*bottom_trim] ])
offset = img_size[0]*.25
dst = np.float32([ [offset, 0], [img_size[0]-offset, 0],\
                   [img_size[0]-offset, img_size[1]], [offset, img_size[1]] ])
```

<!-- This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        | -->
If this step is correctly being done, the left lane line and right lane line will parallel after perspective transform in bird view image. Here is one example: left and right lane lines are parallel after perspective transform.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This is the most important step. The image pipeline is different from video pipeline in this step. Here I only describe the image pipeline procedure. Image pipeline and video pipeline share the same `fit_polynomial()` and `find_lane_pixels()` function. I use 'if else' inside `find_lane_pixels()` to differentiate execution when process image and video.

* Generate column histogram of input binary image
* Separate image into left half and right half, find the maximum value of histogram for each half
* Equally divide the image into 9 sub-images, sliding window to find all the left lane pixels and right lane pixels
* Fit curves using `np.polyfit()` to left lane and right lane pixels
* Calculate left lane and right lane curve and plot them on `out_img`

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After I fit curves to left and right lane line, I got the polynomial coefficients. There is an equation to calculate curvature based on polynomial coefficients. But before that, we need to build a connection between pixel and real-world dimension. Here are all the Python code to calculate curvature and the position with respect to center.

``` python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

# Calculation of R_curve (radius of curvature)
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])  
print(left_curverad, 'm', right_curverad, 'm')
curverad = min(left_curverad, right_curverad)

camera_center = (left_fitx[-1] + right_fitx[-1])/2
center_diff = (camera_center - binary_warped.shape[1]/2)*xm_per_pix
side_pos = 'left'
if center_diff <= 0:
   side_pos = 'right'
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Then I unwarped the bird view image into the original perspective using `cv2.warpPerspective()`, but this time we use `Minv` inverse matrix instead of `M` matrix. Here is the previous image after unwarp:

![alt text][image6]

In the end, I combine the original color input image with `step 5` lane line drawing image together using `weighted_img()`. I also output the curvature and position of the car using `cv2.putText()`. Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to project video result](output_images/project_video_output.mp4); [link to challenge video result](output_images/challenge_output.mp4); [link to harder challenge video result](output_images/harder_challenge_output.mp4);

The main difference video pipeline and image pipeline is about how to find left and right lane line pixels. Instead of using sliding window to search for lane lines in each frame, video pipeline define a region of interest (ROI). Since the position of the lane lines do not change so much, we can define a `margin=30` around previous fitted polynomial curve. It accelerate the algorithm execution speed and also makes it more robust. We use 'global' variable to store the previous frame's information and the code is shown:

```Python
global left_video
global right_video
global i_video

left_lane_inds = ((nonzerox > (left_video[i_video][0]*(nonzeroy**2) + left_video[i_video][1]*nonzeroy+ left_video[i_video][2] - margin)) & (nonzerox < (left_video[i_video][0]*(nonzeroy**2) + left_video[i_video][1]*nonzeroy + left_video[i_video][2] + margin)))

right_lane_inds = ((nonzerox > (right_video[i_video][0]*(nonzeroy**2)+right_video[i_video][1]*nonzeroy + right_video[i_video][2] - margin)) & (nonzerox < (right_video[i_video][0]*(nonzeroy**2) + right_video[i_video][1]*nonzeroy + right_video[i_video][2] + margin)))
```
We also try to use the number of pixels found on lane lines to define our confidence of the lane lines being found. If there are not enough pixels being found, we still use sliding window to find lane lines. But due to the time limitation, the parameters have not being fine tuned.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project works really well with `project_video.mp4`, but it fails to process `challenge_video.mp4` and `harder_challenge_video.mp4`.

For `challenge_video.mp4`, there is big 'black crack' on the road and the shadow of highway barriers, which will cause problems to our `Sobel filter`.  These problems may be able to solve by fine tune `Sobel filter` and `color threshold` parameters.

For `harder_challenge_video.mp4`, the curve is large. When we arrive left/right edge of an image, the number of lane line pixels found using sliding window method is not achieved `minpix`, so the starting position of next window doesn't change. This will leads to very few lane line pixels have been found which will influence polynomial fit. This problem may be solved by decrease `minpix` value or enlarge `margin`.

Outlier rejection, low-pass filter and add weighted mean to each new detection may also be able to help. But I don't have enough time before the deadline.
