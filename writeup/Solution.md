
# Rubric 1: Writeup/Readme
### Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point. 

You are reading it!!


```python
# import necessary modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from scipy.ndimage.measurements import label

%matplotlib inline
```

# Rubric 2: HOG Features
### Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

Following code (borrowed from Udacity lessons), extracts HOG features.
bin_spatial() and color_hist() functions can extract spatial features and colour histograms, respectively.

I settled on extracting HOG feautres from all three channels of YUV colour space, 9 orientations, 8 pixels per cell and 2 cells per block. I also added spatial features and colour histograms. My main objective of selecting these features was to achieve optimal test accuracy. After little experimentation I found that the configuration was giving more than or equal to 98% accuracy on test samples.


```python
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    # only extract bins, no need for bin edges
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

```

 _ extract_\_ _features() _ combines extracting HOG, spatial and binned features from an array of images.


```python

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    featureTypeCount=0
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB)
        #image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
            featureTypeCount+=1
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
            featureTypeCount+=1
        if hog_feat == True:
            featureTypeCount+=1
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            #print(spatial_features.size, hist_features.size, hog_features.size)
            #print(spatial_features.shape, hist_features.shape, hog_features.shape)
            #print(type(spatial_features), type(hist_features), type(hog_features))
            file_features.append(hog_features)
            
        if featureTypeCount==1:
            features.append(file_features)
        else:
            features.append(np.concatenate(file_features))
        #features.append(hog_features)
    # Return list of feature vectors
    return features


```

## Load Data for car and non-car samples

Samples for cars and non-car images have been downloaded and extracted into 'vehicles' and 'non-vehicles' folders respectively. Following code loades their paths to memory into two separate lists.


```python
cars = []
notcars = []

images = glob.glob('./*vehicles/*/*.png')

for i_image in images:
    if 'non' in i_image:
        notcars.append(i_image)
    else:
        cars.append(i_image)
        
#print(type(i_image))
print('Total # of cars:',len(cars),', # of non cars: ',len(notcars))
```

    Total # of cars: 8792 , # of non cars:  8968


Since number of samples in positive and negative classes are about equal, I don't have to increase samples of either class.

## Visualise sample data and HOG Features


```python
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))
    
# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])


# Plot the examples
fig = plt.figure(figsize=(10,6))
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image', fontsize=20)
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image', fontsize=20)
plt.show()

grayCar = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
grayNotCar = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2GRAY)
# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
# Call our function with vis=True to see an image output
features, car_hog_image = get_hog_features(grayCar, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

features, notCar_hog_image = get_hog_features(grayNotCar, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

# Plot the examples
fig = plt.figure(figsize=(10,6))
plt.subplot(121)
plt.imshow(car_hog_image, cmap='gray')
plt.title('Example Car HOG', fontsize=20)
plt.subplot(122)
plt.imshow(notCar_hog_image, cmap='gray')
plt.title('Example Non-car HOG', fontsize=20)
plt.show()
```


![png](output_10_0.png)



![png](output_10_1.png)


# Rubric 3: Training the classifier
### Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Following code extracts HOG features from 'YUV' colour space, from car and non-car images. After experimentation with different colour spaces (RGB, YUV, etc.), it is found that YUV works satisfactarily well. Also, spatial intensities and binned colour features are used with HOG features.


```python

color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # True # Spatial features on or off
hist_feat = True # True # Histogram features on or off
hog_feat = True # True # HOG features on or off

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
#scaled_X = X
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

```

    /home/manoj/.conda/envs/tfcpu/lib/python3.6/site-packages/skimage/feature/_hog.py:119: skimage_deprecation: Default value of `block_norm`==`L1` is deprecated and will be changed to `L2-Hys` in v0.15
      'be changed to `L2-Hys` in v0.15', skimage_deprecation)


## Training the classifier

A linear SVM is used to train a classifier over cars and non-car samples.


```python
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
```

    Using: 9 orientations 8 pixels per cell and 2 cells per block
    Feature vector length: 6156
    14.37 Seconds to train SVC...
    Test Accuracy of SVC =  0.9924


# Rubric 4: Sliding Window
### Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

Following cell contains code for sliding windows to detect cars in an image with multiple scales. Scales and windows overlap was decided after little experimentations.

Different scales help to detect cars of different sizes. In the final pipeline (in detectVehicles() function), I have used scales of 1x, 1.5x, 2x, 2.5x and 3x.


```python
# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list
    
```


```python

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows   

```

# Rubric 5: Demonstration
### Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?

Linear SVM classifier is used to detect cars in an image. In order to optimize test accuracy I experimented with different colour channels, HOG features alone, and HOG features combined with spatial and colour channel intensities.

## Test Classifier


```python
t=time.time() # Start time
images = glob.glob('./test_images/*.jpg')
for i_image in images:
    
    image =cv2.cvtColor(cv2.imread(i_image),cv2.COLOR_BGR2RGB)
    draw_image = np.copy(image)
    #image = image.astype(np.float32)/255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400,640], 
                        xy_window=(128, 128), xy_overlap=(0.85, 0.85))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

    plt.figure(fisize = (10,6))
    plt.title(i_image.split('/')[2])
    plt.imshow(window_img)


```

    /home/manoj/.conda/envs/tfcpu/lib/python3.6/site-packages/skimage/feature/_hog.py:119: skimage_deprecation: Default value of `block_norm`==`L1` is deprecated and will be changed to `L2-Hys` in v0.15
      'be changed to `L2-Hys` in v0.15', skimage_deprecation)



![png](output_20_1.png)



![png](output_20_2.png)



![png](output_20_3.png)



![png](output_20_4.png)



![png](output_20_5.png)



![png](output_20_6.png)


# Rubric 7: Heat Map remove false pasitive

### Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Following cell has code to generate heat map, i.e. area where multiple detection were found, will be brightest, and area where no vehicle was detected, appears darkest.

apply_threshold() function helps in rejecting area with false positive. The idea is, if there is true positive, then it will be detected many times.


Following cell contains code for adding heat to an empty heat map, then applying threshold to heat map, drawing boxes and labels, and other helping function.


```python
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
```

Following cell has find_cars(), which extracts features from images, detects vehicles and then returns list of boxes. This function does not extract HOG features every time, rather extracts them once for an entire image and then reuses them for smaller images.


```python
def find_cars(img,color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    #print("orient = {}, pix_per_cell = {}, cell_per_block={}, spatial_size={}, hit_bins={}".format(orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    boxes=[]    
    draw_img = np.copy(img)
    #img = img.astype(np.float32)/255
    
    
    if color_space != 'RGB':
        if color_space == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    
    img_tosearch = img[ystart:ystop,:,:]
    #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    ctrans_tosearch=img_tosearch
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            #i_features = []
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            #i_features.append.(spatial_features)
            hist_features = color_hist(subimg, nbins=hist_bins)
            #i_features.append.(hist_features)
            # Scale features and make a prediction
#            print(type(spatial_features), type(hist_features), type(hog_features))
#            print(spatial_features.shape, hist_features.shape, hog_features.shape)
            t=np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(t)    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            #test_prediction = svc.predict(hog_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    #return draw_img
    return boxes
```

Generate heat map and apply threshold for a test image.


```python
# Read in image similar to one shown above 
image = cv2.cvtColor(cv2.imread('./test_images/test4.jpg'), cv2.COLOR_BGR2RGB)
heat = np.zeros_like(image[:,:,0]).astype(np.float)

box_list = []

ystart = 400
ystop = 696
colour_space='YUV'

scale = 1.0
ystart = 400
ystop = 500
box_list.extend(find_cars(image, colour_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

scale = 1.5
ystart = 400
ystop = 525
box_list.extend(find_cars(image, colour_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

scale = 2.0
ystart = 400
ystop = 600
box_list.extend(find_cars(image, colour_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))


# Add heat to each box in box list
heat = add_heat(heat,box_list)
    
# Apply threshold to help remove false positives
heat = apply_threshold(heat,2)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

fig = plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
```


![png](output_26_0.png)


# Rubric 6: Video Implementation
### Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

 detectVehicles() implements complete pipeline from extracting features from images, detecting vehicles at multiple scales, generating heat map and applying threshold to the heat map, and finally, to drawing boxes around detected vehicles.


```python
def detectVehicles(image):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    box_list = []

    ystart = 400
    ystop = 696
    colour_space='YUV'
    
    scale = 1.0
    ystart = 400
    ystop = 500
    box_list.extend(find_cars(image, colour_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

    scale = 1.5
    ystart = 400
    ystop = 525
    box_list.extend(find_cars(image, colour_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

    scale = 2.0
    ystart = 400
    ystop = 600
    box_list.extend(find_cars(image, colour_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))


    scale = 2.5
    ystart = 400
    ystop = 650    
    box_list.extend(find_cars(image, colour_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    
    scale = 3.0
    ystart = 400
    ystop = 700    
    box_list.extend(find_cars(image, colour_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,3)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    return draw_img
```

Apply the pipeline to project video


```python
from moviepy.editor import VideoFileClip
isTesting=False

inputVideoName='project_video.mp4'
processedVideoFileName = 'vehiclesDetectedVideo.mp4'
if isTesting:
    inputVideoName='test_video.mp4'
    processedVideoFileName = 'test_vehiclesDetectedVideo.mp4'

#clip = VideoFileClip("project_video.mp4")
clip = VideoFileClip(inputVideoName)
processedVideo = clip.fl_image(detectVehicles) 
%time processedVideo.write_videofile(processedVideoFileName, audio=False)
```

    [MoviePy] >>>> Building video vehiclesDetectedVideo.mp4
    [MoviePy] Writing video vehiclesDetectedVideo.mp4


    
      0%|          | 0/1261 [00:00<?, ?it/s][A
      0%|          | 1/1261 [00:01<22:01,  1.05s/it][A
      0%|          | 2/1261 [00:02<21:47,  1.04s/it][A
      0%|          | 3/1261 [00:03<21:14,  1.01s/it][A
      0%|          | 4/1261 [00:03<20:54,  1.00it/s][A
      0%|          | 5/1261 [00:04<20:43,  1.01it/s][A
      0%|          | 6/1261 [00:05<20:46,  1.01it/s][A
      1%|          | 7/1261 [00:06<20:40,  1.01it/s][A
      1%|          | 8/1261 [00:07<20:36,  1.01it/s][A
      1%|          | 9/1261 [00:08<20:30,  1.02it/s][A
      1%|          | 10/1261 [00:10<21:23,  1.03s/it][A
      1%|          | 11/1261 [00:10<21:07,  1.01s/it][A
      1%|          | 12/1261 [00:11<21:00,  1.01s/it][A
      1%|          | 13/1261 [00:13<21:45,  1.05s/it][A
      1%|          | 14/1261 [00:14<21:45,  1.05s/it][A
      1%|          | 15/1261 [00:15<21:35,  1.04s/it][A
      1%|▏         | 16/1261 [00:16<21:29,  1.04s/it][A
      1%|▏         | 17/1261 [00:17<21:01,  1.01s/it][A
      1%|▏         | 18/1261 [00:18<20:42,  1.00it/s][A
      2%|▏         | 19/1261 [00:19<20:34,  1.01it/s][A
      2%|▏         | 20/1261 [00:20<20:44,  1.00s/it][A
      2%|▏         | 21/1261 [00:21<20:52,  1.01s/it][A
      2%|▏         | 22/1261 [00:22<20:50,  1.01s/it][A
      2%|▏         | 23/1261 [00:23<20:44,  1.01s/it][A
      2%|▏         | 24/1261 [00:24<20:45,  1.01s/it][A
      2%|▏         | 25/1261 [00:25<20:28,  1.01it/s][A
      2%|▏         | 26/1261 [00:26<20:38,  1.00s/it][A
      2%|▏         | 27/1261 [00:27<20:24,  1.01it/s][A
      2%|▏         | 28/1261 [00:28<20:44,  1.01s/it][A
      2%|▏         | 29/1261 [00:29<21:36,  1.05s/it][A
      2%|▏         | 30/1261 [00:30<21:46,  1.06s/it][A
      2%|▏         | 31/1261 [00:31<21:47,  1.06s/it][A
      3%|▎         | 32/1261 [00:32<21:38,  1.06s/it][A
      3%|▎         | 33/1261 [00:33<21:58,  1.07s/it][A
      3%|▎         | 34/1261 [00:34<22:25,  1.10s/it][A
      3%|▎         | 35/1261 [00:35<22:05,  1.08s/it][A
      3%|▎         | 36/1261 [00:36<22:02,  1.08s/it][A
      3%|▎         | 37/1261 [00:38<22:29,  1.10s/it][A
      3%|▎         | 38/1261 [00:39<22:23,  1.10s/it][A
      3%|▎         | 39/1261 [00:40<21:40,  1.06s/it][A
      3%|▎         | 40/1261 [00:41<21:33,  1.06s/it][A
      3%|▎         | 41/1261 [00:42<21:15,  1.05s/it][A
      3%|▎         | 42/1261 [00:43<21:24,  1.05s/it][A
      3%|▎         | 43/1261 [00:44<21:54,  1.08s/it][A
      3%|▎         | 44/1261 [00:45<21:44,  1.07s/it][A
      4%|▎         | 45/1261 [00:46<22:31,  1.11s/it][A
      4%|▎         | 46/1261 [00:47<21:56,  1.08s/it][A
      4%|▎         | 47/1261 [00:48<22:04,  1.09s/it][A
      4%|▍         | 48/1261 [00:49<21:39,  1.07s/it][A
      4%|▍         | 49/1261 [00:50<21:46,  1.08s/it][A
      4%|▍         | 50/1261 [00:52<22:02,  1.09s/it][A
      4%|▍         | 51/1261 [00:53<21:33,  1.07s/it][A
      4%|▍         | 52/1261 [00:54<21:30,  1.07s/it][A
      4%|▍         | 53/1261 [00:55<21:23,  1.06s/it][A
      4%|▍         | 54/1261 [00:56<21:20,  1.06s/it][A
      4%|▍         | 55/1261 [00:57<21:34,  1.07s/it][A
      4%|▍         | 56/1261 [00:58<21:23,  1.07s/it][A
      5%|▍         | 57/1261 [00:59<21:18,  1.06s/it][A
      5%|▍         | 58/1261 [01:00<21:23,  1.07s/it][A
      5%|▍         | 59/1261 [01:01<21:32,  1.08s/it][A
      5%|▍         | 60/1261 [01:02<21:11,  1.06s/it][A
      5%|▍         | 61/1261 [01:03<20:53,  1.04s/it][A
      5%|▍         | 62/1261 [01:04<21:08,  1.06s/it][A
      5%|▍         | 63/1261 [01:05<20:39,  1.03s/it][A
      5%|▌         | 64/1261 [01:06<20:18,  1.02s/it][A
      5%|▌         | 65/1261 [01:07<20:29,  1.03s/it][A
      5%|▌         | 66/1261 [01:08<20:31,  1.03s/it][A
      5%|▌         | 67/1261 [01:09<20:28,  1.03s/it][A
      5%|▌         | 68/1261 [01:10<20:39,  1.04s/it][A
      5%|▌         | 69/1261 [01:11<20:51,  1.05s/it][A
      6%|▌         | 70/1261 [01:13<21:07,  1.06s/it][A
      6%|▌         | 71/1261 [01:14<21:07,  1.07s/it][A
      6%|▌         | 72/1261 [01:15<21:05,  1.06s/it][A
      6%|▌         | 73/1261 [01:16<20:46,  1.05s/it][A
      6%|▌         | 74/1261 [01:17<20:46,  1.05s/it][A
      6%|▌         | 75/1261 [01:18<20:27,  1.04s/it][A
      6%|▌         | 76/1261 [01:19<20:20,  1.03s/it][A
      6%|▌         | 77/1261 [01:20<20:22,  1.03s/it][A
      6%|▌         | 78/1261 [01:21<20:13,  1.03s/it][A
      6%|▋         | 79/1261 [01:22<20:09,  1.02s/it][A
      6%|▋         | 80/1261 [01:23<20:02,  1.02s/it][A
      6%|▋         | 81/1261 [01:24<20:08,  1.02s/it][A
      7%|▋         | 82/1261 [01:25<20:24,  1.04s/it][A
      7%|▋         | 83/1261 [01:26<20:27,  1.04s/it][A
      7%|▋         | 84/1261 [01:27<20:34,  1.05s/it][A
      7%|▋         | 85/1261 [01:28<20:41,  1.06s/it][A
      7%|▋         | 86/1261 [01:29<20:57,  1.07s/it][A
      7%|▋         | 87/1261 [01:30<20:48,  1.06s/it][A
      7%|▋         | 88/1261 [01:31<20:48,  1.06s/it][A
      7%|▋         | 89/1261 [01:32<20:26,  1.05s/it][A
      7%|▋         | 90/1261 [01:33<20:11,  1.03s/it][A
      7%|▋         | 91/1261 [01:34<20:35,  1.06s/it][A
      7%|▋         | 92/1261 [01:36<20:25,  1.05s/it][A
      7%|▋         | 93/1261 [01:37<20:24,  1.05s/it][A
      7%|▋         | 94/1261 [01:38<20:24,  1.05s/it][A
      8%|▊         | 95/1261 [01:39<20:55,  1.08s/it][A
      8%|▊         | 96/1261 [01:40<21:08,  1.09s/it][A
      8%|▊         | 97/1261 [01:41<21:08,  1.09s/it][A
      8%|▊         | 98/1261 [01:42<21:18,  1.10s/it][A
      8%|▊         | 99/1261 [01:43<21:09,  1.09s/it][A
      8%|▊         | 100/1261 [01:44<21:03,  1.09s/it][A
      8%|▊         | 101/1261 [01:45<20:52,  1.08s/it][A
      8%|▊         | 102/1261 [01:46<20:44,  1.07s/it][A
      8%|▊         | 103/1261 [01:47<20:44,  1.07s/it][A
      8%|▊         | 104/1261 [01:48<20:38,  1.07s/it][A
      8%|▊         | 105/1261 [01:50<20:30,  1.06s/it][A
      8%|▊         | 106/1261 [01:51<20:28,  1.06s/it][A
      8%|▊         | 107/1261 [01:52<20:16,  1.05s/it][A
      9%|▊         | 108/1261 [01:53<20:07,  1.05s/it][A
      9%|▊         | 109/1261 [01:54<19:57,  1.04s/it][A
      9%|▊         | 110/1261 [01:55<19:53,  1.04s/it][A
      9%|▉         | 111/1261 [01:56<19:49,  1.03s/it][A
      9%|▉         | 112/1261 [01:57<19:49,  1.04s/it][A
      9%|▉         | 113/1261 [01:58<20:12,  1.06s/it][A
      9%|▉         | 114/1261 [01:59<20:21,  1.07s/it][A
      9%|▉         | 115/1261 [02:00<20:16,  1.06s/it][A
      9%|▉         | 116/1261 [02:01<20:22,  1.07s/it][A
      9%|▉         | 117/1261 [02:02<20:47,  1.09s/it][A
      9%|▉         | 118/1261 [02:03<20:39,  1.08s/it][A
      9%|▉         | 119/1261 [02:04<20:46,  1.09s/it][A
     10%|▉         | 120/1261 [02:06<20:58,  1.10s/it][A
     10%|▉         | 121/1261 [02:07<20:53,  1.10s/it][A
     10%|▉         | 122/1261 [02:08<20:20,  1.07s/it][A
     10%|▉         | 123/1261 [02:09<20:12,  1.07s/it][A
     10%|▉         | 124/1261 [02:10<19:52,  1.05s/it][A
     10%|▉         | 125/1261 [02:11<20:08,  1.06s/it][A
     10%|▉         | 126/1261 [02:12<19:46,  1.05s/it][A
     10%|█         | 127/1261 [02:13<19:28,  1.03s/it][A
     10%|█         | 128/1261 [02:14<19:54,  1.05s/it][A
     10%|█         | 129/1261 [02:15<20:32,  1.09s/it][A
     10%|█         | 130/1261 [02:16<20:45,  1.10s/it][A
     10%|█         | 131/1261 [02:17<20:20,  1.08s/it][A
     10%|█         | 132/1261 [02:18<19:57,  1.06s/it][A
     11%|█         | 133/1261 [02:19<20:10,  1.07s/it][A
     11%|█         | 134/1261 [02:20<19:58,  1.06s/it][A
     11%|█         | 135/1261 [02:22<20:11,  1.08s/it][A
     11%|█         | 136/1261 [02:23<20:22,  1.09s/it][A
     11%|█         | 137/1261 [02:24<20:19,  1.08s/it][A
     11%|█         | 138/1261 [02:25<19:46,  1.06s/it][A
     11%|█         | 139/1261 [02:26<19:25,  1.04s/it][A
     11%|█         | 140/1261 [02:27<19:26,  1.04s/it][A
     11%|█         | 141/1261 [02:28<19:33,  1.05s/it][A
     11%|█▏        | 142/1261 [02:29<19:47,  1.06s/it][A
     11%|█▏        | 143/1261 [02:30<20:05,  1.08s/it][A
     11%|█▏        | 144/1261 [02:31<20:53,  1.12s/it][A
     11%|█▏        | 145/1261 [02:32<20:20,  1.09s/it][A
     12%|█▏        | 146/1261 [02:33<20:19,  1.09s/it][A
     12%|█▏        | 147/1261 [02:34<20:04,  1.08s/it][A
     12%|█▏        | 148/1261 [02:35<19:53,  1.07s/it][A
     12%|█▏        | 149/1261 [02:37<20:24,  1.10s/it][A
     12%|█▏        | 150/1261 [02:38<20:34,  1.11s/it][A
     12%|█▏        | 151/1261 [02:39<20:18,  1.10s/it][A
     12%|█▏        | 152/1261 [02:40<20:54,  1.13s/it][A
     12%|█▏        | 153/1261 [02:41<20:19,  1.10s/it][A
     12%|█▏        | 154/1261 [02:42<19:49,  1.07s/it][A
     12%|█▏        | 155/1261 [02:43<19:53,  1.08s/it][A
     12%|█▏        | 156/1261 [02:44<19:45,  1.07s/it][A
     12%|█▏        | 157/1261 [02:45<19:32,  1.06s/it][A
     13%|█▎        | 158/1261 [02:46<19:25,  1.06s/it][A
     13%|█▎        | 159/1261 [02:47<19:35,  1.07s/it][A
     13%|█▎        | 160/1261 [02:48<19:27,  1.06s/it][A
     13%|█▎        | 161/1261 [02:50<19:51,  1.08s/it][A
     13%|█▎        | 162/1261 [02:51<19:50,  1.08s/it][A
     13%|█▎        | 163/1261 [02:52<19:32,  1.07s/it][A
     13%|█▎        | 164/1261 [02:53<19:25,  1.06s/it][A
     13%|█▎        | 165/1261 [02:54<19:07,  1.05s/it][A
     13%|█▎        | 166/1261 [02:55<19:01,  1.04s/it][A
     13%|█▎        | 167/1261 [02:56<19:36,  1.08s/it][A
     13%|█▎        | 168/1261 [02:57<19:39,  1.08s/it][A
     13%|█▎        | 169/1261 [02:58<19:58,  1.10s/it][A
     13%|█▎        | 170/1261 [02:59<19:48,  1.09s/it][A
     14%|█▎        | 171/1261 [03:00<19:33,  1.08s/it][A
     14%|█▎        | 172/1261 [03:01<19:29,  1.07s/it][A
     14%|█▎        | 173/1261 [03:02<19:20,  1.07s/it][A
     14%|█▍        | 174/1261 [03:04<19:26,  1.07s/it][A
     14%|█▍        | 175/1261 [03:05<19:32,  1.08s/it][A
     14%|█▍        | 176/1261 [03:06<19:46,  1.09s/it][A
     14%|█▍        | 177/1261 [03:07<19:17,  1.07s/it][A
     14%|█▍        | 178/1261 [03:08<19:14,  1.07s/it][A
     14%|█▍        | 179/1261 [03:09<19:03,  1.06s/it][A
     14%|█▍        | 180/1261 [03:10<19:01,  1.06s/it][A
     14%|█▍        | 181/1261 [03:11<18:44,  1.04s/it][A
     14%|█▍        | 182/1261 [03:12<18:37,  1.04s/it][A
     15%|█▍        | 183/1261 [03:13<18:43,  1.04s/it][A
     15%|█▍        | 184/1261 [03:14<18:31,  1.03s/it][A
     15%|█▍        | 185/1261 [03:15<18:45,  1.05s/it][A
     15%|█▍        | 186/1261 [03:16<18:33,  1.04s/it][A
     15%|█▍        | 187/1261 [03:17<18:28,  1.03s/it][A
     15%|█▍        | 188/1261 [03:18<18:27,  1.03s/it][A
     15%|█▍        | 189/1261 [03:19<18:20,  1.03s/it][A
     15%|█▌        | 190/1261 [03:20<18:40,  1.05s/it][A
     15%|█▌        | 191/1261 [03:21<18:36,  1.04s/it][A
     15%|█▌        | 192/1261 [03:22<18:25,  1.03s/it][A
     15%|█▌        | 193/1261 [03:23<18:30,  1.04s/it][A
     15%|█▌        | 194/1261 [03:24<18:16,  1.03s/it][A
     15%|█▌        | 195/1261 [03:25<18:23,  1.04s/it][A
     16%|█▌        | 196/1261 [03:26<18:31,  1.04s/it][A
     16%|█▌        | 197/1261 [03:27<18:20,  1.03s/it][A
     16%|█▌        | 198/1261 [03:29<18:46,  1.06s/it][A
     16%|█▌        | 199/1261 [03:30<18:41,  1.06s/it][A
     16%|█▌        | 200/1261 [03:31<18:24,  1.04s/it][A
     16%|█▌        | 201/1261 [03:32<18:30,  1.05s/it][A
     16%|█▌        | 202/1261 [03:33<18:22,  1.04s/it][A
     16%|█▌        | 203/1261 [03:34<18:21,  1.04s/it][A
     16%|█▌        | 204/1261 [03:35<18:11,  1.03s/it][A
     16%|█▋        | 205/1261 [03:36<18:30,  1.05s/it][A
     16%|█▋        | 206/1261 [03:37<18:22,  1.04s/it][A
     16%|█▋        | 207/1261 [03:38<18:36,  1.06s/it][A
     16%|█▋        | 208/1261 [03:39<18:45,  1.07s/it][A
     17%|█▋        | 209/1261 [03:40<18:38,  1.06s/it][A
     17%|█▋        | 210/1261 [03:41<18:29,  1.06s/it][A
     17%|█▋        | 211/1261 [03:42<18:10,  1.04s/it][A
     17%|█▋        | 212/1261 [03:43<18:02,  1.03s/it][A
     17%|█▋        | 213/1261 [03:44<17:57,  1.03s/it][A
     17%|█▋        | 214/1261 [03:45<18:45,  1.07s/it][A
     17%|█▋        | 215/1261 [03:47<19:16,  1.11s/it][A
     17%|█▋        | 216/1261 [03:48<19:30,  1.12s/it][A
     17%|█▋        | 217/1261 [03:49<19:04,  1.10s/it][A
     17%|█▋        | 218/1261 [03:50<18:59,  1.09s/it][A
     17%|█▋        | 219/1261 [03:51<20:06,  1.16s/it][A
     17%|█▋        | 220/1261 [03:52<19:25,  1.12s/it][A
     18%|█▊        | 221/1261 [03:53<18:49,  1.09s/it][A
     18%|█▊        | 222/1261 [03:54<18:30,  1.07s/it][A
     18%|█▊        | 223/1261 [03:55<18:11,  1.05s/it][A
     18%|█▊        | 224/1261 [03:56<18:36,  1.08s/it][A
     18%|█▊        | 225/1261 [03:57<18:35,  1.08s/it][A
     18%|█▊        | 226/1261 [03:59<18:33,  1.08s/it][A
     18%|█▊        | 227/1261 [04:00<18:45,  1.09s/it][A
     18%|█▊        | 228/1261 [04:01<18:32,  1.08s/it][A
     18%|█▊        | 229/1261 [04:02<18:25,  1.07s/it][A
     18%|█▊        | 230/1261 [04:03<18:38,  1.08s/it][A
     18%|█▊        | 231/1261 [04:04<18:41,  1.09s/it][A
     18%|█▊        | 232/1261 [04:05<18:26,  1.08s/it][A
     18%|█▊        | 233/1261 [04:06<18:08,  1.06s/it][A
     19%|█▊        | 234/1261 [04:07<18:07,  1.06s/it][A
     19%|█▊        | 235/1261 [04:08<18:05,  1.06s/it][A
     19%|█▊        | 236/1261 [04:09<17:55,  1.05s/it][A
     19%|█▉        | 237/1261 [04:10<17:40,  1.04s/it][A
     19%|█▉        | 238/1261 [04:11<17:43,  1.04s/it][A
     19%|█▉        | 239/1261 [04:12<17:39,  1.04s/it][A
     19%|█▉        | 240/1261 [04:13<17:23,  1.02s/it][A
     19%|█▉        | 241/1261 [04:14<17:20,  1.02s/it][A
     19%|█▉        | 242/1261 [04:15<17:23,  1.02s/it][A
     19%|█▉        | 243/1261 [04:16<17:19,  1.02s/it][A
     19%|█▉        | 244/1261 [04:17<17:14,  1.02s/it][A
     19%|█▉        | 245/1261 [04:18<17:16,  1.02s/it][A
     20%|█▉        | 246/1261 [04:19<17:19,  1.02s/it][A
     20%|█▉        | 247/1261 [04:20<17:10,  1.02s/it][A
     20%|█▉        | 248/1261 [04:21<17:03,  1.01s/it][A
     20%|█▉        | 249/1261 [04:22<17:10,  1.02s/it][A
     20%|█▉        | 250/1261 [04:23<17:05,  1.01s/it][A
     20%|█▉        | 251/1261 [04:24<17:17,  1.03s/it][A
     20%|█▉        | 252/1261 [04:25<17:16,  1.03s/it][A
     20%|██        | 253/1261 [04:27<17:37,  1.05s/it][A
     20%|██        | 254/1261 [04:28<17:53,  1.07s/it][A
     20%|██        | 255/1261 [04:29<17:38,  1.05s/it][A
     20%|██        | 256/1261 [04:30<17:52,  1.07s/it][A
     20%|██        | 257/1261 [04:31<17:48,  1.06s/it][A
     20%|██        | 258/1261 [04:32<17:27,  1.04s/it][A
     21%|██        | 259/1261 [04:33<17:21,  1.04s/it][A
     21%|██        | 260/1261 [04:34<17:22,  1.04s/it][A
     21%|██        | 261/1261 [04:35<17:14,  1.03s/it][A
     21%|██        | 262/1261 [04:36<17:13,  1.03s/it][A
     21%|██        | 263/1261 [04:37<17:13,  1.04s/it][A
     21%|██        | 264/1261 [04:38<17:16,  1.04s/it][A
     21%|██        | 265/1261 [04:39<17:02,  1.03s/it][A
     21%|██        | 266/1261 [04:40<17:08,  1.03s/it][A
     21%|██        | 267/1261 [04:41<17:08,  1.03s/it][A
     21%|██▏       | 268/1261 [04:42<17:10,  1.04s/it][A
     21%|██▏       | 269/1261 [04:43<17:24,  1.05s/it][A
     21%|██▏       | 270/1261 [04:44<17:34,  1.06s/it][A
     21%|██▏       | 271/1261 [04:45<17:20,  1.05s/it][A
     22%|██▏       | 272/1261 [04:46<16:56,  1.03s/it][A
     22%|██▏       | 273/1261 [04:47<17:03,  1.04s/it][A
     22%|██▏       | 274/1261 [04:49<17:15,  1.05s/it][A
     22%|██▏       | 275/1261 [04:50<17:00,  1.04s/it][A
     22%|██▏       | 276/1261 [04:51<17:15,  1.05s/it][A
     22%|██▏       | 277/1261 [04:52<17:28,  1.07s/it][A
     22%|██▏       | 278/1261 [04:53<17:32,  1.07s/it][A
     22%|██▏       | 279/1261 [04:54<17:35,  1.08s/it][A
     22%|██▏       | 280/1261 [04:55<17:46,  1.09s/it][A
     22%|██▏       | 281/1261 [04:56<17:32,  1.07s/it][A
     22%|██▏       | 282/1261 [04:57<17:20,  1.06s/it][A
     22%|██▏       | 283/1261 [04:58<16:58,  1.04s/it][A
     23%|██▎       | 284/1261 [04:59<16:46,  1.03s/it][A
     23%|██▎       | 285/1261 [05:00<16:41,  1.03s/it][A
     23%|██▎       | 286/1261 [05:01<16:46,  1.03s/it][A
     23%|██▎       | 287/1261 [05:02<16:39,  1.03s/it][A
     23%|██▎       | 288/1261 [05:03<17:00,  1.05s/it][A
     23%|██▎       | 289/1261 [05:04<17:24,  1.07s/it][A
     23%|██▎       | 290/1261 [05:06<17:57,  1.11s/it][A
     23%|██▎       | 291/1261 [05:07<17:29,  1.08s/it][A
     23%|██▎       | 292/1261 [05:08<17:43,  1.10s/it][A
     23%|██▎       | 293/1261 [05:09<17:29,  1.08s/it][A
     23%|██▎       | 294/1261 [05:10<17:24,  1.08s/it][A
     23%|██▎       | 295/1261 [05:11<17:01,  1.06s/it][A
     23%|██▎       | 296/1261 [05:12<16:59,  1.06s/it][A
     24%|██▎       | 297/1261 [05:13<16:45,  1.04s/it][A
     24%|██▎       | 298/1261 [05:14<16:41,  1.04s/it][A
     24%|██▎       | 299/1261 [05:15<16:21,  1.02s/it][A
     24%|██▍       | 300/1261 [05:16<16:18,  1.02s/it][A
     24%|██▍       | 301/1261 [05:17<16:29,  1.03s/it][A
     24%|██▍       | 302/1261 [05:18<16:43,  1.05s/it][A
     24%|██▍       | 303/1261 [05:19<16:33,  1.04s/it][A
     24%|██▍       | 304/1261 [05:20<16:31,  1.04s/it][A
     24%|██▍       | 305/1261 [05:21<16:17,  1.02s/it][A
     24%|██▍       | 306/1261 [05:22<16:08,  1.01s/it][A
     24%|██▍       | 307/1261 [05:23<16:05,  1.01s/it][A
     24%|██▍       | 308/1261 [05:24<16:01,  1.01s/it][A
     25%|██▍       | 309/1261 [05:25<16:11,  1.02s/it][A
     25%|██▍       | 310/1261 [05:26<16:06,  1.02s/it][A
     25%|██▍       | 311/1261 [05:27<15:51,  1.00s/it][A
     25%|██▍       | 312/1261 [05:28<15:44,  1.00it/s][A
     25%|██▍       | 313/1261 [05:29<15:36,  1.01it/s][A
     25%|██▍       | 314/1261 [05:30<15:35,  1.01it/s][A
     25%|██▍       | 315/1261 [05:31<15:32,  1.01it/s][A
     25%|██▌       | 316/1261 [05:32<15:28,  1.02it/s][A
     25%|██▌       | 317/1261 [05:33<15:41,  1.00it/s][A
     25%|██▌       | 318/1261 [05:34<15:47,  1.00s/it][A
     25%|██▌       | 319/1261 [05:35<15:42,  1.00s/it][A
     25%|██▌       | 320/1261 [05:36<15:42,  1.00s/it][A
     25%|██▌       | 321/1261 [05:37<15:46,  1.01s/it][A
     26%|██▌       | 322/1261 [05:38<15:50,  1.01s/it][A
     26%|██▌       | 323/1261 [05:39<15:51,  1.01s/it][A
     26%|██▌       | 324/1261 [05:40<15:47,  1.01s/it][A
     26%|██▌       | 325/1261 [05:41<15:37,  1.00s/it][A
     26%|██▌       | 326/1261 [05:42<15:27,  1.01it/s][A
     26%|██▌       | 327/1261 [05:43<15:35,  1.00s/it][A
     26%|██▌       | 328/1261 [05:44<15:27,  1.01it/s][A
     26%|██▌       | 329/1261 [05:45<15:23,  1.01it/s][A
     26%|██▌       | 330/1261 [05:46<15:21,  1.01it/s][A
     26%|██▌       | 331/1261 [05:47<15:22,  1.01it/s][A
     26%|██▋       | 332/1261 [05:48<15:21,  1.01it/s][A
     26%|██▋       | 333/1261 [05:49<15:14,  1.01it/s][A
     26%|██▋       | 334/1261 [05:50<15:19,  1.01it/s][A
     27%|██▋       | 335/1261 [05:51<15:19,  1.01it/s][A
     27%|██▋       | 336/1261 [05:52<15:28,  1.00s/it][A
     27%|██▋       | 337/1261 [05:53<15:24,  1.00s/it][A
     27%|██▋       | 338/1261 [05:54<15:22,  1.00it/s][A
     27%|██▋       | 339/1261 [05:55<15:15,  1.01it/s][A
     27%|██▋       | 340/1261 [05:56<15:14,  1.01it/s][A
     27%|██▋       | 341/1261 [05:57<15:15,  1.01it/s][A
     27%|██▋       | 342/1261 [05:58<15:18,  1.00it/s][A
     27%|██▋       | 343/1261 [05:59<15:13,  1.00it/s][A
     27%|██▋       | 344/1261 [06:00<15:28,  1.01s/it][A
     27%|██▋       | 345/1261 [06:01<15:24,  1.01s/it][A
     27%|██▋       | 346/1261 [06:02<15:14,  1.00it/s][A
     28%|██▊       | 347/1261 [06:03<15:15,  1.00s/it][A
     28%|██▊       | 348/1261 [06:04<15:15,  1.00s/it][A
     28%|██▊       | 349/1261 [06:05<15:08,  1.00it/s][A
     28%|██▊       | 350/1261 [06:06<15:02,  1.01it/s][A
     28%|██▊       | 351/1261 [06:07<14:54,  1.02it/s][A
     28%|██▊       | 352/1261 [06:08<14:53,  1.02it/s][A
     28%|██▊       | 353/1261 [06:09<14:49,  1.02it/s][A
     28%|██▊       | 354/1261 [06:10<14:48,  1.02it/s][A
     28%|██▊       | 355/1261 [06:11<14:45,  1.02it/s][A
     28%|██▊       | 356/1261 [06:12<15:01,  1.00it/s][A
     28%|██▊       | 357/1261 [06:13<15:06,  1.00s/it][A
     28%|██▊       | 358/1261 [06:14<15:21,  1.02s/it][A
     28%|██▊       | 359/1261 [06:15<15:21,  1.02s/it][A
     29%|██▊       | 360/1261 [06:16<15:12,  1.01s/it][A
     29%|██▊       | 361/1261 [06:17<15:10,  1.01s/it][A
     29%|██▊       | 362/1261 [06:18<15:20,  1.02s/it][A
     29%|██▉       | 363/1261 [06:19<15:33,  1.04s/it][A
     29%|██▉       | 364/1261 [06:20<15:39,  1.05s/it][A
     29%|██▉       | 365/1261 [06:21<15:34,  1.04s/it][A
     29%|██▉       | 366/1261 [06:22<15:18,  1.03s/it][A
     29%|██▉       | 367/1261 [06:23<15:06,  1.01s/it][A
     29%|██▉       | 368/1261 [06:24<15:12,  1.02s/it][A
     29%|██▉       | 369/1261 [06:25<14:58,  1.01s/it][A
     29%|██▉       | 370/1261 [06:26<15:04,  1.01s/it][A
     29%|██▉       | 371/1261 [06:27<14:59,  1.01s/it][A
     30%|██▉       | 372/1261 [06:28<14:52,  1.00s/it][A
     30%|██▉       | 373/1261 [06:29<14:50,  1.00s/it][A
     30%|██▉       | 374/1261 [06:30<14:45,  1.00it/s][A
     30%|██▉       | 375/1261 [06:31<14:44,  1.00it/s][A
     30%|██▉       | 376/1261 [06:32<14:46,  1.00s/it][A
     30%|██▉       | 377/1261 [06:33<14:39,  1.00it/s][A
     30%|██▉       | 378/1261 [06:34<14:41,  1.00it/s][A
     30%|███       | 379/1261 [06:35<14:54,  1.01s/it][A
     30%|███       | 380/1261 [06:36<14:53,  1.01s/it][A
     30%|███       | 381/1261 [06:37<15:06,  1.03s/it][A
     30%|███       | 382/1261 [06:38<15:02,  1.03s/it][A
     30%|███       | 383/1261 [06:39<14:46,  1.01s/it][A
     30%|███       | 384/1261 [06:40<14:39,  1.00s/it][A
     31%|███       | 385/1261 [06:41<14:34,  1.00it/s][A
     31%|███       | 386/1261 [06:42<14:33,  1.00it/s][A
     31%|███       | 387/1261 [06:43<14:35,  1.00s/it][A
     31%|███       | 388/1261 [06:44<14:37,  1.01s/it][A
     31%|███       | 389/1261 [06:45<14:45,  1.02s/it][A
     31%|███       | 390/1261 [06:46<14:36,  1.01s/it][A
     31%|███       | 391/1261 [06:47<14:51,  1.02s/it][A
     31%|███       | 392/1261 [06:49<14:54,  1.03s/it][A
     31%|███       | 393/1261 [06:50<14:54,  1.03s/it][A
     31%|███       | 394/1261 [06:51<14:48,  1.02s/it][A
     31%|███▏      | 395/1261 [06:52<14:46,  1.02s/it][A
     31%|███▏      | 396/1261 [06:53<14:49,  1.03s/it][A
     31%|███▏      | 397/1261 [06:54<14:38,  1.02s/it][A
     32%|███▏      | 398/1261 [06:55<14:33,  1.01s/it][A
     32%|███▏      | 399/1261 [06:56<14:24,  1.00s/it][A
     32%|███▏      | 400/1261 [06:57<14:17,  1.00it/s][A
     32%|███▏      | 401/1261 [06:58<14:16,  1.00it/s][A
     32%|███▏      | 402/1261 [06:59<14:22,  1.00s/it][A
     32%|███▏      | 403/1261 [07:00<14:17,  1.00it/s][A
     32%|███▏      | 404/1261 [07:01<14:13,  1.00it/s][A
     32%|███▏      | 405/1261 [07:02<14:15,  1.00it/s][A
     32%|███▏      | 406/1261 [07:03<14:09,  1.01it/s][A
     32%|███▏      | 407/1261 [07:04<14:08,  1.01it/s][A
     32%|███▏      | 408/1261 [07:05<14:01,  1.01it/s][A
     32%|███▏      | 409/1261 [07:06<14:06,  1.01it/s][A
     33%|███▎      | 410/1261 [07:07<14:00,  1.01it/s][A
     33%|███▎      | 411/1261 [07:08<14:04,  1.01it/s][A
     33%|███▎      | 412/1261 [07:09<14:08,  1.00it/s][A
     33%|███▎      | 413/1261 [07:10<14:13,  1.01s/it][A
     33%|███▎      | 414/1261 [07:11<14:07,  1.00s/it][A
     33%|███▎      | 415/1261 [07:12<14:10,  1.00s/it][A
     33%|███▎      | 416/1261 [07:13<14:01,  1.00it/s][A
     33%|███▎      | 417/1261 [07:13<13:55,  1.01it/s][A
     33%|███▎      | 418/1261 [07:15<14:12,  1.01s/it][A
     33%|███▎      | 419/1261 [07:16<14:19,  1.02s/it][A
     33%|███▎      | 420/1261 [07:17<14:24,  1.03s/it][A
     33%|███▎      | 421/1261 [07:18<14:19,  1.02s/it][A
     33%|███▎      | 422/1261 [07:19<14:18,  1.02s/it][A
     34%|███▎      | 423/1261 [07:20<14:31,  1.04s/it][A
     34%|███▎      | 424/1261 [07:21<14:17,  1.02s/it][A
     34%|███▎      | 425/1261 [07:22<14:20,  1.03s/it][A
     34%|███▍      | 426/1261 [07:23<14:15,  1.02s/it][A
     34%|███▍      | 427/1261 [07:24<14:07,  1.02s/it][A
     34%|███▍      | 428/1261 [07:25<14:06,  1.02s/it][A
     34%|███▍      | 429/1261 [07:26<14:07,  1.02s/it][A
     34%|███▍      | 430/1261 [07:27<14:00,  1.01s/it][A
     34%|███▍      | 431/1261 [07:28<13:50,  1.00s/it][A
     34%|███▍      | 432/1261 [07:29<13:43,  1.01it/s][A
     34%|███▍      | 433/1261 [07:30<13:37,  1.01it/s][A
     34%|███▍      | 434/1261 [07:31<13:35,  1.01it/s][A
     34%|███▍      | 435/1261 [07:32<13:32,  1.02it/s][A
     35%|███▍      | 436/1261 [07:33<13:34,  1.01it/s][A
     35%|███▍      | 437/1261 [07:34<13:27,  1.02it/s][A
     35%|███▍      | 438/1261 [07:35<13:26,  1.02it/s][A
     35%|███▍      | 439/1261 [07:36<13:34,  1.01it/s][A
     35%|███▍      | 440/1261 [07:37<13:40,  1.00it/s][A
     35%|███▍      | 441/1261 [07:38<13:42,  1.00s/it][A
     35%|███▌      | 442/1261 [07:39<13:43,  1.01s/it][A
     35%|███▌      | 443/1261 [07:40<13:42,  1.01s/it][A
     35%|███▌      | 444/1261 [07:41<13:41,  1.01s/it][A
     35%|███▌      | 445/1261 [07:42<13:43,  1.01s/it][A
     35%|███▌      | 446/1261 [07:43<13:36,  1.00s/it][A
     35%|███▌      | 447/1261 [07:44<13:45,  1.01s/it][A
     36%|███▌      | 448/1261 [07:45<13:38,  1.01s/it][A
     36%|███▌      | 449/1261 [07:46<13:37,  1.01s/it][A
     36%|███▌      | 450/1261 [07:47<13:27,  1.00it/s][A
     36%|███▌      | 451/1261 [07:48<13:35,  1.01s/it][A
     36%|███▌      | 452/1261 [07:49<13:32,  1.00s/it][A
     36%|███▌      | 453/1261 [07:50<13:27,  1.00it/s][A
     36%|███▌      | 454/1261 [07:51<13:32,  1.01s/it][A
     36%|███▌      | 455/1261 [07:52<13:46,  1.03s/it][A
     36%|███▌      | 456/1261 [07:53<13:44,  1.02s/it][A
     36%|███▌      | 457/1261 [07:54<13:44,  1.02s/it][A
     36%|███▋      | 458/1261 [07:55<13:39,  1.02s/it][A
     36%|███▋      | 459/1261 [07:56<13:37,  1.02s/it][A
     36%|███▋      | 460/1261 [07:57<13:39,  1.02s/it][A
     37%|███▋      | 461/1261 [07:58<13:41,  1.03s/it][A
     37%|███▋      | 462/1261 [07:59<13:46,  1.03s/it][A
     37%|███▋      | 463/1261 [08:00<13:46,  1.04s/it][A
     37%|███▋      | 464/1261 [08:01<13:38,  1.03s/it][A
     37%|███▋      | 465/1261 [08:02<13:28,  1.02s/it][A
     37%|███▋      | 466/1261 [08:03<13:25,  1.01s/it][A
     37%|███▋      | 467/1261 [08:04<13:18,  1.01s/it][A
     37%|███▋      | 468/1261 [08:05<13:13,  1.00s/it][A
     37%|███▋      | 469/1261 [08:06<13:12,  1.00s/it][A
     37%|███▋      | 470/1261 [08:07<13:20,  1.01s/it][A
     37%|███▋      | 471/1261 [08:08<13:11,  1.00s/it][A
     37%|███▋      | 472/1261 [08:09<13:07,  1.00it/s][A
     38%|███▊      | 473/1261 [08:10<13:07,  1.00it/s][A
     38%|███▊      | 474/1261 [08:11<13:10,  1.00s/it][A
     38%|███▊      | 475/1261 [08:12<13:11,  1.01s/it][A
     38%|███▊      | 476/1261 [08:13<13:10,  1.01s/it][A
     38%|███▊      | 477/1261 [08:14<13:03,  1.00it/s][A
     38%|███▊      | 478/1261 [08:15<12:59,  1.00it/s][A
     38%|███▊      | 479/1261 [08:16<13:04,  1.00s/it][A
     38%|███▊      | 480/1261 [08:17<13:03,  1.00s/it][A
     38%|███▊      | 481/1261 [08:18<13:04,  1.01s/it][A
     38%|███▊      | 482/1261 [08:19<13:05,  1.01s/it][A
     38%|███▊      | 483/1261 [08:20<13:07,  1.01s/it][A
     38%|███▊      | 484/1261 [08:21<12:58,  1.00s/it][A
     38%|███▊      | 485/1261 [08:22<12:49,  1.01it/s][A
     39%|███▊      | 486/1261 [08:23<12:53,  1.00it/s][A
     39%|███▊      | 487/1261 [08:24<12:49,  1.01it/s][A
     39%|███▊      | 488/1261 [08:25<12:41,  1.01it/s][A
     39%|███▉      | 489/1261 [08:26<12:38,  1.02it/s][A
     39%|███▉      | 490/1261 [08:27<12:34,  1.02it/s][A
     39%|███▉      | 491/1261 [08:28<12:35,  1.02it/s][A
     39%|███▉      | 492/1261 [08:29<12:36,  1.02it/s][A
     39%|███▉      | 493/1261 [08:30<12:37,  1.01it/s][A
     39%|███▉      | 494/1261 [08:31<12:42,  1.01it/s][A
     39%|███▉      | 495/1261 [08:32<12:35,  1.01it/s][A
     39%|███▉      | 496/1261 [08:33<12:37,  1.01it/s][A
     39%|███▉      | 497/1261 [08:34<12:39,  1.01it/s][A
     39%|███▉      | 498/1261 [08:35<12:39,  1.00it/s][A
     40%|███▉      | 499/1261 [08:36<12:48,  1.01s/it][A
     40%|███▉      | 500/1261 [08:37<12:48,  1.01s/it][A
     40%|███▉      | 501/1261 [08:38<12:52,  1.02s/it][A
     40%|███▉      | 502/1261 [08:39<12:52,  1.02s/it][A
     40%|███▉      | 503/1261 [08:40<12:48,  1.01s/it][A
     40%|███▉      | 504/1261 [08:41<12:36,  1.00it/s][A
     40%|████      | 505/1261 [08:42<12:33,  1.00it/s][A
     40%|████      | 506/1261 [08:43<12:38,  1.00s/it][A
     40%|████      | 507/1261 [08:44<12:41,  1.01s/it][A
     40%|████      | 508/1261 [08:45<12:58,  1.03s/it][A
     40%|████      | 509/1261 [08:46<13:15,  1.06s/it][A
     40%|████      | 510/1261 [08:47<13:14,  1.06s/it][A
     41%|████      | 511/1261 [08:48<13:15,  1.06s/it][A
     41%|████      | 512/1261 [08:49<13:20,  1.07s/it][A
     41%|████      | 513/1261 [08:51<13:45,  1.10s/it][A
     41%|████      | 514/1261 [08:52<13:50,  1.11s/it][A
     41%|████      | 515/1261 [08:53<13:29,  1.08s/it][A
     41%|████      | 516/1261 [08:54<13:20,  1.07s/it][A
     41%|████      | 517/1261 [08:55<13:03,  1.05s/it][A
     41%|████      | 518/1261 [08:56<12:52,  1.04s/it][A
     41%|████      | 519/1261 [08:57<12:45,  1.03s/it][A
     41%|████      | 520/1261 [08:58<12:46,  1.03s/it][A
     41%|████▏     | 521/1261 [08:59<12:49,  1.04s/it][A
     41%|████▏     | 522/1261 [09:00<12:44,  1.03s/it][A
     41%|████▏     | 523/1261 [09:01<12:32,  1.02s/it][A
     42%|████▏     | 524/1261 [09:02<12:23,  1.01s/it][A
     42%|████▏     | 525/1261 [09:03<12:21,  1.01s/it][A
     42%|████▏     | 526/1261 [09:04<12:18,  1.00s/it][A
     42%|████▏     | 527/1261 [09:05<12:16,  1.00s/it][A
     42%|████▏     | 528/1261 [09:06<12:11,  1.00it/s][A
     42%|████▏     | 529/1261 [09:07<12:11,  1.00it/s][A
     42%|████▏     | 530/1261 [09:08<12:23,  1.02s/it][A
     42%|████▏     | 531/1261 [09:09<12:21,  1.02s/it][A
     42%|████▏     | 532/1261 [09:10<12:16,  1.01s/it][A
     42%|████▏     | 533/1261 [09:11<12:25,  1.02s/it][A
     42%|████▏     | 534/1261 [09:12<12:17,  1.01s/it][A
     42%|████▏     | 535/1261 [09:13<12:14,  1.01s/it][A
     43%|████▎     | 536/1261 [09:14<12:11,  1.01s/it][A
     43%|████▎     | 537/1261 [09:15<12:05,  1.00s/it][A
     43%|████▎     | 538/1261 [09:16<12:02,  1.00it/s][A
     43%|████▎     | 539/1261 [09:17<12:05,  1.00s/it][A
     43%|████▎     | 540/1261 [09:18<12:01,  1.00s/it][A
     43%|████▎     | 541/1261 [09:19<11:58,  1.00it/s][A
     43%|████▎     | 542/1261 [09:20<12:02,  1.01s/it][A
     43%|████▎     | 543/1261 [09:21<12:05,  1.01s/it][A
     43%|████▎     | 544/1261 [09:22<12:10,  1.02s/it][A
     43%|████▎     | 545/1261 [09:23<12:07,  1.02s/it][A
     43%|████▎     | 546/1261 [09:24<12:04,  1.01s/it][A
     43%|████▎     | 547/1261 [09:25<12:04,  1.01s/it][A
     43%|████▎     | 548/1261 [09:26<12:14,  1.03s/it][A
     44%|████▎     | 549/1261 [09:27<12:11,  1.03s/it][A
     44%|████▎     | 550/1261 [09:28<12:04,  1.02s/it][A
     44%|████▎     | 551/1261 [09:29<12:06,  1.02s/it][A
     44%|████▍     | 552/1261 [09:30<12:08,  1.03s/it][A
     44%|████▍     | 553/1261 [09:31<12:01,  1.02s/it][A
     44%|████▍     | 554/1261 [09:32<12:04,  1.02s/it][A
     44%|████▍     | 555/1261 [09:33<12:02,  1.02s/it][A
     44%|████▍     | 556/1261 [09:34<11:56,  1.02s/it][A
     44%|████▍     | 557/1261 [09:35<11:47,  1.00s/it][A
     44%|████▍     | 558/1261 [09:36<11:42,  1.00it/s][A
     44%|████▍     | 559/1261 [09:37<11:50,  1.01s/it][A
     44%|████▍     | 560/1261 [09:38<11:42,  1.00s/it][A
     44%|████▍     | 561/1261 [09:39<11:35,  1.01it/s][A
     45%|████▍     | 562/1261 [09:40<11:42,  1.01s/it][A
     45%|████▍     | 563/1261 [09:41<11:42,  1.01s/it][A
     45%|████▍     | 564/1261 [09:42<11:37,  1.00s/it][A
     45%|████▍     | 565/1261 [09:43<11:33,  1.00it/s][A
     45%|████▍     | 566/1261 [09:44<11:35,  1.00s/it][A
     45%|████▍     | 567/1261 [09:45<11:40,  1.01s/it][A
     45%|████▌     | 568/1261 [09:46<11:30,  1.00it/s][A
     45%|████▌     | 569/1261 [09:47<11:29,  1.00it/s][A
     45%|████▌     | 570/1261 [09:48<11:34,  1.01s/it][A
     45%|████▌     | 571/1261 [09:49<11:24,  1.01it/s][A
     45%|████▌     | 572/1261 [09:50<11:27,  1.00it/s][A
     45%|████▌     | 573/1261 [09:51<11:31,  1.01s/it][A
     46%|████▌     | 574/1261 [09:52<11:31,  1.01s/it][A
     46%|████▌     | 575/1261 [09:53<11:27,  1.00s/it][A
     46%|████▌     | 576/1261 [09:54<11:24,  1.00it/s][A
     46%|████▌     | 577/1261 [09:55<11:23,  1.00it/s][A
     46%|████▌     | 578/1261 [09:56<11:29,  1.01s/it][A
     46%|████▌     | 579/1261 [09:57<11:32,  1.01s/it][A
     46%|████▌     | 580/1261 [09:58<11:32,  1.02s/it][A
     46%|████▌     | 581/1261 [10:00<11:38,  1.03s/it][A
     46%|████▌     | 582/1261 [10:01<11:46,  1.04s/it][A
     46%|████▌     | 583/1261 [10:02<11:44,  1.04s/it][A
     46%|████▋     | 584/1261 [10:03<11:34,  1.03s/it][A
     46%|████▋     | 585/1261 [10:04<11:19,  1.01s/it][A
     46%|████▋     | 586/1261 [10:05<11:16,  1.00s/it][A
     47%|████▋     | 587/1261 [10:06<11:11,  1.00it/s][A
     47%|████▋     | 588/1261 [10:07<11:05,  1.01it/s][A
     47%|████▋     | 589/1261 [10:08<11:12,  1.00s/it][A
     47%|████▋     | 590/1261 [10:09<11:12,  1.00s/it][A
     47%|████▋     | 591/1261 [10:10<11:17,  1.01s/it][A
     47%|████▋     | 592/1261 [10:11<11:14,  1.01s/it][A
     47%|████▋     | 593/1261 [10:12<11:08,  1.00s/it][A
     47%|████▋     | 594/1261 [10:13<11:07,  1.00s/it][A
     47%|████▋     | 595/1261 [10:14<11:06,  1.00s/it][A
     47%|████▋     | 596/1261 [10:15<11:03,  1.00it/s][A
     47%|████▋     | 597/1261 [10:16<11:06,  1.00s/it][A
     47%|████▋     | 598/1261 [10:17<11:05,  1.00s/it][A
     48%|████▊     | 599/1261 [10:18<11:04,  1.00s/it][A
     48%|████▊     | 600/1261 [10:19<11:04,  1.01s/it][A
     48%|████▊     | 601/1261 [10:20<11:04,  1.01s/it][A
     48%|████▊     | 602/1261 [10:21<11:01,  1.00s/it][A
     48%|████▊     | 603/1261 [10:22<11:01,  1.00s/it][A
     48%|████▊     | 604/1261 [10:23<10:53,  1.00it/s][A
     48%|████▊     | 605/1261 [10:24<10:55,  1.00it/s][A
     48%|████▊     | 606/1261 [10:25<11:05,  1.02s/it][A
     48%|████▊     | 607/1261 [10:26<10:54,  1.00s/it][A
     48%|████▊     | 608/1261 [10:27<10:54,  1.00s/it][A
     48%|████▊     | 609/1261 [10:28<10:45,  1.01it/s][A
     48%|████▊     | 610/1261 [10:29<10:44,  1.01it/s][A
     48%|████▊     | 611/1261 [10:30<10:47,  1.00it/s][A
     49%|████▊     | 612/1261 [10:31<10:48,  1.00it/s][A
     49%|████▊     | 613/1261 [10:32<10:45,  1.00it/s][A
     49%|████▊     | 614/1261 [10:33<10:51,  1.01s/it][A
     49%|████▉     | 615/1261 [10:34<10:51,  1.01s/it][A
     49%|████▉     | 616/1261 [10:35<10:48,  1.01s/it][A
     49%|████▉     | 617/1261 [10:36<10:47,  1.01s/it][A
     49%|████▉     | 618/1261 [10:37<10:45,  1.00s/it][A
     49%|████▉     | 619/1261 [10:38<10:47,  1.01s/it][A
     49%|████▉     | 620/1261 [10:39<10:43,  1.00s/it][A
     49%|████▉     | 621/1261 [10:40<10:40,  1.00s/it][A
     49%|████▉     | 622/1261 [10:41<10:38,  1.00it/s][A
     49%|████▉     | 623/1261 [10:42<10:34,  1.01it/s][A
     49%|████▉     | 624/1261 [10:43<10:33,  1.01it/s][A
     50%|████▉     | 625/1261 [10:44<10:27,  1.01it/s][A
     50%|████▉     | 626/1261 [10:45<10:34,  1.00it/s][A
     50%|████▉     | 627/1261 [10:46<10:36,  1.00s/it][A
     50%|████▉     | 628/1261 [10:47<10:31,  1.00it/s][A
     50%|████▉     | 629/1261 [10:48<10:27,  1.01it/s][A
     50%|████▉     | 630/1261 [10:49<10:21,  1.01it/s][A
     50%|█████     | 631/1261 [10:50<10:24,  1.01it/s][A
     50%|█████     | 632/1261 [10:51<10:21,  1.01it/s][A
     50%|█████     | 633/1261 [10:52<10:22,  1.01it/s][A
     50%|█████     | 634/1261 [10:53<10:23,  1.01it/s][A
     50%|█████     | 635/1261 [10:54<10:16,  1.02it/s][A
     50%|█████     | 636/1261 [10:54<10:14,  1.02it/s][A
     51%|█████     | 637/1261 [10:56<10:17,  1.01it/s][A
     51%|█████     | 638/1261 [10:57<10:20,  1.00it/s][A
     51%|█████     | 639/1261 [10:58<10:20,  1.00it/s][A
     51%|█████     | 640/1261 [10:59<10:23,  1.00s/it][A
     51%|█████     | 641/1261 [11:00<10:21,  1.00s/it][A
     51%|█████     | 642/1261 [11:01<10:15,  1.01it/s][A
     51%|█████     | 643/1261 [11:01<10:12,  1.01it/s][A
     51%|█████     | 644/1261 [11:02<10:09,  1.01it/s][A
     51%|█████     | 645/1261 [11:03<10:09,  1.01it/s][A
     51%|█████     | 646/1261 [11:04<10:09,  1.01it/s][A
     51%|█████▏    | 647/1261 [11:05<10:12,  1.00it/s][A
     51%|█████▏    | 648/1261 [11:06<10:08,  1.01it/s][A
     51%|█████▏    | 649/1261 [11:07<10:06,  1.01it/s][A
     52%|█████▏    | 650/1261 [11:08<10:04,  1.01it/s][A
     52%|█████▏    | 651/1261 [11:09<10:05,  1.01it/s][A
     52%|█████▏    | 652/1261 [11:10<10:09,  1.00s/it][A
     52%|█████▏    | 653/1261 [11:11<10:05,  1.00it/s][A
     52%|█████▏    | 654/1261 [11:12<10:10,  1.01s/it][A
     52%|█████▏    | 655/1261 [11:13<10:04,  1.00it/s][A
     52%|█████▏    | 656/1261 [11:14<09:59,  1.01it/s][A
     52%|█████▏    | 657/1261 [11:15<09:55,  1.01it/s][A
     52%|█████▏    | 658/1261 [11:16<09:51,  1.02it/s][A
     52%|█████▏    | 659/1261 [11:17<09:52,  1.02it/s][A
     52%|█████▏    | 660/1261 [11:18<09:57,  1.01it/s][A
     52%|█████▏    | 661/1261 [11:19<10:11,  1.02s/it][A
     52%|█████▏    | 662/1261 [11:20<10:07,  1.01s/it][A
     53%|█████▎    | 663/1261 [11:21<10:08,  1.02s/it][A
     53%|█████▎    | 664/1261 [11:23<10:11,  1.02s/it][A
     53%|█████▎    | 665/1261 [11:23<10:02,  1.01s/it][A
     53%|█████▎    | 666/1261 [11:24<09:59,  1.01s/it][A
     53%|█████▎    | 667/1261 [11:25<09:58,  1.01s/it][A
     53%|█████▎    | 668/1261 [11:26<09:54,  1.00s/it][A
     53%|█████▎    | 669/1261 [11:27<09:52,  1.00s/it][A
     53%|█████▎    | 670/1261 [11:28<09:46,  1.01it/s][A
     53%|█████▎    | 671/1261 [11:29<09:43,  1.01it/s][A
     53%|█████▎    | 672/1261 [11:30<09:41,  1.01it/s][A
     53%|█████▎    | 673/1261 [11:31<09:49,  1.00s/it][A
     53%|█████▎    | 674/1261 [11:33<09:55,  1.01s/it][A
     54%|█████▎    | 675/1261 [11:34<09:51,  1.01s/it][A
     54%|█████▎    | 676/1261 [11:34<09:47,  1.00s/it][A
     54%|█████▎    | 677/1261 [11:36<09:51,  1.01s/it][A
     54%|█████▍    | 678/1261 [11:37<09:53,  1.02s/it][A
     54%|█████▍    | 679/1261 [11:38<09:50,  1.01s/it][A
     54%|█████▍    | 680/1261 [11:39<09:54,  1.02s/it][A
     54%|█████▍    | 681/1261 [11:40<09:50,  1.02s/it][A
     54%|█████▍    | 682/1261 [11:41<09:49,  1.02s/it][A
     54%|█████▍    | 683/1261 [11:42<09:39,  1.00s/it][A
     54%|█████▍    | 684/1261 [11:43<09:31,  1.01it/s][A
     54%|█████▍    | 685/1261 [11:44<09:27,  1.01it/s][A
     54%|█████▍    | 686/1261 [11:45<09:24,  1.02it/s][A
     54%|█████▍    | 687/1261 [11:45<09:24,  1.02it/s][A
     55%|█████▍    | 688/1261 [11:47<09:27,  1.01it/s][A
     55%|█████▍    | 689/1261 [11:47<09:27,  1.01it/s][A
     55%|█████▍    | 690/1261 [11:48<09:23,  1.01it/s][A
     55%|█████▍    | 691/1261 [11:49<09:17,  1.02it/s][A
     55%|█████▍    | 692/1261 [11:50<09:17,  1.02it/s][A
     55%|█████▍    | 693/1261 [11:51<09:29,  1.00s/it][A
     55%|█████▌    | 694/1261 [11:52<09:29,  1.01s/it][A
     55%|█████▌    | 695/1261 [11:53<09:22,  1.01it/s][A
     55%|█████▌    | 696/1261 [11:54<09:20,  1.01it/s][A
     55%|█████▌    | 697/1261 [11:55<09:22,  1.00it/s][A
     55%|█████▌    | 698/1261 [11:56<09:21,  1.00it/s][A
     55%|█████▌    | 699/1261 [11:57<09:18,  1.01it/s][A
     56%|█████▌    | 700/1261 [11:58<09:16,  1.01it/s][A
     56%|█████▌    | 701/1261 [11:59<09:13,  1.01it/s][A
     56%|█████▌    | 702/1261 [12:00<09:08,  1.02it/s][A
     56%|█████▌    | 703/1261 [12:01<09:08,  1.02it/s][A
     56%|█████▌    | 704/1261 [12:02<09:08,  1.01it/s][A
     56%|█████▌    | 705/1261 [12:03<09:17,  1.00s/it][A
     56%|█████▌    | 706/1261 [12:04<09:13,  1.00it/s][A
     56%|█████▌    | 707/1261 [12:05<09:13,  1.00it/s][A
     56%|█████▌    | 708/1261 [12:06<09:11,  1.00it/s][A
     56%|█████▌    | 709/1261 [12:07<09:14,  1.00s/it][A
     56%|█████▋    | 710/1261 [12:08<09:13,  1.00s/it][A
     56%|█████▋    | 711/1261 [12:09<09:09,  1.00it/s][A
     56%|█████▋    | 712/1261 [12:10<09:03,  1.01it/s][A
     57%|█████▋    | 713/1261 [12:11<09:02,  1.01it/s][A
     57%|█████▋    | 714/1261 [12:12<09:08,  1.00s/it][A
     57%|█████▋    | 715/1261 [12:13<09:13,  1.01s/it][A
     57%|█████▋    | 716/1261 [12:14<09:15,  1.02s/it][A
     57%|█████▋    | 717/1261 [12:15<09:11,  1.01s/it][A
     57%|█████▋    | 718/1261 [12:16<09:12,  1.02s/it][A
     57%|█████▋    | 719/1261 [12:17<09:07,  1.01s/it][A
     57%|█████▋    | 720/1261 [12:18<09:10,  1.02s/it][A
     57%|█████▋    | 721/1261 [12:20<09:13,  1.03s/it][A
     57%|█████▋    | 722/1261 [12:21<09:08,  1.02s/it][A
     57%|█████▋    | 723/1261 [12:22<09:08,  1.02s/it][A
     57%|█████▋    | 724/1261 [12:23<09:05,  1.02s/it][A
     57%|█████▋    | 725/1261 [12:24<09:00,  1.01s/it][A
     58%|█████▊    | 726/1261 [12:25<08:59,  1.01s/it][A
     58%|█████▊    | 727/1261 [12:26<08:59,  1.01s/it][A
     58%|█████▊    | 728/1261 [12:27<08:59,  1.01s/it][A
     58%|█████▊    | 729/1261 [12:28<08:58,  1.01s/it][A
     58%|█████▊    | 730/1261 [12:29<08:54,  1.01s/it][A
     58%|█████▊    | 731/1261 [12:30<08:49,  1.00it/s][A
     58%|█████▊    | 732/1261 [12:31<08:42,  1.01it/s][A
     58%|█████▊    | 733/1261 [12:32<08:41,  1.01it/s][A
     58%|█████▊    | 734/1261 [12:33<08:47,  1.00s/it][A
     58%|█████▊    | 735/1261 [12:34<08:52,  1.01s/it][A
     58%|█████▊    | 736/1261 [12:35<08:48,  1.01s/it][A
     58%|█████▊    | 737/1261 [12:36<08:52,  1.02s/it][A
     59%|█████▊    | 738/1261 [12:37<08:46,  1.01s/it][A
     59%|█████▊    | 739/1261 [12:38<08:48,  1.01s/it][A
     59%|█████▊    | 740/1261 [12:39<08:50,  1.02s/it][A
     59%|█████▉    | 741/1261 [12:40<08:45,  1.01s/it][A
     59%|█████▉    | 742/1261 [12:41<08:44,  1.01s/it][A
     59%|█████▉    | 743/1261 [12:42<08:39,  1.00s/it][A
     59%|█████▉    | 744/1261 [12:43<08:39,  1.00s/it][A
     59%|█████▉    | 745/1261 [12:44<08:44,  1.02s/it][A
     59%|█████▉    | 746/1261 [12:45<08:45,  1.02s/it][A
     59%|█████▉    | 747/1261 [12:46<08:46,  1.02s/it][A
     59%|█████▉    | 748/1261 [12:47<08:44,  1.02s/it][A
     59%|█████▉    | 749/1261 [12:48<08:40,  1.02s/it][A
     59%|█████▉    | 750/1261 [12:49<08:40,  1.02s/it][A
     60%|█████▉    | 751/1261 [12:50<08:38,  1.02s/it][A
     60%|█████▉    | 752/1261 [12:51<08:40,  1.02s/it][A
     60%|█████▉    | 753/1261 [12:52<08:35,  1.02s/it][A
     60%|█████▉    | 754/1261 [12:53<08:32,  1.01s/it][A
     60%|█████▉    | 755/1261 [12:54<08:36,  1.02s/it][A
     60%|█████▉    | 756/1261 [12:55<08:32,  1.02s/it][A
     60%|██████    | 757/1261 [12:56<08:25,  1.00s/it][A
     60%|██████    | 758/1261 [12:57<08:28,  1.01s/it][A
     60%|██████    | 759/1261 [12:58<08:21,  1.00it/s][A
     60%|██████    | 760/1261 [12:59<08:23,  1.01s/it][A
     60%|██████    | 761/1261 [13:00<08:18,  1.00it/s][A
     60%|██████    | 762/1261 [13:01<08:20,  1.00s/it][A
     61%|██████    | 763/1261 [13:02<08:24,  1.01s/it][A
     61%|██████    | 764/1261 [13:03<08:22,  1.01s/it][A
     61%|██████    | 765/1261 [13:04<08:18,  1.01s/it][A
     61%|██████    | 766/1261 [13:05<08:19,  1.01s/it][A
     61%|██████    | 767/1261 [13:06<08:27,  1.03s/it][A
     61%|██████    | 768/1261 [13:07<08:20,  1.01s/it][A
     61%|██████    | 769/1261 [13:08<08:15,  1.01s/it][A
     61%|██████    | 770/1261 [13:09<08:11,  1.00s/it][A
     61%|██████    | 771/1261 [13:10<08:16,  1.01s/it][A
     61%|██████    | 772/1261 [13:11<08:14,  1.01s/it][A
     61%|██████▏   | 773/1261 [13:12<08:10,  1.01s/it][A
     61%|██████▏   | 774/1261 [13:13<08:04,  1.00it/s][A
     61%|██████▏   | 775/1261 [13:14<08:01,  1.01it/s][A
     62%|██████▏   | 776/1261 [13:15<07:59,  1.01it/s][A
     62%|██████▏   | 777/1261 [13:16<08:00,  1.01it/s][A
     62%|██████▏   | 778/1261 [13:17<07:58,  1.01it/s][A
     62%|██████▏   | 779/1261 [13:18<08:02,  1.00s/it][A
     62%|██████▏   | 780/1261 [13:19<08:00,  1.00it/s][A
     62%|██████▏   | 781/1261 [13:20<08:05,  1.01s/it][A
     62%|██████▏   | 782/1261 [13:21<08:05,  1.01s/it][A
     62%|██████▏   | 783/1261 [13:22<08:07,  1.02s/it][A
     62%|██████▏   | 784/1261 [13:23<08:06,  1.02s/it][A
     62%|██████▏   | 785/1261 [13:24<08:04,  1.02s/it][A
     62%|██████▏   | 786/1261 [13:25<08:03,  1.02s/it][A
     62%|██████▏   | 787/1261 [13:26<08:03,  1.02s/it][A
     62%|██████▏   | 788/1261 [13:27<07:57,  1.01s/it][A
     63%|██████▎   | 789/1261 [13:28<07:54,  1.00s/it][A
     63%|██████▎   | 790/1261 [13:29<07:47,  1.01it/s][A
     63%|██████▎   | 791/1261 [13:30<07:48,  1.00it/s][A
     63%|██████▎   | 792/1261 [13:31<07:49,  1.00s/it][A
     63%|██████▎   | 793/1261 [13:32<07:50,  1.01s/it][A
     63%|██████▎   | 794/1261 [13:33<07:45,  1.00it/s][A
     63%|██████▎   | 795/1261 [13:34<07:41,  1.01it/s][A
     63%|██████▎   | 796/1261 [13:35<07:39,  1.01it/s][A
     63%|██████▎   | 797/1261 [13:36<07:37,  1.01it/s][A
     63%|██████▎   | 798/1261 [13:37<07:38,  1.01it/s][A
     63%|██████▎   | 799/1261 [13:38<07:39,  1.01it/s][A
     63%|██████▎   | 800/1261 [13:39<07:38,  1.00it/s][A
     64%|██████▎   | 801/1261 [13:40<07:38,  1.00it/s][A
     64%|██████▎   | 802/1261 [13:41<07:37,  1.00it/s][A
     64%|██████▎   | 803/1261 [13:42<07:39,  1.00s/it][A
     64%|██████▍   | 804/1261 [13:43<07:38,  1.00s/it][A
     64%|██████▍   | 805/1261 [13:44<07:34,  1.00it/s][A
     64%|██████▍   | 806/1261 [13:45<07:45,  1.02s/it][A
     64%|██████▍   | 807/1261 [13:46<07:50,  1.04s/it][A
     64%|██████▍   | 808/1261 [13:47<07:52,  1.04s/it][A
     64%|██████▍   | 809/1261 [13:48<08:00,  1.06s/it][A
     64%|██████▍   | 810/1261 [13:49<07:46,  1.04s/it][A
     64%|██████▍   | 811/1261 [13:50<07:50,  1.05s/it][A
     64%|██████▍   | 812/1261 [13:51<07:55,  1.06s/it][A
     64%|██████▍   | 813/1261 [13:53<07:50,  1.05s/it][A
     65%|██████▍   | 814/1261 [13:54<07:46,  1.04s/it][A
     65%|██████▍   | 815/1261 [13:55<07:37,  1.02s/it][A
     65%|██████▍   | 816/1261 [13:56<07:32,  1.02s/it][A
     65%|██████▍   | 817/1261 [13:57<07:26,  1.00s/it][A
     65%|██████▍   | 818/1261 [13:58<07:25,  1.01s/it][A
     65%|██████▍   | 819/1261 [13:59<07:23,  1.00s/it][A
     65%|██████▌   | 820/1261 [14:00<07:21,  1.00s/it][A
     65%|██████▌   | 821/1261 [14:00<07:17,  1.01it/s][A
     65%|██████▌   | 822/1261 [14:01<07:18,  1.00it/s][A
     65%|██████▌   | 823/1261 [14:03<07:19,  1.00s/it][A
     65%|██████▌   | 824/1261 [14:04<07:20,  1.01s/it][A
     65%|██████▌   | 825/1261 [14:05<07:17,  1.00s/it][A
     66%|██████▌   | 826/1261 [14:06<07:16,  1.00s/it][A
     66%|██████▌   | 827/1261 [14:07<07:15,  1.00s/it][A
     66%|██████▌   | 828/1261 [14:08<07:12,  1.00it/s][A
     66%|██████▌   | 829/1261 [14:09<07:14,  1.00s/it][A
     66%|██████▌   | 830/1261 [14:10<07:11,  1.00s/it][A
     66%|██████▌   | 831/1261 [14:11<07:09,  1.00it/s][A
     66%|██████▌   | 832/1261 [14:12<07:07,  1.00it/s][A
     66%|██████▌   | 833/1261 [14:12<07:02,  1.01it/s][A
     66%|██████▌   | 834/1261 [14:13<07:04,  1.01it/s][A
     66%|██████▌   | 835/1261 [14:14<07:02,  1.01it/s][A
     66%|██████▋   | 836/1261 [14:15<07:02,  1.01it/s][A
     66%|██████▋   | 837/1261 [14:17<07:06,  1.01s/it][A
     66%|██████▋   | 838/1261 [14:18<07:10,  1.02s/it][A
     67%|██████▋   | 839/1261 [14:19<07:11,  1.02s/it][A
     67%|██████▋   | 840/1261 [14:20<07:07,  1.02s/it][A
     67%|██████▋   | 841/1261 [14:21<07:07,  1.02s/it][A
     67%|██████▋   | 842/1261 [14:22<07:01,  1.01s/it][A
     67%|██████▋   | 843/1261 [14:23<07:00,  1.01s/it][A
     67%|██████▋   | 844/1261 [14:24<06:57,  1.00s/it][A
     67%|██████▋   | 845/1261 [14:25<06:52,  1.01it/s][A
     67%|██████▋   | 846/1261 [14:26<06:50,  1.01it/s][A
     67%|██████▋   | 847/1261 [14:27<06:49,  1.01it/s][A
     67%|██████▋   | 848/1261 [14:28<06:50,  1.01it/s][A
     67%|██████▋   | 849/1261 [14:29<06:47,  1.01it/s][A
     67%|██████▋   | 850/1261 [14:30<06:50,  1.00it/s][A
     67%|██████▋   | 851/1261 [14:31<06:48,  1.00it/s][A
     68%|██████▊   | 852/1261 [14:32<06:49,  1.00s/it][A
     68%|██████▊   | 853/1261 [14:33<06:49,  1.00s/it][A
     68%|██████▊   | 854/1261 [14:34<06:50,  1.01s/it][A
     68%|██████▊   | 855/1261 [14:35<06:49,  1.01s/it][A
     68%|██████▊   | 856/1261 [14:36<06:54,  1.02s/it][A
     68%|██████▊   | 857/1261 [14:37<06:50,  1.02s/it][A
     68%|██████▊   | 858/1261 [14:38<06:46,  1.01s/it][A
     68%|██████▊   | 859/1261 [14:39<06:44,  1.01s/it][A
     68%|██████▊   | 860/1261 [14:40<06:42,  1.00s/it][A
     68%|██████▊   | 861/1261 [14:41<06:38,  1.00it/s][A
     68%|██████▊   | 862/1261 [14:42<06:43,  1.01s/it][A
     68%|██████▊   | 863/1261 [14:43<06:43,  1.01s/it][A
     69%|██████▊   | 864/1261 [14:44<06:50,  1.03s/it][A
     69%|██████▊   | 865/1261 [14:45<06:45,  1.02s/it][A
     69%|██████▊   | 866/1261 [14:46<06:44,  1.02s/it][A
     69%|██████▉   | 867/1261 [14:47<06:39,  1.01s/it][A
     69%|██████▉   | 868/1261 [14:48<06:35,  1.01s/it][A
     69%|██████▉   | 869/1261 [14:49<06:33,  1.00s/it][A
     69%|██████▉   | 870/1261 [14:50<06:30,  1.00it/s][A
     69%|██████▉   | 871/1261 [14:51<06:32,  1.01s/it][A
     69%|██████▉   | 872/1261 [14:52<06:32,  1.01s/it][A
     69%|██████▉   | 873/1261 [14:53<06:29,  1.00s/it][A
     69%|██████▉   | 874/1261 [14:54<06:29,  1.01s/it][A
     69%|██████▉   | 875/1261 [14:55<06:25,  1.00it/s][A
     69%|██████▉   | 876/1261 [14:56<06:21,  1.01it/s][A
     70%|██████▉   | 877/1261 [14:57<06:24,  1.00s/it][A
     70%|██████▉   | 878/1261 [14:58<06:22,  1.00it/s][A
     70%|██████▉   | 879/1261 [14:59<06:19,  1.01it/s][A
     70%|██████▉   | 880/1261 [15:00<06:16,  1.01it/s][A
     70%|██████▉   | 881/1261 [15:01<06:18,  1.00it/s][A
     70%|██████▉   | 882/1261 [15:02<06:19,  1.00s/it][A
     70%|███████   | 883/1261 [15:03<06:20,  1.01s/it][A
     70%|███████   | 884/1261 [15:04<06:16,  1.00it/s][A
     70%|███████   | 885/1261 [15:05<06:15,  1.00it/s][A
     70%|███████   | 886/1261 [15:06<06:17,  1.01s/it][A
     70%|███████   | 887/1261 [15:07<06:21,  1.02s/it][A
     70%|███████   | 888/1261 [15:08<06:17,  1.01s/it][A
     70%|███████   | 889/1261 [15:09<06:15,  1.01s/it][A
     71%|███████   | 890/1261 [15:10<06:11,  1.00s/it][A
     71%|███████   | 891/1261 [15:11<06:07,  1.01it/s][A
     71%|███████   | 892/1261 [15:12<06:10,  1.00s/it][A
     71%|███████   | 893/1261 [15:13<06:05,  1.01it/s][A
     71%|███████   | 894/1261 [15:14<06:03,  1.01it/s][A
     71%|███████   | 895/1261 [15:15<06:03,  1.01it/s][A
     71%|███████   | 896/1261 [15:16<06:06,  1.00s/it][A
     71%|███████   | 897/1261 [15:17<06:11,  1.02s/it][A
     71%|███████   | 898/1261 [15:18<06:07,  1.01s/it][A
     71%|███████▏  | 899/1261 [15:19<06:05,  1.01s/it][A
     71%|███████▏  | 900/1261 [15:20<06:09,  1.02s/it][A
     71%|███████▏  | 901/1261 [15:21<06:07,  1.02s/it][A
     72%|███████▏  | 902/1261 [15:22<06:01,  1.01s/it][A
     72%|███████▏  | 903/1261 [15:23<05:56,  1.01it/s][A
     72%|███████▏  | 904/1261 [15:24<05:59,  1.01s/it][A
     72%|███████▏  | 905/1261 [15:25<06:00,  1.01s/it][A
     72%|███████▏  | 906/1261 [15:26<05:59,  1.01s/it][A
     72%|███████▏  | 907/1261 [15:27<05:53,  1.00it/s][A
     72%|███████▏  | 908/1261 [15:28<05:51,  1.00it/s][A
     72%|███████▏  | 909/1261 [15:29<05:48,  1.01it/s][A
     72%|███████▏  | 910/1261 [15:30<05:49,  1.01it/s][A
     72%|███████▏  | 911/1261 [15:31<05:50,  1.00s/it][A
     72%|███████▏  | 912/1261 [15:32<05:49,  1.00s/it][A
     72%|███████▏  | 913/1261 [15:33<05:50,  1.01s/it][A
     72%|███████▏  | 914/1261 [15:34<05:46,  1.00it/s][A
     73%|███████▎  | 915/1261 [15:35<05:42,  1.01it/s][A
     73%|███████▎  | 916/1261 [15:36<05:42,  1.01it/s][A
     73%|███████▎  | 917/1261 [15:37<05:43,  1.00it/s][A
     73%|███████▎  | 918/1261 [15:38<05:39,  1.01it/s][A
     73%|███████▎  | 919/1261 [15:39<05:38,  1.01it/s][A
     73%|███████▎  | 920/1261 [15:40<05:38,  1.01it/s][A
     73%|███████▎  | 921/1261 [15:41<05:36,  1.01it/s][A
     73%|███████▎  | 922/1261 [15:42<05:34,  1.01it/s][A
     73%|███████▎  | 923/1261 [15:43<05:35,  1.01it/s][A
     73%|███████▎  | 924/1261 [15:44<05:38,  1.00s/it][A
     73%|███████▎  | 925/1261 [15:45<05:35,  1.00it/s][A
     73%|███████▎  | 926/1261 [15:46<05:36,  1.01s/it][A
     74%|███████▎  | 927/1261 [15:47<05:38,  1.01s/it][A
     74%|███████▎  | 928/1261 [15:48<05:34,  1.01s/it][A
     74%|███████▎  | 929/1261 [15:49<05:32,  1.00s/it][A
     74%|███████▍  | 930/1261 [15:50<05:30,  1.00it/s][A
     74%|███████▍  | 931/1261 [15:51<05:27,  1.01it/s][A
     74%|███████▍  | 932/1261 [15:52<05:25,  1.01it/s][A
     74%|███████▍  | 933/1261 [15:53<05:26,  1.01it/s][A
     74%|███████▍  | 934/1261 [15:54<05:25,  1.01it/s][A
     74%|███████▍  | 935/1261 [15:55<05:28,  1.01s/it][A
     74%|███████▍  | 936/1261 [15:56<05:28,  1.01s/it][A
     74%|███████▍  | 937/1261 [15:57<05:24,  1.00s/it][A
     74%|███████▍  | 938/1261 [15:58<05:23,  1.00s/it][A
     74%|███████▍  | 939/1261 [15:59<05:20,  1.00it/s][A
     75%|███████▍  | 940/1261 [16:00<05:20,  1.00it/s][A
     75%|███████▍  | 941/1261 [16:01<05:18,  1.00it/s][A
     75%|███████▍  | 942/1261 [16:02<05:18,  1.00it/s][A
     75%|███████▍  | 943/1261 [16:03<05:17,  1.00it/s][A
     75%|███████▍  | 944/1261 [16:04<05:16,  1.00it/s][A
     75%|███████▍  | 945/1261 [16:05<05:15,  1.00it/s][A
     75%|███████▌  | 946/1261 [16:06<05:14,  1.00it/s][A
     75%|███████▌  | 947/1261 [16:07<05:14,  1.00s/it][A
     75%|███████▌  | 948/1261 [16:08<05:11,  1.00it/s][A
     75%|███████▌  | 949/1261 [16:09<05:09,  1.01it/s][A
     75%|███████▌  | 950/1261 [16:10<05:09,  1.01it/s][A
     75%|███████▌  | 951/1261 [16:11<05:11,  1.00s/it][A
     75%|███████▌  | 952/1261 [16:12<05:10,  1.01s/it][A
     76%|███████▌  | 953/1261 [16:13<05:06,  1.00it/s][A
     76%|███████▌  | 954/1261 [16:14<05:07,  1.00s/it][A
     76%|███████▌  | 955/1261 [16:15<05:07,  1.00s/it][A
     76%|███████▌  | 956/1261 [16:16<05:07,  1.01s/it][A
     76%|███████▌  | 957/1261 [16:17<05:09,  1.02s/it][A
     76%|███████▌  | 958/1261 [16:18<05:09,  1.02s/it][A
     76%|███████▌  | 959/1261 [16:19<05:06,  1.01s/it][A
     76%|███████▌  | 960/1261 [16:20<05:04,  1.01s/it][A
     76%|███████▌  | 961/1261 [16:21<05:04,  1.02s/it][A
     76%|███████▋  | 962/1261 [16:22<05:07,  1.03s/it][A
     76%|███████▋  | 963/1261 [16:23<05:03,  1.02s/it][A
     76%|███████▋  | 964/1261 [16:24<05:00,  1.01s/it][A
     77%|███████▋  | 965/1261 [16:25<04:57,  1.01s/it][A
     77%|███████▋  | 966/1261 [16:26<04:56,  1.00s/it][A
     77%|███████▋  | 967/1261 [16:27<04:56,  1.01s/it][A
     77%|███████▋  | 968/1261 [16:28<04:55,  1.01s/it][A
     77%|███████▋  | 969/1261 [16:29<04:54,  1.01s/it][A
     77%|███████▋  | 970/1261 [16:30<04:56,  1.02s/it][A
     77%|███████▋  | 971/1261 [16:31<04:53,  1.01s/it][A
     77%|███████▋  | 972/1261 [16:32<04:51,  1.01s/it][A
     77%|███████▋  | 973/1261 [16:33<04:51,  1.01s/it][A
     77%|███████▋  | 974/1261 [16:34<04:47,  1.00s/it][A
     77%|███████▋  | 975/1261 [16:35<04:46,  1.00s/it][A
     77%|███████▋  | 976/1261 [16:36<04:46,  1.00s/it][A
     77%|███████▋  | 977/1261 [16:37<04:49,  1.02s/it][A
     78%|███████▊  | 978/1261 [16:38<04:46,  1.01s/it][A
     78%|███████▊  | 979/1261 [16:39<04:45,  1.01s/it][A
     78%|███████▊  | 980/1261 [16:40<04:44,  1.01s/it][A
     78%|███████▊  | 981/1261 [16:41<04:42,  1.01s/it][A
     78%|███████▊  | 982/1261 [16:42<04:46,  1.03s/it][A
     78%|███████▊  | 983/1261 [16:43<04:45,  1.03s/it][A
     78%|███████▊  | 984/1261 [16:44<04:47,  1.04s/it][A
     78%|███████▊  | 985/1261 [16:45<04:45,  1.04s/it][A
     78%|███████▊  | 986/1261 [16:46<04:44,  1.03s/it][A
     78%|███████▊  | 987/1261 [16:47<04:42,  1.03s/it][A
     78%|███████▊  | 988/1261 [16:48<04:39,  1.03s/it][A
     78%|███████▊  | 989/1261 [16:49<04:37,  1.02s/it][A
     79%|███████▊  | 990/1261 [16:50<04:33,  1.01s/it][A
     79%|███████▊  | 991/1261 [16:51<04:30,  1.00s/it][A
     79%|███████▊  | 992/1261 [16:52<04:28,  1.00it/s][A
     79%|███████▊  | 993/1261 [16:53<04:28,  1.00s/it][A
     79%|███████▉  | 994/1261 [16:54<04:30,  1.01s/it][A
     79%|███████▉  | 995/1261 [16:55<04:29,  1.01s/it][A
     79%|███████▉  | 996/1261 [16:56<04:29,  1.02s/it][A
     79%|███████▉  | 997/1261 [16:57<04:28,  1.02s/it][A
     79%|███████▉  | 998/1261 [16:58<04:28,  1.02s/it][A
     79%|███████▉  | 999/1261 [16:59<04:27,  1.02s/it][A
     79%|███████▉  | 1000/1261 [17:00<04:24,  1.01s/it][A
     79%|███████▉  | 1001/1261 [17:01<04:23,  1.01s/it][A
     79%|███████▉  | 1002/1261 [17:02<04:21,  1.01s/it][A
     80%|███████▉  | 1003/1261 [17:03<04:18,  1.00s/it][A
     80%|███████▉  | 1004/1261 [17:04<04:16,  1.00it/s][A
     80%|███████▉  | 1005/1261 [17:05<04:15,  1.00it/s][A
     80%|███████▉  | 1006/1261 [17:06<04:15,  1.00s/it][A
     80%|███████▉  | 1007/1261 [17:07<04:15,  1.01s/it][A
     80%|███████▉  | 1008/1261 [17:08<04:13,  1.00s/it][A
     80%|████████  | 1009/1261 [17:09<04:11,  1.00it/s][A
     80%|████████  | 1010/1261 [17:10<04:12,  1.00s/it][A
     80%|████████  | 1011/1261 [17:11<04:10,  1.00s/it][A
     80%|████████  | 1012/1261 [17:12<04:09,  1.00s/it][A
     80%|████████  | 1013/1261 [17:14<04:09,  1.01s/it][A
     80%|████████  | 1014/1261 [17:15<04:12,  1.02s/it][A
     80%|████████  | 1015/1261 [17:16<04:10,  1.02s/it][A
     81%|████████  | 1016/1261 [17:17<04:08,  1.02s/it][A
     81%|████████  | 1017/1261 [17:18<04:07,  1.01s/it][A
     81%|████████  | 1018/1261 [17:19<04:07,  1.02s/it][A
     81%|████████  | 1019/1261 [17:20<04:09,  1.03s/it][A
     81%|████████  | 1020/1261 [17:21<04:07,  1.03s/it][A
     81%|████████  | 1021/1261 [17:22<04:05,  1.02s/it][A
     81%|████████  | 1022/1261 [17:23<04:02,  1.02s/it][A
     81%|████████  | 1023/1261 [17:24<04:01,  1.02s/it][A
     81%|████████  | 1024/1261 [17:25<04:00,  1.01s/it][A
     81%|████████▏ | 1025/1261 [17:26<03:57,  1.01s/it][A
     81%|████████▏ | 1026/1261 [17:27<03:58,  1.02s/it][A
     81%|████████▏ | 1027/1261 [17:28<03:55,  1.01s/it][A
     82%|████████▏ | 1028/1261 [17:29<03:52,  1.00it/s][A
     82%|████████▏ | 1029/1261 [17:30<03:54,  1.01s/it][A
     82%|████████▏ | 1030/1261 [17:31<03:53,  1.01s/it][A
     82%|████████▏ | 1031/1261 [17:32<03:53,  1.02s/it][A
     82%|████████▏ | 1032/1261 [17:33<03:50,  1.00s/it][A
     82%|████████▏ | 1033/1261 [17:34<03:47,  1.00it/s][A
     82%|████████▏ | 1034/1261 [17:35<03:46,  1.00it/s][A
     82%|████████▏ | 1035/1261 [17:36<03:44,  1.01it/s][A
     82%|████████▏ | 1036/1261 [17:37<03:44,  1.00it/s][A
     82%|████████▏ | 1037/1261 [17:38<03:43,  1.00it/s][A
     82%|████████▏ | 1038/1261 [17:39<03:41,  1.01it/s][A
     82%|████████▏ | 1039/1261 [17:40<03:41,  1.00it/s][A
     82%|████████▏ | 1040/1261 [17:41<03:39,  1.00it/s][A
     83%|████████▎ | 1041/1261 [17:42<03:38,  1.01it/s][A
     83%|████████▎ | 1042/1261 [17:43<03:37,  1.01it/s][A
     83%|████████▎ | 1043/1261 [17:44<03:36,  1.01it/s][A
     83%|████████▎ | 1044/1261 [17:45<03:38,  1.01s/it][A
     83%|████████▎ | 1045/1261 [17:46<03:36,  1.00s/it][A
     83%|████████▎ | 1046/1261 [17:47<03:33,  1.01it/s][A
     83%|████████▎ | 1047/1261 [17:48<03:32,  1.01it/s][A
     83%|████████▎ | 1048/1261 [17:49<03:30,  1.01it/s][A
     83%|████████▎ | 1049/1261 [17:50<03:30,  1.01it/s][A
     83%|████████▎ | 1050/1261 [17:51<03:31,  1.00s/it][A
     83%|████████▎ | 1051/1261 [17:52<03:30,  1.00s/it][A
     83%|████████▎ | 1052/1261 [17:53<03:29,  1.00s/it][A
     84%|████████▎ | 1053/1261 [17:54<03:30,  1.01s/it][A
     84%|████████▎ | 1054/1261 [17:55<03:28,  1.00s/it][A
     84%|████████▎ | 1055/1261 [17:56<03:27,  1.01s/it][A
     84%|████████▎ | 1056/1261 [17:57<03:27,  1.01s/it][A
     84%|████████▍ | 1057/1261 [17:58<03:26,  1.01s/it][A
     84%|████████▍ | 1058/1261 [17:59<03:23,  1.00s/it][A
     84%|████████▍ | 1059/1261 [18:00<03:23,  1.01s/it][A
     84%|████████▍ | 1060/1261 [18:01<03:21,  1.00s/it][A
     84%|████████▍ | 1061/1261 [18:02<03:20,  1.00s/it][A
     84%|████████▍ | 1062/1261 [18:03<03:19,  1.00s/it][A
     84%|████████▍ | 1063/1261 [18:04<03:17,  1.00it/s][A
     84%|████████▍ | 1064/1261 [18:05<03:16,  1.00it/s][A
     84%|████████▍ | 1065/1261 [18:06<03:18,  1.01s/it][A
     85%|████████▍ | 1066/1261 [18:07<03:18,  1.02s/it][A
     85%|████████▍ | 1067/1261 [18:08<03:17,  1.02s/it][A
     85%|████████▍ | 1068/1261 [18:09<03:16,  1.02s/it][A
     85%|████████▍ | 1069/1261 [18:10<03:15,  1.02s/it][A
     85%|████████▍ | 1070/1261 [18:11<03:13,  1.01s/it][A
     85%|████████▍ | 1071/1261 [18:12<03:13,  1.02s/it][A
     85%|████████▌ | 1072/1261 [18:13<03:12,  1.02s/it][A
     85%|████████▌ | 1073/1261 [18:14<03:11,  1.02s/it][A
     85%|████████▌ | 1074/1261 [18:15<03:09,  1.01s/it][A
     85%|████████▌ | 1075/1261 [18:16<03:08,  1.01s/it][A
     85%|████████▌ | 1076/1261 [18:17<03:09,  1.02s/it][A
     85%|████████▌ | 1077/1261 [18:18<03:07,  1.02s/it][A
     85%|████████▌ | 1078/1261 [18:19<03:06,  1.02s/it][A
     86%|████████▌ | 1079/1261 [18:20<03:03,  1.01s/it][A
     86%|████████▌ | 1080/1261 [18:21<03:02,  1.01s/it][A
     86%|████████▌ | 1081/1261 [18:22<03:02,  1.01s/it][A
     86%|████████▌ | 1082/1261 [18:23<03:02,  1.02s/it][A
     86%|████████▌ | 1083/1261 [18:24<03:01,  1.02s/it][A
     86%|████████▌ | 1084/1261 [18:25<02:59,  1.02s/it][A
     86%|████████▌ | 1085/1261 [18:26<02:58,  1.02s/it][A
     86%|████████▌ | 1086/1261 [18:27<02:58,  1.02s/it][A
     86%|████████▌ | 1087/1261 [18:28<03:04,  1.06s/it][A
     86%|████████▋ | 1088/1261 [18:29<03:01,  1.05s/it][A
     86%|████████▋ | 1089/1261 [18:30<02:57,  1.03s/it][A
     86%|████████▋ | 1090/1261 [18:31<02:57,  1.04s/it][A
     87%|████████▋ | 1091/1261 [18:32<02:56,  1.04s/it][A
     87%|████████▋ | 1092/1261 [18:33<02:53,  1.03s/it][A
     87%|████████▋ | 1093/1261 [18:34<02:51,  1.02s/it][A
     87%|████████▋ | 1094/1261 [18:36<02:53,  1.04s/it][A
     87%|████████▋ | 1095/1261 [18:37<02:53,  1.04s/it][A
     87%|████████▋ | 1096/1261 [18:38<02:53,  1.05s/it][A
     87%|████████▋ | 1097/1261 [18:39<02:49,  1.03s/it][A
     87%|████████▋ | 1098/1261 [18:40<02:47,  1.03s/it][A
     87%|████████▋ | 1099/1261 [18:41<02:46,  1.03s/it][A
     87%|████████▋ | 1100/1261 [18:42<02:43,  1.01s/it][A
     87%|████████▋ | 1101/1261 [18:43<02:42,  1.02s/it][A
     87%|████████▋ | 1102/1261 [18:44<02:40,  1.01s/it][A
     87%|████████▋ | 1103/1261 [18:45<02:43,  1.03s/it][A
     88%|████████▊ | 1104/1261 [18:46<02:47,  1.07s/it][A
     88%|████████▊ | 1105/1261 [18:47<02:45,  1.06s/it][A
     88%|████████▊ | 1106/1261 [18:48<02:46,  1.07s/it][A
     88%|████████▊ | 1107/1261 [18:49<02:44,  1.07s/it][A
     88%|████████▊ | 1108/1261 [18:50<02:46,  1.09s/it][A
     88%|████████▊ | 1109/1261 [18:51<02:47,  1.10s/it][A
     88%|████████▊ | 1110/1261 [18:52<02:42,  1.08s/it][A
     88%|████████▊ | 1111/1261 [18:53<02:39,  1.06s/it][A
     88%|████████▊ | 1112/1261 [18:54<02:36,  1.05s/it][A
     88%|████████▊ | 1113/1261 [18:55<02:33,  1.04s/it][A
     88%|████████▊ | 1114/1261 [18:56<02:32,  1.03s/it][A
     88%|████████▊ | 1115/1261 [18:57<02:29,  1.03s/it][A
     89%|████████▊ | 1116/1261 [18:58<02:27,  1.02s/it][A
     89%|████████▊ | 1117/1261 [18:59<02:24,  1.00s/it][A
     89%|████████▊ | 1118/1261 [19:00<02:23,  1.00s/it][A
     89%|████████▊ | 1119/1261 [19:01<02:21,  1.00it/s][A
     89%|████████▉ | 1120/1261 [19:02<02:20,  1.00it/s][A
     89%|████████▉ | 1121/1261 [19:03<02:20,  1.00s/it][A
     89%|████████▉ | 1122/1261 [19:04<02:18,  1.00it/s][A
     89%|████████▉ | 1123/1261 [19:05<02:19,  1.01s/it][A
     89%|████████▉ | 1124/1261 [19:06<02:16,  1.00it/s][A
     89%|████████▉ | 1125/1261 [19:07<02:16,  1.01s/it][A
     89%|████████▉ | 1126/1261 [19:08<02:14,  1.01it/s][A
     89%|████████▉ | 1127/1261 [19:09<02:14,  1.00s/it][A
     89%|████████▉ | 1128/1261 [19:10<02:12,  1.01it/s][A
     90%|████████▉ | 1129/1261 [19:11<02:10,  1.01it/s][A
     90%|████████▉ | 1130/1261 [19:12<02:11,  1.00s/it][A
     90%|████████▉ | 1131/1261 [19:13<02:11,  1.01s/it][A
     90%|████████▉ | 1132/1261 [19:14<02:09,  1.01s/it][A
     90%|████████▉ | 1133/1261 [19:15<02:08,  1.00s/it][A
     90%|████████▉ | 1134/1261 [19:16<02:06,  1.01it/s][A
     90%|█████████ | 1135/1261 [19:17<02:05,  1.00it/s][A
     90%|█████████ | 1136/1261 [19:18<02:05,  1.00s/it][A
     90%|█████████ | 1137/1261 [19:19<02:04,  1.00s/it][A
     90%|█████████ | 1138/1261 [19:20<02:03,  1.00s/it][A
     90%|█████████ | 1139/1261 [19:21<02:01,  1.00it/s][A
     90%|█████████ | 1140/1261 [19:22<01:59,  1.01it/s][A
     90%|█████████ | 1141/1261 [19:23<01:57,  1.02it/s][A
     91%|█████████ | 1142/1261 [19:24<01:57,  1.01it/s][A
     91%|█████████ | 1143/1261 [19:25<01:57,  1.00it/s][A
     91%|█████████ | 1144/1261 [19:26<01:56,  1.00it/s][A
     91%|█████████ | 1145/1261 [19:27<01:55,  1.01it/s][A
     91%|█████████ | 1146/1261 [19:28<01:53,  1.01it/s][A
     91%|█████████ | 1147/1261 [19:29<01:52,  1.01it/s][A
     91%|█████████ | 1148/1261 [19:30<01:52,  1.00it/s][A
     91%|█████████ | 1149/1261 [19:31<01:51,  1.01it/s][A
     91%|█████████ | 1150/1261 [19:32<01:50,  1.00it/s][A
     91%|█████████▏| 1151/1261 [19:33<01:49,  1.01it/s][A
     91%|█████████▏| 1152/1261 [19:34<01:49,  1.00s/it][A
     91%|█████████▏| 1153/1261 [19:35<01:49,  1.01s/it][A
     92%|█████████▏| 1154/1261 [19:36<01:48,  1.01s/it][A
     92%|█████████▏| 1155/1261 [19:37<01:46,  1.01s/it][A
     92%|█████████▏| 1156/1261 [19:38<01:45,  1.00s/it][A
     92%|█████████▏| 1157/1261 [19:39<01:43,  1.01it/s][A
     92%|█████████▏| 1158/1261 [19:40<01:42,  1.01it/s][A
     92%|█████████▏| 1159/1261 [19:41<01:41,  1.01it/s][A
     92%|█████████▏| 1160/1261 [19:42<01:39,  1.01it/s][A
     92%|█████████▏| 1161/1261 [19:43<01:39,  1.00it/s][A
     92%|█████████▏| 1162/1261 [19:44<01:38,  1.01it/s][A
     92%|█████████▏| 1163/1261 [19:45<01:37,  1.00it/s][A
     92%|█████████▏| 1164/1261 [19:46<01:36,  1.01it/s][A
     92%|█████████▏| 1165/1261 [19:47<01:34,  1.02it/s][A
     92%|█████████▏| 1166/1261 [19:48<01:34,  1.01it/s][A
     93%|█████████▎| 1167/1261 [19:49<01:34,  1.01s/it][A
     93%|█████████▎| 1168/1261 [19:50<01:32,  1.00it/s][A
     93%|█████████▎| 1169/1261 [19:51<01:31,  1.01it/s][A
     93%|█████████▎| 1170/1261 [19:52<01:29,  1.01it/s][A
     93%|█████████▎| 1171/1261 [19:53<01:28,  1.02it/s][A
     93%|█████████▎| 1172/1261 [19:54<01:28,  1.00it/s][A
     93%|█████████▎| 1173/1261 [19:55<01:27,  1.00it/s][A
     93%|█████████▎| 1174/1261 [19:56<01:27,  1.00s/it][A
     93%|█████████▎| 1175/1261 [19:57<01:26,  1.00s/it][A
     93%|█████████▎| 1176/1261 [19:58<01:24,  1.00it/s][A
     93%|█████████▎| 1177/1261 [19:59<01:23,  1.00it/s][A
     93%|█████████▎| 1178/1261 [20:00<01:22,  1.00it/s][A
     93%|█████████▎| 1179/1261 [20:01<01:22,  1.00s/it][A
     94%|█████████▎| 1180/1261 [20:02<01:21,  1.01s/it][A
     94%|█████████▎| 1181/1261 [20:03<01:20,  1.01s/it][A
     94%|█████████▎| 1182/1261 [20:04<01:19,  1.01s/it][A
     94%|█████████▍| 1183/1261 [20:05<01:18,  1.01s/it][A
     94%|█████████▍| 1184/1261 [20:06<01:17,  1.01s/it][A
     94%|█████████▍| 1185/1261 [20:07<01:16,  1.01s/it][A
     94%|█████████▍| 1186/1261 [20:08<01:15,  1.01s/it][A
     94%|█████████▍| 1187/1261 [20:09<01:14,  1.00s/it][A
     94%|█████████▍| 1188/1261 [20:10<01:12,  1.00it/s][A
     94%|█████████▍| 1189/1261 [20:11<01:12,  1.01s/it][A
     94%|█████████▍| 1190/1261 [20:12<01:12,  1.02s/it][A
     94%|█████████▍| 1191/1261 [20:13<01:11,  1.02s/it][A
     95%|█████████▍| 1192/1261 [20:14<01:10,  1.02s/it][A
     95%|█████████▍| 1193/1261 [20:15<01:09,  1.02s/it][A
     95%|█████████▍| 1194/1261 [20:16<01:07,  1.01s/it][A
     95%|█████████▍| 1195/1261 [20:18<01:07,  1.02s/it][A
     95%|█████████▍| 1196/1261 [20:19<01:06,  1.02s/it][A
     95%|█████████▍| 1197/1261 [20:20<01:05,  1.02s/it][A
     95%|█████████▌| 1198/1261 [20:21<01:04,  1.02s/it][A
     95%|█████████▌| 1199/1261 [20:22<01:02,  1.01s/it][A
     95%|█████████▌| 1200/1261 [20:23<01:01,  1.01s/it][A
     95%|█████████▌| 1201/1261 [20:24<01:00,  1.01s/it][A
     95%|█████████▌| 1202/1261 [20:25<00:59,  1.00s/it][A
     95%|█████████▌| 1203/1261 [20:26<00:58,  1.02s/it][A
     95%|█████████▌| 1204/1261 [20:27<00:57,  1.00s/it][A
     96%|█████████▌| 1205/1261 [20:28<00:55,  1.00it/s][A
     96%|█████████▌| 1206/1261 [20:29<00:55,  1.01s/it][A
     96%|█████████▌| 1207/1261 [20:30<00:55,  1.03s/it][A
     96%|█████████▌| 1208/1261 [20:31<00:55,  1.04s/it][A
     96%|█████████▌| 1209/1261 [20:32<00:53,  1.03s/it][A
     96%|█████████▌| 1210/1261 [20:33<00:51,  1.01s/it][A
     96%|█████████▌| 1211/1261 [20:34<00:50,  1.01s/it][A
     96%|█████████▌| 1212/1261 [20:35<00:48,  1.00it/s][A
     96%|█████████▌| 1213/1261 [20:36<00:48,  1.00s/it][A
     96%|█████████▋| 1214/1261 [20:37<00:47,  1.01s/it][A
     96%|█████████▋| 1215/1261 [20:38<00:46,  1.02s/it][A
     96%|█████████▋| 1216/1261 [20:39<00:46,  1.03s/it][A
     97%|█████████▋| 1217/1261 [20:40<00:44,  1.02s/it][A
     97%|█████████▋| 1218/1261 [20:41<00:44,  1.03s/it][A
     97%|█████████▋| 1219/1261 [20:42<00:43,  1.04s/it][A
     97%|█████████▋| 1220/1261 [20:43<00:42,  1.05s/it][A
     97%|█████████▋| 1221/1261 [20:44<00:42,  1.05s/it][A
     97%|█████████▋| 1222/1261 [20:45<00:40,  1.04s/it][A
     97%|█████████▋| 1223/1261 [20:46<00:38,  1.02s/it][A
     97%|█████████▋| 1224/1261 [20:47<00:37,  1.00s/it][A
     97%|█████████▋| 1225/1261 [20:48<00:35,  1.00it/s][A
     97%|█████████▋| 1226/1261 [20:49<00:34,  1.00it/s][A
     97%|█████████▋| 1227/1261 [20:50<00:33,  1.01it/s][A
     97%|█████████▋| 1228/1261 [20:51<00:32,  1.01it/s][A
     97%|█████████▋| 1229/1261 [20:52<00:31,  1.01it/s][A
     98%|█████████▊| 1230/1261 [20:53<00:30,  1.01it/s][A
     98%|█████████▊| 1231/1261 [20:54<00:29,  1.01it/s][A
     98%|█████████▊| 1232/1261 [20:55<00:28,  1.01it/s][A
     98%|█████████▊| 1233/1261 [20:56<00:27,  1.00it/s][A
     98%|█████████▊| 1234/1261 [20:57<00:27,  1.00s/it][A
     98%|█████████▊| 1235/1261 [20:58<00:26,  1.01s/it][A
     98%|█████████▊| 1236/1261 [20:59<00:25,  1.01s/it][A
     98%|█████████▊| 1237/1261 [21:00<00:24,  1.01s/it][A
     98%|█████████▊| 1238/1261 [21:01<00:23,  1.01s/it][A
     98%|█████████▊| 1239/1261 [21:02<00:22,  1.00s/it][A
     98%|█████████▊| 1240/1261 [21:03<00:21,  1.00s/it][A
     98%|█████████▊| 1241/1261 [21:04<00:19,  1.01it/s][A
     98%|█████████▊| 1242/1261 [21:05<00:19,  1.00s/it][A
     99%|█████████▊| 1243/1261 [21:06<00:18,  1.01s/it][A
     99%|█████████▊| 1244/1261 [21:07<00:17,  1.02s/it][A
     99%|█████████▊| 1245/1261 [21:08<00:16,  1.04s/it][A
     99%|█████████▉| 1246/1261 [21:09<00:15,  1.06s/it][A
     99%|█████████▉| 1247/1261 [21:10<00:14,  1.04s/it][A
     99%|█████████▉| 1248/1261 [21:11<00:13,  1.02s/it][A
     99%|█████████▉| 1249/1261 [21:12<00:12,  1.03s/it][A
     99%|█████████▉| 1250/1261 [21:13<00:11,  1.03s/it][A
     99%|█████████▉| 1251/1261 [21:14<00:10,  1.06s/it][A
     99%|█████████▉| 1252/1261 [21:15<00:09,  1.05s/it][A
     99%|█████████▉| 1253/1261 [21:16<00:08,  1.03s/it][A
     99%|█████████▉| 1254/1261 [21:17<00:07,  1.02s/it][A
    100%|█████████▉| 1255/1261 [21:18<00:06,  1.00s/it][A
    100%|█████████▉| 1256/1261 [21:19<00:05,  1.01s/it][A
    100%|█████████▉| 1257/1261 [21:20<00:04,  1.01s/it][A
    100%|█████████▉| 1258/1261 [21:21<00:03,  1.02s/it][A
    100%|█████████▉| 1259/1261 [21:22<00:02,  1.01s/it][A
    100%|█████████▉| 1260/1261 [21:23<00:01,  1.02s/it][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: vehiclesDetectedVideo.mp4 
    
    CPU times: user 2h 23min 54s, sys: 3min 27s, total: 2h 27min 22s
    Wall time: 21min 24s


# Rubric 8: Discussion
### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

This project requires lots of parameter tuning, e.g. what features to use (HOG, spatial intensities and colour channel intensities), colour channels, different parameters for searching using sliding windows, scales (e.g. 1x, 1.5x, 2x etc) of detection, heat map threshold and so on. Fine tuning these takes time.

This pipeline still detects some false positives.

Also, I don't think this pipeline is ready to drive cars on roads just yet. It is very slow. Either we need to improve this algorithm, and/or arrange high computing power to make real time predictions.

This pipeline may fail in case of very crowded road, e.g. if the car is running on a busy road of NYC. Also, lighting conditions (night, noon or morning) and weather conditions (snow or rain) can make it difficult to detect colour features and HOG features from the image.

To imrpove this pipeline's accuracy and make it robust, I would test it on multiple videos of different driving conditions, e.g. if there are pedestrians, rain, steep slope of the road, sharp turns etc. Then analyse results and engineer better features.

This video has frames captured from front camera only, to further improve this pipeline, I would mount cameras on both sides and back of the car and collect images from them also. This will give a better estimate of cars coming from behind.
