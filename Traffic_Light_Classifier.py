# ### Import resources
# 
# Import the libraries and resources that you'll need.

import cv2 # computer vision library
import helpers # helper functions
import test_functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

%matplotlib inline

# ## Define the image directories
# 
# First, we set some variables to keep track of some where our images are stored:
# 
#     IMAGE_DIR_TRAINING: the directory where our training image data is stored
#     IMAGE_DIR_TEST: the directory where our test image data is stored

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# ## Load the datasets
# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
IMAGE_LIST[0][:]

# ## Visualize the Data
# 
# ### Visualize the input images

image_num = 730
selected_image = IMAGE_LIST[image_num][0]
selected_label = IMAGE_LIST[image_num][1]

plt.imshow(selected_image)

print("The shape: ", str(selected_image.shape))
print("The Label: ", str(selected_label))


# # 2. Pre-process the Data

# ### Standardize the input images

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
      
    standard_im = cv2.resize(image,(32,32))
    
    return standard_im
    
# ## Standardize the output
# ### Implement one-hot encoding

# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):
    
    one_hot_encoded = [0, 0, 1]       # Green Case
    if(label == 'red'):
        one_hot_encoded = [1, 0, 0]   # Red Case
    elif(label == 'yellow'):
        one_hot_encoded = [0, 1, 0]   # Yellow Case
    
    return one_hot_encoded            # otherwise we return the default case which is Green 

# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)


# ## Construct a `STANDARDIZED_LIST` of input images and output labels.
# 
# This function takes in a list of image-label pairs and outputs a **standardized** list of resized images and one-hot encoded labels.

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)


# ## Visualize the standardized data

image_num = 0
std_image = STANDARDIZED_LIST[image_num][0]
std_label = STANDARDIZED_LIST[image_num][1]

plt.imshow(std_image)
print("The shape: ", str(std_image.shape))
print("The label [1, 0, 0] if red, [0, 1, 0] if yellow, [0, 0, 1] if green: ", str(std_label))


# # 3. Feature Extraction
# Convert and image to HSV colorspace
# Visualize the individual color channels

image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

# ### Create a brightness feature that uses HSV color space
# 
# A function that takes in an RGB image and returns a 1D feature vector and/or single value that will help classify an image of a traffic light. 
# 
# From this feature, we should be able to estimate an image's label and classify it as either a red, green, or yellow traffic light.

def extract_regions(v_channel):
    
    # identifing thre regions
    
    red = v_channel[4:12, 12:22]     # red region     
    yellow = v_channel[12:20, 12:22] # yellow region
    green = v_channel[20:28, 12:22]  # green region
    
    return red, yellow, green
    
## This feature should use HSV colorspace values
def create_feature(rgb_image):
    
    ## Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    ## Create and return a feature value and/or vector
    v = hsv[:,:,2]
    
    red, yellow, green = extract_regions(v)
    
    red_brightness = np.sum(red)
    yellow_brightness = np.sum(yellow)
    green_brightness = np.sum(green)
    
    feature = [red_brightness, yellow_brightness, green_brightness]
    
    return feature

image_num = 760
test_im = STANDARDIZED_LIST[image_num][0]

avg = create_feature(test_im)
print('Avg brightness: ' + str(avg))
plt.imshow(test_im)

# # 4. Classification and Visualizing Error
# 
# A function that takes in an RGB image and, using the extracted features, outputs whether a light is red, green or yellow as a one-hot encoded label. This classification function should be able to classify any image of a traffic light!

# ### Build a complete classifier 

# This function should take in RGB image input
# Analyze that image using the feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    
    ## Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    feature = create_feature(rgb_image)
    max_brightness = max(feature)
    
    predicted_label = [0, 1, 0]          # default value is set to yellow
    
    if(feature[0] == max_brightness):
        predicted_label = [1, 0, 0]      # red case  
    elif(feature[2] == max_brightness): 
        predicted_label = [0, 0, 1]      # green case
        
    return predicted_label   
    
# ## Testing the classifier

# A "good" classifier in this case should meet the following criteria:
# 1. Get above 90% classification accuracy.
# 2. Never classify a red light as a green light. 
# 
# ### Test dataset

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# ## Determine the Accuracy
# 
# Compare the output of classification algorithm with the true labels and determine the accuracy.
# 
# This code stores all the misclassified images, their predicted labels, and their true labels, in a list called `MISCLASSIFIED`. This code is used for testing.

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

# ### Visualize the misclassified images

image_num = 1  # change this between 0 and 1 to see both misclassified images
test_mis_im = MISCLASSIFIED[image_num][0]

plt.imshow(test_mis_im)
print(estimate_label(test_mis_im))

# Importing the tests
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")