from skimage.feature import hog
from skimage import exposure
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity

def checkSimilarity(filename):
    # Load the images
    image1 = cv2.imread('static/assets/img/colonca31.jpeg')
    image2 = cv2.imread(filename)
    image3 = cv2.imread('static/assets/img/colonn31.jpeg')


    # Resize images to 768x768
    image1 = cv2.resize(image1, (512, 512))
    image2 = cv2.resize(image2, (512, 512))
    image3 = cv2.resize(image3, (512, 512))

    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray_image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

    # Compute HOG descriptors
    hog_features1, hog_image1 = hog(gray_image1, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, block_norm='L2-Hys')
    hog_features2, hog_image2 = hog(gray_image2, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, block_norm='L2-Hys')
    hog_features3, hog_image3 = hog(gray_image3, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, block_norm='L2-Hys')

    # Rescale histogram for better visualization
    hog_image_rescaled1 = exposure.rescale_intensity(hog_image1, in_range=(0, 10))
    hog_image_rescaled2 = exposure.rescale_intensity(hog_image2, in_range=(0, 10))
    hog_image_rescaled3 = exposure.rescale_intensity(hog_image3, in_range=(0, 10))

    # Compare HOG descriptors using cosine similarity
    similarity1 = cosine_similarity(hog_features1.reshape(1, -1), hog_features2.reshape(1, -1))[0][0]
    similarity2 = cosine_similarity(hog_features3.reshape(1, -1), hog_features2.reshape(1, -1))[0][0]
    print("Similarity with Malignant Case : ", similarity1, "\n")
    print("Similarity with Normal Case : ", similarity2)
    if similarity1>=0.87 or similarity2>=0.87:
        return True
    return False