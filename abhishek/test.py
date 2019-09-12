import ConfigParser

import cv2
import numpy
import skimage.feature as sk_feature
import skimage.transform as sk_transform
from sklearn.metrics.pairwise import cosine_similarity

config_file = open('/media/adv/Shared/PROJECTS/CSE515_MWDB/Code/variables.cfg')
config = ConfigParser.RawConfigParser(allow_no_value=True)
config.readfp(config_file)

PROJECT_ROOT_DIR = config.get('PATH', 'project_root_dir')
HAND_IMAGE_DATASET_DIR = config.get('PATH', 'hand_image_dataset_dir')
OUTPUT_DIR = config.get('PATH', 'output_dir')

# Open Image and Pre-process
sample_image = HAND_IMAGE_DATASET_DIR + '/' + 'Hand_0000002' + '.jpg'
sample_image2 = HAND_IMAGE_DATASET_DIR + '/' + 'Hand_0000080' + '.jpg'
# ndarray indexing starting from top left; Image representation - Width x Height x [R, G, B];
src_image_rgb = cv2.imread(sample_image)
src_image_rgb2 = cv2.imread(sample_image2)

# Scale image 1:10
scaled_img = sk_transform.rescale(src_image_rgb, 0.1, anti_aliasing=True)  # Anti-aliasing applies gaussian filter
scaled_img2 = sk_transform.rescale(src_image_rgb2, 0.1, anti_aliasing=True)

hog_feature_vector, hog_image = sk_feature.hog(scaled_img, orientations=9, pixels_per_cell=(8, 8),
                                               cells_per_block=(2, 2), block_norm='L2-Hys',
                                               visualize=True, feature_vector=True, multichannel=True)
hog_feature_vector2, hog_image2 = sk_feature.hog(scaled_img2, orientations=9, pixels_per_cell=(8, 8),
                                                 cells_per_block=(2, 2), block_norm='L2-Hys',
                                                 visualize=True, feature_vector=True, multichannel=True)

# numpy.savetxt(OUTPUT_DIR + '/' + 'hog_feature_descriptor.txt', hog_feature_vector, fmt='%f')
print "No of elements in HOG feature vector", hog_feature_vector.size
print "No of non-zero elements in HOG features: ", len([x for x in hog_feature_vector.tolist() if x != 0])

# cv2.imshow('RGB Image', scaled_img)
# cv2.imshow('hog', hog_image)
# cv2.imshow('hog2', hog_image2)

with open(OUTPUT_DIR + '/' + 'hog_cosine_value.txt', 'w+') as f:
    f.write(str(cosine_similarity(hog_feature_vector, hog_feature_vector2)))
    
print cosine_similarity(hog_feature_vector, hog_feature_vector2)
# SIFT
sift = cv2.xfeatures2d.SIFT_create()
image_grey = cv2.cvtColor(src_image_rgb, cv2.COLOR_BGR2GRAY)
image_grey = cv2.resize(image_grey, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
key_points, sift_feature_descriptor = sift.detectAndCompute(image_grey, None)

# Shape of sift_feature_descriptor = Number_of_Keypoints * 128
numpy.savetxt(OUTPUT_DIR + '/' + 'sift_feature_descriptor.txt', sift_feature_descriptor, fmt='%f')
img = cv2.drawKeypoints(image_grey, key_points, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('sift', img)


cv2.waitKey(0)
cv2.destroyAllWindows()
