import ConfigParser
import os
import warnings
from collections import OrderedDict

import cv2

# To suppress DeprecationWarnings in the terminal
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    from sklearn.metrics.pairwise import cosine_similarity
    import skimage.feature as sk_feature
    import skimage.transform as sk_transform

config_file = open('/media/adv/Shared/PROJECTS/CSE515_MWDB/Code/variables.cfg')
config = ConfigParser.RawConfigParser(allow_no_value=True)
config.readfp(config_file)

PROJECT_ROOT_DIR = config.get('PATH', 'project_root_dir')
HAND_IMAGE_DATASET_DIR = config.get('PATH', 'hand_image_dataset_dir')
OUTPUT_DIR = config.get('PATH', 'output_dir')


def hog_feature_extraction(image_file_name):
    image_file_path = HAND_IMAGE_DATASET_DIR + os.sep + image_file_name
    src_image = cv2.imread(image_file_path)
    scaled_image = sk_transform.rescale(src_image, 0.1, anti_aliasing=True)  # Anti-aliasing applies gaussian filter
    hog_feature_vector, hog_image = sk_feature.hog(scaled_image, orientations=9, pixels_per_cell=(8, 8),
                                                   cells_per_block=(2, 2), block_norm='L2-Hys',
                                                   visualize=True, feature_vector=True, multichannel=True)
    
    # TODO:
    # Insert into DB
    
    # numpy.savetxt(OUTPUT_DIR + os.sep + 'hog_feature_descriptor_' + image_file_name + '.txt', hog_feature_vector,
    #               fmt='%f')
    return hog_feature_vector


def sift_feature_extraction(image_file_name):
    image_file_path = HAND_IMAGE_DATASET_DIR + os.sep + image_file_name
    src_image = cv2.imread(image_file_path)
    sift = cv2.xfeatures2d.SIFT_create()
    image_grey = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    key_points, sift_feature_descriptor = sift.detectAndCompute(image_grey, None)
    
    # TODO:
    # Insert into DB
    
    # numpy.savetxt(OUTPUT_DIR + os.sep + 'sift_feature_descriptor_' + image_file_name + '.txt', sift_feature_descriptor,
    #               fmt='%f')
    return sift_feature_descriptor


def extract_hog_features_for_all_images(src_image_file_name):
    similarity_scores = {}  # Store as Filename-Score pairs
    src_feature_vector = hog_feature_extraction(src_image_file_name)
    for image_file_name in os.listdir(HAND_IMAGE_DATASET_DIR):
        if image_file_name != src_image_file_name:
            print "Processing HOG for image ", image_file_name
            target_feature_vector = hog_feature_extraction(image_file_name)
            # Cosine similarity returns (1, 1) ndarray
            similarity_scores[image_file_name] = cosine_similarity(src_feature_vector, target_feature_vector)[0, 0]
    sorted_similarity_scores = OrderedDict(sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True))
    with open(OUTPUT_DIR + os.sep + 'hog_feature_similarity_' + src_image_file_name + '.txt', 'w+') as output_file:
        output_file.write(str(sorted_similarity_scores))
    return sorted_similarity_scores


def extract_sift_features_for_all_images():
    pass
