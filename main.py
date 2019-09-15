import ConfigParser
import os
from scipy import misc

import matplotlib.pyplot as plt
import skimage.transform as sk_transform

from abhishek import *

config = ConfigParser.RawConfigParser(allow_no_value=True)
with open('/media/adv/Shared/PROJECTS/CSE515_MWDB/Code/variables.cfg') as config_file:
    config.readfp(config_file)

HAND_IMAGE_DATASET_DIR = config.get('PATH', 'hand_image_dataset_dir')

if __name__ == '__main__':
    k = input("Enter value of k: ")
    model = input("Emter the model - 1. HOG \t 2. SIFT\t")
    file_name = raw_input("Select one of the images \n" + str(os.listdir(HAND_IMAGE_DATASET_DIR)) + "\n:\t")
    
    sorted_similarity_values = {}
    if model == 1:
        sorted_similarity_values = extract_hog_features_for_all_images(file_name)
    elif model == 2:
        sorted_similarity_values = extract_sift_features_for_all_images(file_name)
    else:
        exit(1)
    
    if config.get('config_key', 'visualize') == "True":
        src_image = sk_transform.rescale(misc.imread(HAND_IMAGE_DATASET_DIR + os.sep + file_name), 0.5,
                                         anti_aliasing=True)
        plt.imshow(misc.imread(HAND_IMAGE_DATASET_DIR + os.sep + file_name))
        plt.title("Source Image: " + file_name)
        plt.show()
    similarity_values_iter = iter(sorted_similarity_values)
    for i in range(k):
        image_file = similarity_values_iter.next()
        similarity_value = sorted_similarity_values[image_file]
        similar_image = sk_transform.rescale(misc.imread(HAND_IMAGE_DATASET_DIR + os.sep + image_file), 0.5,
                                             anti_aliasing=True)
        print similarity_value, image_file
        if config.get('config_key', 'visualize') == "True":
            plt.imshow(similar_image)
            plt.title(
                "Similar Image " + str(i + 1) + ": " + image_file + "  Similarity Score = " + str(similarity_value))
            plt.show()
