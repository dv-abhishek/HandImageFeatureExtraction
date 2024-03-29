import ConfigParser
import os
from collections import OrderedDict
from scipy import misc

import cv2
import numpy
import skimage.feature as sk_feature
import skimage.transform as sk_transform
from pymongo import MongoClient

from functions import cosine_similarity, euclidean_distance, sift_similarity_function

config = ConfigParser.RawConfigParser(allow_no_value=True)
with open('/media/adv/Shared/PROJECTS/CSE515_MWDB/Code/variables.cfg') as config_file:
    config.readfp(config_file)

PROJECT_ROOT_DIR = config.get('PATH', 'project_root_dir')
HAND_IMAGE_DATASET_DIR = config.get('PATH', 'hand_image_dataset_dir')
OUTPUT_DIR = config.get('PATH', 'output_dir')

db_client = MongoClient(host=config.get('db', 'host'),
                        port=int(config.get('db', 'port')))
db = db_client[HAND_IMAGE_DATASET_DIR.split('/', 5)[-1].replace('/', '#')]


def hog_feature_extraction(image_file_name):
    image_file_path = HAND_IMAGE_DATASET_DIR + os.sep + image_file_name
    hog_feature_vector = None
    found_in_db = False
    # If already present in the DB, then read and populate it
    if config.get('config_key', 'read_from_db') == "True":
        query_output = db.hog.find({"image": image_file_name.replace(".jpg", "")})
        if query_output.count() > 0:
            hog_feature_vector = numpy.asarray([row["vector"] for row in query_output])
            found_in_db = True
            print "Found HOG vector for image " + image_file_name
    if not found_in_db:
        src_image = misc.imread(image_file_path)
        scaled_image = sk_transform.rescale(src_image, 0.1, anti_aliasing=True)  # Anti-aliasing applies gaussian filter
        hog_feature_vector, hog_image = sk_feature.hog(scaled_image, orientations=9, pixels_per_cell=(8, 8),
                                                       cells_per_block=(2, 2), block_norm='L2-Hys',
                                                       visualize=True, feature_vector=True, multichannel=True)
    
    # Insert into DB
    if config.get('config_key', 'write_to_db') == "True" and not found_in_db:
        output = db.hog.update_one(
            {"image": image_file_name.replace(".jpg", "")},
            {"$set": {"vector": hog_feature_vector.flatten().tolist()}},
            upsert=True
        )
        if not output.acknowledged:
            print "ERROR: Could not add/update HOG vector for image " + image_file_name
        if output.upserted_id:
            print "Inserted HOG vector for image " + image_file_name
        else:
            print "Updated HOG vector for image " + image_file_name
    
    if config.get('config_key', 'write_to_file') == "True":
        numpy.savetxt(OUTPUT_DIR + os.sep + 'hog_feature_descriptor_' + image_file_name + '.txt',
                      hog_feature_vector, fmt='%f')
    return hog_feature_vector


def sift_feature_extraction(image_file_name):
    image_file_path = HAND_IMAGE_DATASET_DIR + os.sep + image_file_name
    sift_feature_descriptor = None
    found_in_db = False
    if config.get('config_key', 'read_from_db') == "True":
        query_output = db.sift.find({"image": image_file_name.replace(".jpg", "")})
        if query_output.count() > 0:
            sift_feature_descriptor = numpy.asarray([row["vector"] for row in query_output])
            found_in_db = True
            print "Found HOG vector for image " + image_file_name

    if not found_in_db:
        src_image = misc.imread(image_file_path)
        sift = cv2.xfeatures2d.SIFT_create()
        image_grey = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        key_points, sift_feature_descriptor = sift.detectAndCompute(image_grey, None)
    
    ''' Insert into DB
        Convert ndarray (n, 128) to list_of_list. Get back ndarray using
        numpy.asarray(list_of_list, dtype=numpy.float32)
    '''
    if config.get('config_key', 'write_to_db') == "True" and not found_in_db:
        output = db.sift.update_one(
            {"image": image_file_name.replace(".jpg", "")},
            {"$set": {"vector": [list(x) for x in sift_feature_descriptor]}},
            upsert=True
        )
        if not output.acknowledged:
            print "ERROR: Could not add/update HOG vector for image " + image_file_name
        if output.upserted_id:
            print "Inserted HOG vector for image " + image_file_name
        else:
            print "Updated HOG vector for image " + image_file_name
    
    if config.get('config_key', 'write_to_file') == "True":
        numpy.savetxt(OUTPUT_DIR + os.sep + 'sift_feature_descriptor_' + image_file_name + '.txt',
                      sift_feature_descriptor, fmt='%f')
    return sift_feature_descriptor


def extract_hog_features_for_all_images(src_image_file_name):
    similarity_scores = {}  # Store as Filename-Score pairs
    src_feature_vector = hog_feature_extraction(src_image_file_name)
    for image_file_name in os.listdir(HAND_IMAGE_DATASET_DIR):
        if image_file_name != src_image_file_name:
            print "Processing HOG for image ", image_file_name
            target_feature_vector = hog_feature_extraction(image_file_name)
            # Cosine similarity returns (1, 1) ndarray
            similarity_scores[image_file_name] = cosine_similarity(src_feature_vector, target_feature_vector)
    sorted_similarity_scores = OrderedDict(sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True))
    
    with open(OUTPUT_DIR + os.sep + 'hog_feature_similarity_' + src_image_file_name + '.txt', 'w+') as output_file:
        output_file.write(str(sorted_similarity_scores))
    return sorted_similarity_scores


def extract_sift_features_for_all_images(src_image_file_name):
    similarity_scores = {}  # Store as Filename-Score pairs
    src_feature_vector = sift_feature_extraction(src_image_file_name)
    for image_file_name in os.listdir(HAND_IMAGE_DATASET_DIR):
        if image_file_name != src_image_file_name:
            print "Processing HOG for image ", image_file_name
            target_feature_vector = sift_feature_extraction(image_file_name)
            similarity_scores[image_file_name] = find_sift_image_similarity(src_feature_vector, target_feature_vector)
    sorted_similarity_scores = OrderedDict(sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True))
    return sorted_similarity_scores


def find_sift_image_similarity(src_feature_vector, target_feature_vector):
    matches = []
    for vector in src_feature_vector:
        min_distance = min([euclidean_distance(vector, x) for x in target_feature_vector])
        matches.append(min_distance)
    return sift_similarity_function(matches)
