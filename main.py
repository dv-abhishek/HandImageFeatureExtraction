from abhishek import *

file_name = 'Hand_0000569.jpg'

if __name__ == '__main__':
    k = input("Enter value of k: ")
    model = input("Emter the model - 1. HOG \t 2. SIFT")
    if model == 1:
        extract_hog_features_for_all_images(file_name)
    elif model == 2:
        extract_sift_features_for_all_images(file_name)
