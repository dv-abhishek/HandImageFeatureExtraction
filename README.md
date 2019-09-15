# Extraction of hand image features
Features of hand images are extracted using HoG and SIFT feature extraction model. 'k' similar images to the given image are returned.

<br/>

## Image Dataset
https://sites.google.com/view/11khands

<br/>

## Tools used
- MongoDB - NoSQL database
- Python - 2.7.12

### Python Libraries
- skimage - Transformation and HoG feature extraction of images \n
- matplotlib - Plot matching images
- ConfigParser - Read config files
- cv2 - OpenCV library for image transformation and SIFT feature extraction
- numpy - Image and feature representation as an N-dim array
- pymongo - MongoDB API for python
<br/>

## Image Feature Extraction Models
- Histogram of Gradients (HoG)
- Scale Invariant Feature Transformation (SIFT)

### HoG Parameters
- Number of orientation bins = 9
- Cell size (pixels per cell) = 8 * 8
- Block size (cells per block) = 2 * 2
- L2-norm clipping threshold = 0.2
- Image Scale (Downscale) = 10:1
- Similarity function - Cosine Similarity

### SIFT Parameters
- Brute-Force SIFT descriptor matching
- L2-norm distance function for SIFT descriptor matching
- Similarity function - 1 / sum(matching_distances)
