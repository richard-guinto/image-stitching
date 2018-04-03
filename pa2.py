#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:07:46 2018

@author: Richard Guinto
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

fnameleft = 'imgpairs/tower_left.jpg'
fnameright = 'imgpairs/tower_right.jpg'


## load two images to be stitched (left and right)
image1 = cv2.imread(fnameleft)
image2 = cv2.imread(fnameright)

## convert to gray scale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

## convert to floating point
grayfloat1 = np.float32(gray1)
grayfloat2 = np.float32(gray2)

plt.imshow(cv2.cvtColor(np.hstack([image1,image2]),cv2.COLOR_BGR2RGB))
plt.title('Input Image (Left and Right)')
plt.show()

## get feature points based on Harris Detector
blocksize = 2
ksize = 3
k = 0.04

dst1 = cv2.cornerHarris(grayfloat1, blocksize, ksize, k)
dst2 = cv2.cornerHarris(grayfloat2, blocksize, ksize, k)

dstnorm1 = cv2.normalize(dst1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)
dstnorm2 = cv2.normalize(dst2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)

plt.imshow(np.hstack([dstnorm1,dstnorm2]), cmap='gray')
plt.title('Harris Corner - Normalized')
plt.show()

threshold = 130

## filter the feature points based on some threshold
keyPoint1 = []
for i in range(dstnorm1.shape[0]):
    for j in range(dstnorm1.shape[1]):
        if dstnorm1[i,j] > threshold:
            keyPoint1.append(cv2.KeyPoint(j,i,10))

keyPoint2 = []
for i in range(dstnorm2.shape[0]):
    for j in range(dstnorm2.shape[1]):
        if dstnorm2[i,j] > threshold:
            keyPoint2.append(cv2.KeyPoint(j,i,10))
            
print('Keypoint Counts: ', len(keyPoint1), len(keyPoint2))

## Extract fixed size patches around every keypoint of both images
patch_size = np.uint(40)
des1 = np.zeros((len(keyPoint1),64),np.uint8)
row = 0

for keyPoint in keyPoint1:
    x1 = np.uint(keyPoint.pt[0] - patch_size/2)
    y1 = np.uint(keyPoint.pt[1] - patch_size/2)
    patch = gray1[y1:y1+patch_size,x1:x1+patch_size]
    patch8x8 = cv2.resize(patch, (8,8))
    patchvector = patch8x8.flatten()
    des1[row] = patchvector.copy()
    row += 1


des2 = np.zeros((len(keyPoint2),64),np.uint8)
row = 0

for keyPoint in keyPoint2:
    x2 = np.uint(keyPoint.pt[0] - patch_size/2)
    y2 = np.uint(keyPoint.pt[1] - patch_size/2)
    patch = gray2[y2:y2+patch_size,x2:x2+patch_size]
    patch8x8 = cv2.resize(patch, (8,8))
    patchvector = patch8x8.flatten()
    des2[row] = patchvector.copy()
    row += 1
    
## select putative matches based on the matrix of pairwise descriptor distances 
bf = cv2.BFMatcher(cv2.NORM_L2, False)
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches_sorted = sorted(matches, key = lambda x:x.distance)

MAX_POINTS = 30

## Draw the Putative matches between keypoints in left and right image
img1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2RGB)
img2 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2RGB)

img3 = cv2.drawKeypoints(gray1, keyPoint1, None, (255,0,0))
img4 = cv2.drawKeypoints(gray2, keyPoint2, None, (255,0,0))
img5 = np.hstack([img3,img4])

plt.title('Feature Points')
plt.imshow(img5),plt.show()

good_matches = matches_sorted[:MAX_POINTS]
srcpoints = np.float32([ keyPoint1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
destpoints = np.float32([ keyPoint2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)

def displayEuclideanDistance(src, dest, transform, mask):
    count = 0
    destH = cv2.perspectiveTransform(dest, transform)
    ## display the Euclidean distance between the transformed right keypoint and orignal left keypoint
    print('InlierID Distance')
    for i in range(mask.shape[0]):
        if mask[i] == 1:
            dist = (destH[i,0,0] - src[i,0,0])**2 + (destH[i,0,1] - src[i,0,1])**2
            count += 1
            print(i, dist)
    print('TOTAL INLIERS: ', count)        
    return

## get Affine Transformation between left and right image keypoints
rightA = cv2.convertPointsToHomogeneous(destpoints)
leftA = cv2.convertPointsToHomogeneous(srcpoints)
retval, A,inliersA = cv2.estimateAffine3D(rightA, leftA)

## merge 3rd and 4th column of the 3x4 Affine matrix to make a 3x3 matrix for the Euclidean distance
## this can be done because z-values in rightA and leftA are technically constant value of 1
A3x3 = A[:,0:3].copy()
A3x3[:,2] = A3x3[:,2] + A[:,3]
print('Affine Matrix: ', A3x3)

img6 = cv2.drawMatches(img1,keyPoint1,img2,keyPoint2,good_matches, inliersA, flags=2)
plt.title('Affine Transformation - Inlier Locations')
plt.imshow(img6),plt.show()

displayEuclideanDistance(srcpoints, destpoints, A3x3, inliersA)
# Obtain the homography matrix
H, mask = cv2.findHomography(destpoints, srcpoints, method=cv2.RANSAC, ransacReprojThreshold=3.0)
print('Homography Matrix: ', H)
displayEuclideanDistance(srcpoints, destpoints, H, mask)

img7 = cv2.drawMatches(img1,keyPoint1,img2,keyPoint2,good_matches, mask, flags=2)
plt.title('Homography Transformation - Inlier Locations')
plt.imshow(img7),plt.show()

for match in matches_sorted:
    print(match.trainIdx, match.queryIdx, match.distance, keyPoint1[match.queryIdx].pt, keyPoint2[match.trainIdx].pt)
    
## Show first 5 sample patches based on descriptor distance
start = 0
samples = 5

for i in range(start,start+samples):
    #if mask[i] == 1:
        match = good_matches[i]
        idx = match.queryIdx
        (x,y) = keyPoint1[idx].pt
        x = np.uint(keyPoint1[idx].pt[0] - patch_size/2)
        y = np.uint(keyPoint1[idx].pt[1] - patch_size/2)
        patch = gray1[y:y+patch_size,x:x+patch_size]
        plt.subplot(samples,4,(i-start)*4+1)
        plt.imshow(patch, cmap='gray')
        patch8x8 = cv2.resize(patch, (8,8))
        plt.subplot(samples,4,(i-start)*4+3)
        plt.imshow(patch8x8, cmap='gray')

        idx = match.trainIdx
        (x,y) = keyPoint2[idx].pt
        x = np.uint(keyPoint2[idx].pt[0] - patch_size/2)
        y = np.uint(keyPoint2[idx].pt[1] - patch_size/2)
        patch = gray2[y:y+patch_size,x:x+patch_size]
        plt.subplot(samples,4,(i-start)*4+2)
        plt.imshow(patch, cmap='gray')
        patch8x8 = cv2.resize(patch, (8,8))
        plt.subplot(samples,4,(i-start)*4+4)
        plt.imshow(patch8x8, cmap='gray')
#plt.figtitle('Sample Patches and Descriptors from Keypoints (Left & Right)')
plt.show()

## get the vertical boundary of the right image inlier points
inliers = destpoints[mask == 1]
bound = inliers[:,0].max()

## convert image 1 and 2 format from BGR to RGB
imgrgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
imgrgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

## Warp the two images using Affine matrix
img8 = cv2.warpPerspective(imgrgb2, A3x3, (np.uint(imgrgb2.shape[1] + imgrgb1.shape[1] - bound), 
        imgrgb2.shape[0]), cv2.INTER_CUBIC)
img9 = np.hstack([imgrgb1,img8[:,imgrgb1.shape[1]:]])
plt.title('Stitched images using Affine Transformation')
plt.imshow(img9),plt.show()

## Warp two images using Homography
img10 = cv2.warpPerspective(imgrgb2, H, (np.uint(imgrgb2.shape[1] + imgrgb1.shape[1] - bound), 
        imgrgb2.shape[0]), cv2.INTER_CUBIC)
img11 = np.hstack([imgrgb1,img10[:,imgrgb1.shape[1]:]])
plt.title('Stitched images using Homography Transformation')
plt.imshow(img11),plt.show()
