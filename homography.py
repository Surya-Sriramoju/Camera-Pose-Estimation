import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def get_Keypoints(img1, img2):
    img1_Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_Gray,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_Gray,None)

    FLANN_INDEX_KDTREE = 2
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    good_matches = []
    i = 0
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            matchesMask[i]=[1,0]
            good_matches.append(m)
            i += 1

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,keypoints_1,img2,keypoints_2,matches,None,**draw_params)
    
    source_points = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    destination_points = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    key_pts1 = np.zeros((source_points.shape[0],2))
    key_pts2 = np.zeros((destination_points.shape[0],2))

    for i in range(source_points.shape[0]):
      key_pts1[i] = source_points[i]
      key_pts2[i] = destination_points[i]

    return key_pts1, key_pts2

def three_point_form(points):
  points1 = np.zeros((points.shape[0],3))
  
  for i in range(points.shape[0]):
    points1[i][0] = points[i][0]
    points1[i][1] = points[i][1]
    points1[i][2] = 1
  return points1

def computeH(pts1, pts2):
  n = pts1.shape[0]
  A = np.zeros((2*n, 9))
  A = []
  for i in range(n):
    x = [pts1[i][0], pts1[i][1], 1, 0, 0, 0, (-pts2[i][0]*pts1[i][0]), (-pts2[i][0]*pts1[i][1]), -pts2[i][0]]
    A.append(x)
    x = [0, 0, 0, pts1[i][0], pts1[i][1], 1, (-pts2[i][1]*pts1[i][0]), (-pts2[i][1]*pts1[i][1]), -pts2[i][1]]
    A.append(x)
  A = np.array(A)
  A.reshape((2*n,9))

  U, S, V = np.linalg.svd(A)
  H = V[-1].reshape(3, 3)
  return H

def get_inliers(p1,p2,H):
  p2_estimate = np.dot(H, np.transpose(p1))
  p2_estimate = (1 / p2_estimate[2]) * p2_estimate
  return np.linalg.norm(np.transpose(p2) - p2_estimate)

def ransac(points1, points2, thresh):
  p1_norm =  three_point_form(points1)
  p2_norm = three_point_form(points2)
  num_iter = 10000
  num_inliers = 0

  best_H = np.zeros((3,3))
  
  for iter in range(num_iter):
    
    inliers = 0
    p1_p = []
    p2_p = []
    random_points = random.sample(range(0, p1_norm.shape[0]), 4)
    for i in random_points:
      p1_p.append(p1_norm[i])
      p2_p.append(p2_norm[i])
    p1_p = np.array(p1_p)
    p2_p = np.array(p2_p)

    H = computeH(p1_p, p2_p)
    for i in range(p1_norm.shape[0]):
      d = get_inliers(p1_norm[i],p2_norm[i],H)
      if d<thresh:
        inliers += 1
    if inliers > num_inliers:
      num_inliers = inliers
      best_H = H
  return best_H

def get_stitched_image(images):
  temp = images[0]
  height = max([img.shape[0] for img in images])
  width = sum([img.shape[1] for img in images])
  # blank = np.uint8(np.zeros((2*height,width,3)))
  blank = np.uint8(np.zeros((height,width,3)))
  blank[0:images[0].shape[0],0:images[0].shape[1]] = images[0]

  for i in range(len(images)-1):
    key1, key2 = get_Keypoints(temp,images[i+1])
    H = ransac(key2, key1, 0.3)
    # result = cv2.warpPerspective(images[i+1], H, (width, 2*height))
    result = cv2.warpPerspective(images[i+1], H, (width, height))
    blank[blank==0] = result[blank == 0]
  non_zero_pixels = np.where(blank!=0)
  final = blank[np.min(non_zero_pixels[0]):np.max(non_zero_pixels[0]), np.min(non_zero_pixels[1]):np.max(non_zero_pixels[1]), :]
  return final

