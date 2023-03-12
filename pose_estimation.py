import cv2
import numpy as np
import matplotlib.pyplot as plt
from homography import computeH

def four_points(points):
    thresh = 0.2
    if len(points) == 4:
        real_points = points 
    else:
        for i in range(len(points)):
            for j in range(len(points)):
                if points[i] == points[j]:
                    continue
                else:
                    d = np.linalg.norm(np.array(points[i])-np.array(points[j]))
                    if d<thresh:
                        points.remove(points[j])
        if len(points) == 4:
            real_points = points
        else:
            print(len(points))
            return False
    three_point = np.zeros((len(real_points),3))
    two_point = np.array(real_points).reshape(len(real_points), 2)
    for i in range(three_point.shape[0]):
        three_point[i][0] = two_point[i][0] 
        three_point[i][1] = two_point[i][1]
        three_point[i][2] = 1
    
    return three_point

def decHomography(A, H):
    H = np.transpose(H)
    h1 = H[0]
    h2 = H[1]
    h3 = H[2]

    Ainv = np.linalg.inv(A)

    L = 1 / np.linalg.norm(np.dot(Ainv, h1))

    r1 = L * np.dot(Ainv, h1)
    r2 = L * np.dot(Ainv, h2)
    r3 = np.cross(r1, r2)

    T = L * np.dot(Ainv, h3)

    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))
    U, S, V = np.linalg.svd(R, full_matrices=True)

    U = np.matrix(U)
    V = np.matrix(V)
    R = U * V

    return (R, T)

def pose_estimation(points):
    K = np.array([[1.38E+03,	0,	9.46E+02],
                [0,	1.38E+03,	5.27E+02],
                [0,	0,	1]])
    paper_width = 0.216
    paper_height = 0.279

    paper_corners_3D = np.array([
    [0, 0, 0],
    [0, paper_height, 0],
    [paper_width, paper_height, 0],
    [paper_width, 0, 0]
    ], dtype=np.float32)

    two_D_points = four_points(points)
    H = computeH(paper_corners_3D, two_D_points)
    R, T = decHomography(K, H)
    print("Rotation: ", R)
    print("Translation:", T)
    pose = np.hstack((R,T.reshape(3,1)))
    return pose