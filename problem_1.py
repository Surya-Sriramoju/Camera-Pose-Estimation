import cv2
import numpy as np
import matplotlib.pyplot as plt
from hough_transform import hough_transform
from pose_estimation import pose_estimation


def canny(img):
    t_lower = 200
    t_upper = 255
    return cv2.Canny(img, t_lower, t_upper)

def main():
    i = 0
    final_points = {}
    alpha = 100 # Contrast control
    beta = 100 # Brightness control
    cap = cv2.VideoCapture('project2.avi')
    camera_trajectory = []
    initial_camera_pose = np.array([[0], [0], [0], [1]])
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5),0)
            temp, thresholded = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY)
            canny_edge = canny(thresholded)
            accumulator, thetas, rhos, canny_edge, points = hough_transform(canny_edge, frame)
                
            if points != False and len(points) == 4:
                pose = pose_estimation(points)
                camera_trajectory.append(pose @ initial_camera_pose)
            i += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for cam_pose in camera_trajectory:
        ax.scatter(cam_pose[0], cam_pose[1], cam_pose[2], color='r')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.title.set_text('Camera Pose Estimation')
    plt.show()

if __name__ == '__main__':
    main()