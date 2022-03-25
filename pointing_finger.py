
# importing relevant libraries

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import sys
# from sympy import Point3D, Line3D

# defining interpretor to initiate the pretrained model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# initiating intel realsense connecting to a pipeline
pipeline = rs.pipeline()
config = rs.config()
# defining resolution

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Drawing function to connect detected points of an elbow and hand
EDGES = {
    (0, 2): 'm',
}

# function to draw keypoints of a hand
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    # mutiplying detected points to camera resolution.
    shaped = np.squeeze(np.multiply(keypoints_with_scores[0][0][7], [y, x, 1]))
    shaped1 = np.squeeze(np.multiply(keypoints_with_scores[0][0][9], [y, x, 1]))
    ky, kx, kp_conf = shaped
    ky1, kx1, kc_conf = shaped1
    # if greater than confidence threshold draw a point for elbow
    if kp_conf > confidence_threshold:
        cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
    # if greater than confidence threshold draw a point for hand
    if kc_conf > confidence_threshold:
        cv2.circle(frame, (int(kx1), int(ky1)), 4, (0, 255, 0), -1)

# function to draw connections of a hand
def draw_connections(frame, keypoints, edges, confidence_threshold):
            y, x, c = frame.shape
            # mutiplying detected points to camera resolution.
            shaped = np.squeeze(np.multiply(keypoints_with_scores[0][0][7:11], [y, x, 1]))
            # identifying points and connecting them through a line
            for edge, color in edges.items():
                p1, p2 = edge
                y1, x1, c1 = shaped[p1]
                y2, x2, c2 = shaped[p2]
                # greater than threshold draw a line
                if (c1 > confidence_threshold) :
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


# function to colour up the detected aruco tags.
def arucoAug(bbox, id, img, imgAug):

    # identifying corners of deteceted aruco marker
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    # height, width of the image to be used to cover the aruco tag.
    h, w, c = imgAug.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # finding homographies from the image
    matrix, _ = cv2.findHomography(pts2, pts1)
    # To blacken out the surroundings
    imgout = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    # for transparency of surroundings
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgout1 = img + imgout
    # result output a frame with detected red aruco markers

    return imgout1

# function to identify intersection of a pointing line( connecting a hand and elbow) and aruco tag surface.

def collinear(centerX, centerY,zDepth_plane, pts1, data):

    # finding 2 vectors on a plane of aruco tag surface. ( ux- x, uy -y , ux - depth)
    ux, uy, uz = [pts1[0][0] - centerX, pts1[0][1] - centerY, zDepth_plane_tl - zDepth_plane]
    vx, vy, vz = [pts1[1][0] - centerX, pts1[1][1] - centerY, zDepth_plane_tr - zDepth_plane]
    # converting these Two 3D vectors into np array
    a = np.asarray([ux, uy, uz])
    b = np.asarray([vx, vy, vz])
    # calculating cross product
    n = np.cross(a, b)
    # center point the aruco tag surface
    V0 = np.asarray([centerX, centerY, zDepth_plane])
    # 3d points of elbow
    P1 = np.asarray([x1, y1, zDepth_1])
    # 3d points of hand
    P0 = np.asarray([x3, y3, zDepth_3])
    # w is vector one
    w = P0 - V0
    # u is vector 2
    u = P1 - P0
    # dot product of normal and vector 1
    N = np.dot(n, w)
    # dot product of normal and vector 2
    D = np.dot(n, u)
    # ratio to calcuate if the line passes the plane of the aruco surface
    sI = N / D
    # calculating that exact 3d point of intersection( b/w line and aruco plane)
    I = P0 + sI * u
    # print(I)

    # it intersects the plane only of the ration is between 0 and 1
    if 0 < abs(sI) < 1 :
        # if it crosses the plane
        # calculating the intersection points if the plane and 3d line intersects
        for i in range(0,len(pts1)):
            # tried to provide the limits for the area of the square but results were not good.
            # if pts1[0][0]  <I[0] < pts1[2][0] and pts1[0][1] < I[1] < pts1[2][1] :
                # if it there is an intersection the colour of cube turns black.
                data = arucoAug(corners, id, data, imgAug_2)
                # point of intersection for every frame.
                print(I)
                return data





# initiating the while loop

while True:
    # extracting frame from intelrealsense
    frames = pipeline.wait_for_frames()
    # depth frame
    depth_frame = frames.get_depth_frame()
    # colour frame
    color_frame = frames.get_color_frame()
    # converted to solvable data calculation
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    # storing the img matrix in data
    data = color_frame.get_data()
    # converting to array
    data = np.array(data)

    ##  DEFINING THE PRETRAINED MODEL FOR HAND DETECTION

    # resizing the image
    img = tf.image.resize_with_pad(np.expand_dims(data, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions from pretrained tensorflowhub model

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    # storing x,y and confidence values in an array for multiple points of human
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # calling connection functions
    draw_connections(data, keypoints_with_scores, EDGES, 0.3)
    draw_keypoints(data, keypoints_with_scores, 0.3)
    y, x, c = data.shape
    shaped = np.squeeze(np.multiply(keypoints_with_scores[0][0][7:11], [y, x, 1]))
    # storing these values for future use.
    # naming points from x,y and z
    y1, x1 = shaped[0][0:2] # left elbow
    #     y2,x2 = shaped[1][0:2] # right elbow
    y3,x3 = shaped[2][0:2] # left wrist
    #     y4,x4 = shaped[2][0:2] # right wrist
    # depth values of left elbow points
    zDepth_1 = depth_frame.get_distance(int(x1), int(y1))
    #     zDepth_2 = depth_frame.get_distance(int(x2),int(y2))
    # depth values for left wrist points
    zDepth_3 = depth_frame.get_distance(int(x3),int(y3))
    #     zDepth_4 = depth_frame.get_distance(int(x4),int(y4))

    # Start the detection of aruco markers
    # accessing test.yaml for intrinsic parameters
    cv_file = cv2.FileStorage("test.yaml", cv2.FILE_STORAGE_READ)

    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_matrix = cv_file.getNode("dist_coeff").mat()

    # to detect aruco markers
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    # to identify corners and id's
    corners, ids, rejectedImgPoints = aruco.detectMarkers(data, aruco_dict, parameters = parameters)

    # defining the red images to cover up the surface
    imgAug_1 = cv2.imread("red_img.jpg")
    imgAug_2 = cv2.imread("white_img.png")
    # converting corners for to array of square aruco surface
    corners= np.asarray(corners)
    # print(corners)

    # command to draw boundaries of detected aruco tags
    data = aruco.drawDetectedMarkers(data, corners, ids)

    # multiple aruco tags found
    if len(corners) != 0:
        for corners,id in zip( corners, ids):
            # storing values of each corner( tl: top left, tr: top right, br: bottom right, bl: bottom left)
            tl = corners[0][0][0], corners[0][0][1]
            tr = corners[0][1][0], corners[0][1][1]
            br = corners[0][2][0], corners[0][2][1]
            bl = corners[0][3][0], corners[0][3][1]
            data = arucoAug(corners, id, data, imgAug_1)
            # all the points in pts1
            pts1 = np.array([tl, tr, br, bl])
            # command to identify rotation and translation matrix for each points of the found aruco tag
            r_vec, t_vec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)

            # to calculate center of each aruco tag square.
            centerX = (corners[0][0][0] + corners[0][1][0] + corners[0][2][0] + corners[0][3][0]) / 4
            centerY = (corners[0][0][1] + corners[0][1][1] + corners[0][2][1] + corners[0][3][1]) / 4
            # converting to int because depth argument only takes int values
            center = (int(centerX), int(centerY))
            # calculating depths of each corner point of an aruco tag
            zDepth_plane = depth_frame.get_distance(int(centerX), int(centerY))
            zDepth_plane_tl = depth_frame.get_distance(int(pts1[0][0]), int(pts1[0][1]))
            zDepth_plane_tr = depth_frame.get_distance(int(pts1[1][0]), int(pts1[1][1]))
            zDepth_plane_br = depth_frame.get_distance(pts1[2][0], pts1[2][1])
            zDepth_plane_bl = depth_frame.get_distance(pts1[3][0], pts1[3][1])
            # calling function for finding point of intersection
            collinear(centerX, centerY, zDepth_plane, pts1, data)


            # ANOTHER METHOD TO CALCULATE INTERSECTION POINT

            # print(pts1[0])
            # # dis_x = np.linspace(pts1[0][0], pts1[2][0], 100)
            # # dis_y = np.linspace(pts1[0][1], pts1[2][1], 100)
            # for i,j in zip( dis_x,dis_y):
            #     x_new,y_new = [i,j]
            #     collinear(x1, y1, x3, y3, x_new, y_new,data)
            #     # print(x_new)


            # print(pts1)
            # print(dis_x)
            # for i,j,k in pts1[]
            # r_vec, t_vec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)
            # imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_matrix)

            # METHOD ENDS

            # To draw the axis of detected tags
            for rvec, tvec in zip(r_vec, t_vec): # using rot and trans matrixes
                data = aruco.drawAxis(data, camera_matrix, dist_matrix, rvec, tvec, 0.05)
                # To project 3d points to 2D surface
                # imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_matrix)

    # display the frame
    cv2.imshow('Mask',  data)

    if cv2.waitKey(25) == ord('q'):
        break

cv2.destroyAllWindows()
pipeline.stop()




