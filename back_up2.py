import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import sys

# defining interpretor to initiate the model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# initiating intel realsense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Drawing functions
EDGES = {
    (0, 2): 'm',
    (1, 3): 'c',
}

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints_with_scores[0][0][7:11], [y, x, 1]))
    #     y1,x1 = shaped[0][0:2]
    #     y2,x2 = shaped[1][0:2]
    #     y3,x3 = shaped[2][0:2]
    #     y4,x4 = shaped[2][0:2]
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
            y, x, c = frame.shape
            shaped = np.squeeze(np.multiply(keypoints_with_scores[0][0][7:11], [y, x, 1]))

            for edge, color in edges.items():
                p1, p2 = edge
                y1, x1, c1 = shaped[p1]
                y2, x2, c2 = shaped[p2]

                if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    # img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return
# imgAug  = cv2.imread("black_img.jpg")



#
# def augment_aruco( corners, id,data, imgAug):
#     tl = corners[0][0][0][0] , corners[0][0][1]
#     tr = corners[0][1][0], corners[0][1][1]
#     br = corners[0][2][0], corners[0][2][1]
#     bl = corners[0][3][0], corners[0][3][1]
#
#     h,w,c = imgAug.shape
#
#     pts1= np.array( [ tl,tr,br,bl])
#     pts2 = np.float([[ 0,0],[w,0],[w,h],[0,h]])
#     matrix,_ = cv2.findHomography(pts2,pts1)
#     imgout = cv2.warpPerspective(imgAug, matrix, (data.shape[1], data.shape[0]))
#     cv2.fillConvexPoly(data,pts1.astype(int),(0,0,0))



# activities for while loop to be implemented

while True:
    # starting the pipeline for intelrealsense
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    data = color_frame.get_data()
    data = np.array(data)

    # defining the model for hand detection
    img = tf.image.resize_with_pad(np.expand_dims(data, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # calling connection functions
    draw_connections(data, keypoints_with_scores, EDGES, 0.2)
    draw_keypoints(data, keypoints_with_scores, 0.2)
    y, x, c = data.shape
    shaped = np.squeeze(np.multiply(keypoints_with_scores[0][0][7:11], [y, x, 1]))

    # naming points from x,y and z
    y1, x1 = shaped[0][0:2]
    #     y2,x2 = shaped[1][0:2]
    #     y3,x3 = shaped[2][0:2]
    #     y4,x4 = shaped[2][0:2]
    #     print(y1,x1)

    zDepth_1 = depth_frame.get_distance(int(x1), int(y1))
    #     zDepth_2 = depth_frame.get_distance(int(x2),int(y2))
    #     zDepth_3 = depth_frame.get_distance(int(x3),int(y3))
    #     zDepth_4 = depth_frame.get_distance(int(x4),int(y4))

    print(y1, x1, zDepth_1)

    # Start the detection of aruco markers

    cv_file = cv2.FileStorage("test.yaml", cv2.FILE_STORAGE_READ)

    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_matrix = cv_file.getNode("dist_coeff").mat()

    # to detect aruco markers
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(data, aruco_dict, parameters = parameters)
    axis = np.float32([[-.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0], [.5, -.5, 0],
                       [-.5, -.5, 1], [-.5, .5, 1], [.5, .5, 1], [.5, -.5, 1]])


    data = aruco.drawDetectedMarkers(data, corners, ids)
    if ids is not None and len(ids) > 0:
        # Estimate the posture per each Aruco marker
        r_vec, t_vec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)
        # print(t_vec)

        for rvec, tvec in zip(r_vec, t_vec):
            if len(sys.argv) == 2 and sys.argv[1] == 'cube':
                try:
                    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_matrix)
                    data = drawCube(data, corners, imgpts)
                except:
                    continue
            else:
                data = aruco.drawAxis(data, camera_matrix, dist_matrix, rvec, tvec, 0.05)

    # aruco.drawMarker(data,ids,)

    # aruco.
    # print(corners)
    # tl = corners[0][0][0]

    # augment_aruco(corners,id,data,imgAug)
    # print(corners[0][0][0][0])
    # tl = corners[0][0][0], corners[0][0][1]q
    # print(tl)
    # data = aruco.drawDetectedMarkers(data, corners, ids)
    # aruco.drawDetectedMarkers(data, rejectedImgPoints, borderColor=(100, 0, 240)



    cv2.imshow('Mask',  data)
    #     cv2.imshow('Mask', depth_frame)

    if cv2.waitKey(25) == ord('q'):
        break

cv2.destroyAllWindows()
pipeline.stop()

# To save the callibration on yaml file

# print(rvec)
# print(rvec[0])
