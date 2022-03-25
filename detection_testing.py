mport numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import sys

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)



while True:
    #     pipeline = rs.pipeline()
    #     config = rs.config()
    #     config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    #     config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    #     ret, depth_frame1, color_frame = dc.get_frame()


    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    data = color_frame.get_data()
    data = np.array(data)
    # gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    # print(data.shape[:2])
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
    # aruco.drawMarker(data,ids,)
    # aruco.
    # print(corners)q
    ProjectImage = aruco.drawDetectedMarkers(data, corners, ids)
    # aruco.drawDetectedMarkers(data, rejectedImgPoints, borderColor=(100, 0, 240))
    if ids is not None and len(ids) > 0:
        # Estimate the posture per each Aruco marker
        r_vec, t_vec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)
        # print(t_vec)

        for rvec, tvec in zip(r_vec, t_vec):
            if len(sys.argv) == 2 and sys.argv[1] == 'cube':
                try:
                    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_matrix)
                    # ProjectImage = drawCube(ProjectImage, corners, imgpts)
                except:
                    continue
            else:
                ProjectImage = aruco.drawAxis(ProjectImage, camera_matrix, dist_matrix, rvec, tvec, 0.05)

    # rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix )
    # (rvec - tvec).any()


    # aruco.drawAxis(data, camera_matrix, dist_matrix, rvec, tvec, 0.1)




    # print(rvec,tvec)
    # print(rot_trans)

    # aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)


        # draw a square around the markers
    # aruco.drawDetectedMarkers(frame, corners)


    cv2.imshow('Mask',  ProjectImage)
    #     cv2.imshow('Mask', depth_frame)

    if cv2.waitKey(25) == ord('q'):
        break

cv2.destroyAllWindows()
pipeline.stop()

# To save the callibration on yaml file

# print(rvec)
