"""
This script contain the aruco marker utils

Aouther:Mohammed A. Alsaggaf
date: 2/18/2021

"""



import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import pandas as pd

def get_aruco_pose(frame, aruco_dict, id_list,  mtx, dist, marker_size=4.5):

    
    #-- Convert in gray scale
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()

    #lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    translation_vector = []
    rotation_vector = []
    returned_ids = []

    if ids is not None:

        #check that the detected ids needs to be detected by the user
        for i, detceted_id in enumerate(ids):
            for toFind_id in id_list:

                if detceted_id[0] == toFind_id:
                    
                    # estimate pose of each marker and return the values
                    # rvet and tvec-different from camera coefficients
                    rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)
                    (rvec - tvec).any() # get rid of that nasty numpy value array error

                    returned_ids.append([detceted_id[0]])
                    translation_vector.append(tvec[i][0])
                    rotation_vector.append(rvec[i][0])
        
        returned_ids = np.array(returned_ids)
        rotation_vector = np.array(rotation_vector)
        translation_vector = np.array(translation_vector)


    return corners, returned_ids, rotation_vector, translation_vector

def get_camera_pose(rvec, tvec):

    #-- Obtain the rotation matrix tag->camera
    
    R_ct = np.matrix(cv2.Rodrigues(rvec)[0])

    R_tc = R_ct.T

    #-- Now get Position and attitude f the camera respect to the marker
    pos_camera = -R_tc*np.matrix(tvec).T
    

    return pos_camera


#test
if __name__ == "__main__":

    McsvName = "45mm_30cm_Mpose.csv"
    CcsvName = "45mm_30cm_Cpose.csv"
    mrk_size = 4.5
    readsThresh = 40

    mtx = np.loadtxt("cameraMatrix.txt", delimiter=",", dtype=np.float32)
    dist = np.loadtxt("cameraDistortion.txt", delimiter=",", dtype=np.float32)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    id_list = [0,1,2,3,4]
    
    Mdict ={"id":[] ,"x":[], "y":[], "z":[]}
    Cdict ={"id":[] ,"x":[], "y":[], "z":[]}
    #Mpose = open(""25mm_90cm_Mpose.csv","w")
    #Cpose = open("25mm_90cm_Cpose.csv","w")
    #Cpose.write("id,x,y,z\n")
    #Mpose.write("id,x,y,z\n")

    count = 0
    
    while (True):

        ret, frame = cap.read()

        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        

        corners, returned_ids, rotation_vector, translation_vector = get_aruco_pose(frame, aruco_dict, id_list, mtx, dist,marker_size=mrk_size)
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX


        for i in range(0, len(returned_ids)):
            # draw axis for the aruco markers
            aruco.drawAxis(frame, mtx, dist, rotation_vector[i], translation_vector[i], 2.5)
            
            Mdict["id"].append(returned_ids[i][0])
            Mdict["x"].append(translation_vector[i][0])
            Mdict["y"].append(translation_vector[i][1])
            Mdict["z"].append(translation_vector[i][2])
            
            #str_position = "%d,%4.0f,%4.0f,%4.0f\n"%(returned_ids[i],translation_vector[i][0], translation_vector[i][1], translation_vector[i][2])
            #Mpose.write(str_position)

            #str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(translation_vector[i][0], translation_vector[i][1], translation_vector[i][2])
            #cv2.putText(frame, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            pos_camera = get_camera_pose(rotation_vector[i], translation_vector[i])
            Cdict["id"].append(returned_ids[i][0])
            Cdict["x"].append(pos_camera.item(0))
            Cdict["y"].append(pos_camera.item(1))
            Cdict["z"].append(pos_camera.item(2))

            #str_position = "%d,%4.0f,%4.0f,%4.0f\n"%(returned_ids[i],pos_camera[0], pos_camera[1], pos_camera[2])
            #Cpose.write(str_position)
            #cv2.putText(frame, str_position, (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)    

        aruco.drawDetectedMarkers(frame, corners)
        count = count + 1

        # display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif count == readsThresh:
            break

        # When everything done, release the capture
    #Cpose.close()
    #Mpose.close()

    Mdf = pd.DataFrame(data=Mdict)
    Cdf = pd.DataFrame(data=Cdict)
    Mdf =  Mdf.sort_values(by=['id'])
    Cdf = Cdf.sort_values(by=['id'])
    
    Mdf.to_csv(McsvName, float_format='%.2f', index=False)
    Cdf.to_csv(CcsvName, float_format='%.2f', index=False)

    cap.release()
    cv2.destroyAllWindows()
