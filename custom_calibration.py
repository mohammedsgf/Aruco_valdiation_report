import sys
from camera_calibration import Camera_Calibration_API
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

images_path_list = glob.glob("/home/pi/Desktop/seniorProj/callImgs4/*.jpg")

print("Number of images in directory",len(images_path_list))

chessboard = Camera_Calibration_API(pattern_type="chessboard",
                                    pattern_rows=8,
                                    pattern_columns=6,
                                    distance_in_world_units = 2.5 )

results = chessboard.calibrate_camera(images_path_list)



print(chessboard.calibration_df)

refined_images_paths = [img_path for i,img_path in enumerate(chessboard.calibration_df.image_names) if chessboard.calibration_df.reprojection_error[i] < 0.04]

refined_chessboard = Camera_Calibration_API(pattern_type="chessboard",
                                    pattern_rows=8,
                                    pattern_columns=6,
                                    distance_in_world_units = 2.5 )

refined_results = refined_chessboard.calibrate_camera(refined_images_paths)

np.savetxt("cameraMatrix.txt", refined_results['intrinsic_matrix'], delimiter=',')
np.savetxt("cameraDistortion.txt",refined_results['distortion_coefficients'],delimiter=',')

