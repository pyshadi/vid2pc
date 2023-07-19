import os
import open3d as o3d
from depth_estimation import *

def create_point_cloud(input_file):
    _, ext = os.path.splitext(input_file)
    ext = ext.lower()

    if ext in ['.jpg', '.png']:  # image formats to accept
        frame = cv2.imread(input_file)
        process_frame(frame, 0)
    elif ext in ['.avi']:  # video formats to accept
        cap = cv2.VideoCapture(input_file)
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                process_frame(frame, frame_num)
                frame_num += 1
            else:
                break

        cap.release()

def process_frame(frame, frame_num):
    depth_map = depth_estimation(frame, model)
    point_cloud = []

    height, width = depth_map.shape[:2]
    fx = fy = 300  # Focal length
    cx = width / 2.0  # Principal point x
    cy = height / 2.0  # Principal point y

    for i in range(height):
        for j in range(width):
            depth_value = depth_map[i, j]
            # Convert from pixel coordinates to 3D coordinates
            x = (j - cx) * depth_value / fx
            y = (i - cy) * depth_value / fy
            z = depth_value

            point = [x, y, z]
            point_cloud.append(point)

    point_cloud = np.array(point_cloud)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    o3d.io.write_point_cloud(f"../data/pc/frame_{frame_num}.pcd", pcd)
