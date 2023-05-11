import open3d as o3d
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2

from layers import BilinearUpSampling2D
from tensorflow.keras.layers import Layer, InputSpec
# Load the DenseDepth model
model_path = '../model/nyu.h5'


# Custom object needed for inference and training
def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))


# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(model_path, custom_objects=custom_objects, compile=False)

def depth_estimation(frame, model):
    # Resize the frame to 640x480
    frame_resized = cv2.resize(frame, (640, 480))

    # Normalize the frame
    frame_normalized = frame_resized.astype('float32') / 255

    # Add batch and channel dimensions
    frame_input = np.expand_dims(frame_normalized, axis=[0, -1])

    # Predict the depth map
    depth_map = model.predict(frame_input)

    # Remove the batch dimension and squeeze the depth map to 2D
    depth_map = np.squeeze(depth_map, axis=(0, -1))


    return depth_map

def vid_capture():
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            out.write(frame)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Close all the frames
    cv2.destroyAllWindows()


def create_point_cloud():
    cap = cv2.VideoCapture('outpy.avi')
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            depth_map = depth_estimation(frame, model)
            point_cloud = []

            height, width = depth_map.shape[:2]
            fx = fy = 500  # Focal length
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
            frame_num += 1

        else:
            break

    cap.release()


#vid_capture()
create_point_cloud()
# Read and visualize each .pcd file
for i in range(2):
    pcd = o3d.io.read_point_cloud(f"../data/pc/frame_{i}.pcd")
    o3d.visualization.draw_geometries([pcd])