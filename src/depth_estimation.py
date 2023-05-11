from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from layers import *


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
custom_objects = {'BilinearUpSampling2D': BilinearUpscaler2D, 'depth_loss_function': depth_loss_function}

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
