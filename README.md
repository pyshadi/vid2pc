# vid2pc: Video to Point Cloud Scene using Depth Estimation 

The repo provides a depth estimation functionality using the DenseDepth model. It can process both images and videos to generate depth maps. 
The code is written in Python and utilizes the Open3D library for point cloud visualization.

## Prerequisites

- Open3D
- TensorFlow
- Keras
- NumPy
- OpenCV

## Usage

To use the depth estimation functionality, follow these steps:

1. Import the required libraries:

```
import os
import open3d as o3d
from depth_estimation import *
```

2. (Optional) If you want to capture a video using the webcam and save it as a file, you can use the <code>vid_capture()</code> function provided in <code>capture.py</code> file. 
This function captures video frames from the webcam, writes them to a video file, and displays the frames in a window. 
To use this function, simply call it in your code:

```
vid_capture()

```

3. Call the <code>create_point_cloud(input_file)</code> function, where input_file is the path to the image or video file you want to process. 
The function will automatically detect the file format and process it accordingly. 
Supported image formats are .jpg and .png, and supported video format is .avi.

```
create_point_cloud('path/to/input_file')

```

4. The code will generate a point cloud representing the depth information and it will be displayed using Open3D.

You can modify the desired frame rate <code>desired_fps</code> and the output file path <code>('../data/video/outpy.avi')</code> within the function according to your requirements.

5. To merge different .pcd frames into a single point cloud, you can use the <code>merge_point_clouds(file_prefix, num_frames)</code> function provided. 
This function takes the file prefix and the number of frames as input and merges the point clouds into a single scene. 
To use this function, simply call it in your code:

```
file_prefix = "../data/pc/frame_"
num_frames = 10
scene = merge_point_clouds(file_prefix, num_frames)

```

You can modify the file prefix <code>file_prefix</code>  and the number of frames (num_frames) according to your file naming convention and the actual number of frames to process. 
The merged point cloud will be saved as a PCD file, and the file name and path can be customized within the function.

## Customization
- Supported Image Formats: You can add or modify the list of supported image formats by editing the ext list in the <code>create_point_cloud()</code> function.

- Supported Video Formats: You can add or modify the list of supported video formats by editing the ext list in the <code>create_point_cloud()</code> function.

- Model Path: The code assumes that the DenseDepth model is located at '../model/nyu.h5'. 
If your model is located at a different path, you can modify the model_path variable in the code.

- Custom Loss Function: The code includes a custom loss function called depth_loss_function(). 
You can modify this function according to your specific requirements.

## Notes
- The code loads the pre-trained [DenseDepth](https://github.com/ialhashim/DenseDepth) model [NYU Depth V2 (165 MB)](https://drive.google.com/file/d/19dfvGvDfCRYaqxVKypp1fRHwK7XtSjVu/view) and performs inference using it. 
Make sure to have the model file (nyu.h5) available at the specified path.
- The BilinearUpscaler2D class is a compact variant to [BilinearSampling2D](https://github.com/ialhashim/DenseDepth/blob/master/layers.py) by DenseDepth.


## License
This code is released under the MIT License.
