from generate_pc import *
from vid_capture import vid_capture

vid_capture()
create_point_cloud('../data/video/outpy.avi')
# Read and visualize each .pcd file
i=1
pcd = o3d.io.read_point_cloud(f"../data/pc/frame_{i}.pcd")
o3d.visualization.draw_geometries([pcd])