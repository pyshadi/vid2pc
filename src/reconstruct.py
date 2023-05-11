import numpy as np
import open3d as o3d

def merge_point_clouds(file_prefix, num_frames):
    # Create an empty point cloud object
    scene = o3d.geometry.PointCloud()

    # Iterate over the frames and load each point cloud file
    for i in range(num_frames):
        filename = f"{file_prefix}{i}.pcd"
        cloud = o3d.io.read_point_cloud(filename)

        # Merge the current point cloud with the scene
        scene += cloud

    return scene

# Specify the file prefix and the number of frames to process
file_prefix = "../data/pc/frame_"
num_frames = 10

# Merge the point clouds into a scene
scene = merge_point_clouds(file_prefix, num_frames)

# Save the merged point cloud as a PCD file
output_filename = "scene.pcd"
o3d.io.write_point_cloud(output_filename, scene)

print(f"Scene reconstruction completed. Merged point cloud saved to {output_filename}.")
pcd = o3d.io.read_point_cloud("scene.pcd")
o3d.visualization.draw_geometries([pcd])