import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree, cKDTree

ply_data = PlyData.read("/home/vge/Downloads/Lille2_scene.ply")
vertices = ply_data['vertex']
points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float64)
labels = vertices['class']

# 提取标签为1的地面点云
ground_indices = np.where(labels == 1)[0]
ground_cloud = points[ground_indices, :]
print(len(ground_cloud))

# 计算地面点云包围盒的长宽高，并按最长的一边进行切片，获取1000块地面点云
min_xyz = np.min(ground_cloud, axis=0)
max_xyz = np.max(ground_cloud, axis=0)
size_xyz = max_xyz - min_xyz
max_dim = np.argmax(size_xyz)
slice_size = size_xyz[max_dim] / 500
slices = np.arange(min_xyz[max_dim], max_xyz[max_dim], slice_size)
ground_slices = []
for i in range(len(slices)-1):
    mask = np.logical_and(ground_cloud[:, max_dim] >= slices[i], ground_cloud[:, max_dim] < slices[i+1])
    slice_cloud = ground_cloud[mask]
    if slice_cloud.shape[0] > 0:
        ground_slices.append(slice_cloud)

# 对每块点云计算中心点
center_points = []
for slice_cloud in ground_slices:
    center = np.mean(slice_cloud, axis=0)
    center_points.append(center)

# 连接每块点云的中心点云，包括路线中所有点，形成连续的中心线
line = np.array(center_points)
if len(line) > 0:
    start_point = line[0]
    end_point = line[-1]
    line = np.vstack([start_point, line, end_point])

# 以line中每个点为中心点，构建半径为1m的球领域，并保存球领域内的地面点
radius = 2.0
output_file = open("sphere_points.txt", "w")
tree = cKDTree(ground_cloud)  # 只构建一次KDTree
remaining_points = np.arange(ground_cloud.shape[0])  # 未被保存的点的索引
for center_point in line:
    indices = tree.query_ball_point(center_point, r=radius)
    indices = np.intersect1d(indices, remaining_points)  # 剔除已保存的点
    if len(indices) > 0:  # 加速处理，如果球领域内没有地面点，则跳过该点
        sphere_cloud = ground_cloud[indices, :]
        np.savetxt(output_file, sphere_cloud, delimiter=' ', fmt='%.18e')
        remaining_points = np.setdiff1d(remaining_points, indices)  # 更新未被保存的点的索引
output_file.close()
