import numpy as np
import torch
from sklearn.cluster import DBSCAN
from plyfile import PlyData, PlyElement

# 读取点云数据
plydata = PlyData.read('Lille2.ply')
x = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], axis=1).astype(np.float32)
class_array = plydata['vertex']['class'].astype(np.int32)

# 创建pytorch tensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x_tensor = torch.from_numpy(x).to(device)
class_tensor = torch.from_numpy(class_array).to(device)

# 提取类别为8的点云
car_indices = torch.where(class_tensor == 8)[0]
car_points = x_tensor[car_indices]

# DBSCAN聚类
dbscan = DBSCAN(eps=0.45, min_samples=10).fit(car_points.cpu().numpy())
labels = dbscan.labels_

# 按点数量从大到小排序
unique_labels, counts = np.unique(labels, return_counts=True)
sorted_indices = np.argsort(-counts)
sorted_labels = unique_labels[sorted_indices]

# 尾部类别实例提取
for i in range(min(5, len(sorted_labels))):
    label = sorted_labels[i]
    indices = np.where(labels == label)[0]
    car_points_cluster = car_points[indices].cpu().numpy()

    # 保存点云数据
    new_data = np.zeros(len(car_points_cluster), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('class', 'i4')])
    new_data['x'] = car_points_cluster[:, 0]
    new_data['y'] = car_points_cluster[:, 1]
    new_data['z'] = car_points_cluster[:, 2]
    new_data['class'] = 8

    new_ply = PlyData([PlyElement.describe(new_data, 'vertex')], text=False)
    new_ply.write(f'car_cluster_{i}.ply')