import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
import open3d as o3d

# 定义获取球领域内点的函数
def get_points_in_sphere(x, center, radius):
    distance = np.linalg.norm(x - center, axis=1)
    return x[distance < radius]


def get_class_by_coord(coord_array, data_array):
    # 构造一个字典，将坐标点和对应的类别映射起来
    coord_dict = dict(zip(map(tuple, data_array[:, :3]), data_array[:, 3]))
    # 将coord_array中的每个坐标点转换为元组，然后在字典中查找对应的类别
    return np.array([coord_dict.get(tuple(coord), -1) for coord in coord_array])

# 读取场景点云数据
scene_plydata = PlyData.read('/Downloads/Lille2_scene.ply')
scene_x = np.stack([scene_plydata['vertex']['x'], scene_plydata['vertex']['y'], scene_plydata['vertex']['z']], axis=1).astype(np.float64)
scene_class_array = scene_plydata['vertex']['class'].astype(np.int32)

# 获取地面点索引
ground_indices = np.where(scene_class_array == 1)[0]
ground_points = scene_x[ground_indices]
# 上下文感知模块
context_point = np.loadtxt('sphere_lille2.txt',dtype=np.float64)
# 对B中每个点在A中找到对应的索引
A=ground_points
B=context_point
A_tuple = [tuple(x) for x in A]
B_tuple = [tuple(x) for x in B]

# 计算A中除B外的点
result = np.array(list(set(A_tuple).difference(B_tuple)))
# 将点坐标从元组转化为数组
result = np.array(result)

# 读取所有物体点云数据
object_folder = '/Downloads/Lille2/sample' # 存放所有物体点云文件的文件夹
object_files = os.listdir(object_folder)

# 创建pytorch tensor用于存放所有物体点云数据
x_tensor_list = []
class_tensor_list = []
i=0
for file in object_files:
    # 读取物体点云数据
    object_plydata = PlyData.read(os.path.join(object_folder, file))
    object_x = np.stack([object_plydata['vertex']['x'], object_plydata['vertex']['y'], object_plydata['vertex']['z']], axis=1).astype(np.float32)
    object_class_array = object_plydata['vertex']['class'].astype(np.int32)

    # 计算物体最低点并更新位置
    min_height_index = np.argmin(object_x[:, 2])
    min_height_point = object_x[min_height_index]
    while True:
        # random_ground_point = ground_points[np.random.randint(len(ground_points))]
        random_ground_point = result[np.random.randint(len(result))]

        # 获取球领域内点的位置
        points_in_sphere = get_points_in_sphere(scene_x, random_ground_point, 1.2)
        # 场景类别转为（n，1）
        scene_class_arrays=scene_class_array.reshape(-1,1)
        # 获取坐标类别数组（n,4)
        coord_and_class = np.concatenate((scene_x,scene_class_arrays),axis=1)
        class_in_sphere = get_class_by_coord(points_in_sphere,coord_and_class)
        if np.all(class_in_sphere == 1.):
            break

    object_x[:, 0] += (random_ground_point[0] - min_height_point[0])
    object_x[:, 1] += (random_ground_point[1] - min_height_point[1])
    object_x[:, 2] += (random_ground_point[2] - min_height_point[2])

    # 创建pytorch tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    object_x_tensor = torch.from_numpy(object_x).to(device)
    object_class_tensor = torch.from_numpy(object_class_array).to(device)

        
    # 将物体点云合并到场景点云中
    i+=1
    print('正在合并第{}个物体'.format(i))
    x_tensor_list.append(object_x_tensor)
    class_tensor_list.append(object_class_tensor)
    scene_x = np.concatenate((scene_x, object_x), axis=0)
    scene_class_array = np.concatenate((scene_class_array, object_class_array), axis=0)

# 将所有点云转换成Tensor
x_tensor = torch.from_numpy(scene_x).to(device)
class_tensor = torch.from_numpy(scene_class_array).to(device)

# 保存新的点云数据
new_data = np.zeros(len(scene_x), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),  ('class', 'i4')])
new_data['x'] = scene_x[:, 0]
new_data['y'] = scene_x[:, 1]
new_data['z'] = scene_x[:, 2]
new_data['class'] = scene_class_array

new_ply = PlyData([PlyElement.describe(new_data, 'vertex')], text=False)
new_ply.write('new_Lille2.ply')