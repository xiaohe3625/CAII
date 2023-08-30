import numpy as np
import torch
from plyfile import PlyData, PlyElement

# 读取点云数据
plydata = PlyData.read('/home/vge/KPConv/Data/Paris/train/L001.ply')
x = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], axis=1).astype(np.float32)
class_array = plydata['vertex']['class'].astype(np.int32)

# 创建pytorch tensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x_tensor = torch.from_numpy(x).to(device)
class_tensor = torch.from_numpy(class_array).to(device)

# 随机采样class1和class2
class1_indices = np.where(class_array == 1)[0]
class2_indices = np.where(class_array == 2)[0]
class3_indices = np.where(class_array == 3)[0]

class1_sampled_indices = np.random.choice(class1_indices, size=int(len(class1_indices) * 0.2), replace=False)
class2_sampled_indices = np.random.choice(class2_indices, size=int(len(class2_indices) * 0.2), replace=False)

sampled_indices = np.concatenate([class1_sampled_indices, class2_sampled_indices])

# 对class1和class2进行随机采样
x_sampled = x_tensor[sampled_indices, :]
class_sampled = class_tensor[sampled_indices]

# 未被采样汽车和树，就是保存采样地面、建筑和未被采样
unsampled_indices = np.concatenate([class3_indices])
x_unsampled = x_tensor[unsampled_indices, :]
class_unsampled = class_tensor[unsampled_indices]

#保存采样和未采样，这里保存采样地面、建筑和未采样车和树
x_new = torch.cat([x_sampled, x_unsampled])
class_new = torch.cat([class_sampled, class_unsampled])

# 转回numpy数组并保存新的点云数据
x_new = x_new.cpu().numpy()
class_new = class_new.cpu().numpy()
new_data = np.zeros(len(x_new), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('class', 'i4')])
new_data['x'] = x_new[:, 0]
new_data['y'] = x_new[:, 1]
new_data['z'] = x_new[:, 2]
new_data['class'] = class_new
new_ply = PlyData([PlyElement.describe(new_data, 'vertex')], text=False)
new_ply.write('L001_scene.ply')