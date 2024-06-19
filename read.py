import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def images(filename):
    with open(filename, 'rb') as f:
        # 读取文件头信息
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # 读取图像数据,同时调整大小
        images = torch.frombuffer(f.read(), dtype=torch.uint8).reshape(num_images, num_rows, num_cols)
    return images

def labels(filename):
    with open(filename, 'rb') as f:
        # 跳过无关信息
        f.read(8)
        labels = torch.frombuffer(f.read(), dtype=torch.uint8)
    return labels
