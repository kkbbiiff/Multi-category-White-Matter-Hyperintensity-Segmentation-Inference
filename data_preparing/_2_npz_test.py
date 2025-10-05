import os
import numpy as np
from PIL import Image

def test_load_images_and_labels(image_dir, image_dir_t1, label_dir):
    image_files = sorted(os.listdir(image_dir))
    image_files_t1 = sorted(os.listdir(image_dir_t1))
    label_files = sorted(os.listdir(label_dir))
    data = []
    for img_file, img_file_t1, lbl_file in zip(image_files, image_files_t1, label_files):
        if img_file[:25] == img_file_t1[:25] and img_file[:25] == lbl_file[:25]:
            img_path = os.path.join(image_dir, img_file)
            img_path_t1 = os.path.join(image_dir_t1, img_file_t1)
            lbl_path = os.path.join(label_dir, lbl_file)
            image = np.array(Image.open(img_path)) / 255.0  # 归一化到0~1
            image_t1 = np.array(Image.open(img_path_t1)) / 255.0  # 归一化到0~1
            label = np.array(Image.open(lbl_path)).astype(np.float32)  # 转换为浮点数
            # 修改标签像素值
            label[label == 0] = 0.0
            label[label == 255] = 1.0
            label[label == 150] = 2.0
            label[label == 100] = 3.0
            label[label == 70] = 0.0
            label[label == 30] = 0.0
            data.append({'image': image, 'image_t1': image_t1, 'label': label, 'image_path': img_path})
    return data


def test_save_data_as_npz(data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for item in data:
        image_name = os.path.splitext(os.path.basename(item['image_path']))[0]
        output_file = os.path.join(output_dir, f"{image_name[:-5]}{image_name[-2:]}.npz")
        np.savez(output_file, image=item['image'], image_t1=item['image_t1'], label=item['label'])

