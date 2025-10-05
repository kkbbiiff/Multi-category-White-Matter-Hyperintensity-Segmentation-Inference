import os
import nibabel as nib
import nrrd
import cv2
import numpy as np


def nifti_to_png(nii_data, file_name, save_dir):
    nii_array = nii_data.get_fdata()

    os.makedirs(save_dir, exist_ok=True)
    for i in range(nii_array.shape[-1]):
        slice_data = nii_array[:, :, i]
        slice_data = slice_data * 255 / np.max(slice_data)
        slice_data = np.uint8(slice_data)
        rotated_slice_data = cv2.rotate(slice_data, cv2.ROTATE_90_COUNTERCLOCKWISE)

        rotated_slice_data = np.fliplr(rotated_slice_data)

        slice_num = str(i + 1).zfill(2)
        png_filename = os.path.join(save_dir, f"{file_name}_{slice_num}.png")
        cv2.imwrite(png_filename, rotated_slice_data)

def convert_to_2d_images(input_dir, output_dir):
    files = os.listdir(input_dir)

    for file in files:
        file_path = os.path.join(input_dir, file)

        if file.endswith('.nii.gz'):
            file_name = os.path.splitext(os.path.basename(file))[0][:-4]  # 删除最后四个字符
            nii_data = nib.load(file_path)
            nifti_to_png(nii_data, file_name, output_dir)
        elif file.endswith('.nrrd'):
            file_name = os.path.splitext(os.path.basename(file))[0]
            nrrd_data, nrrd_options = nrrd.read(file_path)
            nii_data = nib.Nifti1Image(nrrd_data, np.eye(4))
            nifti_to_png(nii_data, file_name, output_dir)