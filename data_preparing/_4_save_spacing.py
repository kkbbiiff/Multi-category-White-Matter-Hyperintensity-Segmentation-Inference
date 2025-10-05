import os
import nibabel as nib


def get_nifti_spacing(file_path):
    nii = nib.load(file_path)
    header = nii.header
    voxel_size = header.get_zooms()
    return voxel_size


def save_spacing_and_voxel_size(file_path, output_dir):
    base_name = os.path.splitext(os.path.basename(file_path))[0][:-7]
    output_file = os.path.join(output_dir, base_name + '.txt')

    spacing = get_nifti_spacing(file_path)

    with open(output_file, 'w') as f:
        f.write(f"Spacing: {spacing}\n")
        f.write(f"Voxel Size: {spacing}\n")


def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.nii.gz'):
            file_path = os.path.join(input_dir, file)
            save_spacing_and_voxel_size(file_path, output_dir)

