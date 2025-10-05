import os
import h5py
import numpy as np

input_folder = "test_npz"
output_folder = "test_h5"

def make_h5(input_folder, output_folder):
    npz_files = [f for f in os.listdir(input_folder) if f.endswith(".npz")]

    grouped_npz_files = {}
    for npz_file in npz_files:
        prefix = npz_file[:-7]
        if prefix not in grouped_npz_files:
            grouped_npz_files[prefix] = []
        grouped_npz_files[prefix].append(npz_file)

    for prefix, npz_files in grouped_npz_files.items():
        print(f"Prefix: {prefix}, Number of NPZ files: {len(npz_files)}")
        npz_files = sorted(npz_files)
        print(npz_files)

        group_image_data = []
        group_label_t1_data = []
        group_label_data = []


        for npz_file in npz_files:

            data = np.load(os.path.join(input_folder, npz_file))
            image_data = data['image']
            image_t1_data = data['image_t1']
            label_data = data['label']

            group_image_data.append(image_data)
            group_label_t1_data.append(image_t1_data)
            group_label_data.append(label_data)


        group_image_data = np.array(group_image_data)
        group_label_t1_data = np.array(group_label_t1_data)
        group_label_data = np.array(group_label_data)
        print(group_image_data.shape)
        print(group_label_t1_data.shape)
        print(group_label_data.shape)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file = f"{prefix}.npy.h5"
        with h5py.File(os.path.join(output_folder, output_file), "w") as hf:
            hf.create_dataset("image", data=group_image_data)
            hf.create_dataset("image_t1", data=group_label_t1_data)
            hf.create_dataset("label", data=group_label_data)

        print("Created:", output_file)


