from data_preparing._1_nrrd_png_image import *
from data_preparing._2_npz_test import *
from data_preparing._3_h5_2 import *
from data_preparing._4_save_spacing import *

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "data")

if __name__ == '__main__':
    '''save_spacing'''
    input_dir = data_file + "/FLAIR"
    output_dir = data_file + "/spacing_img"
    process_folder(input_dir, output_dir)

    '''image to png'''
    input_dir = data_file + "/FLAIR"
    output_dir = data_file + "/png_FLAIR"
    convert_to_2d_images(input_dir, output_dir)

    input_dir = data_file + "/T1"
    output_dir = data_file + "/png_T1"
    convert_to_2d_images(input_dir, output_dir)

    '''test_npz'''
    image_dir = data_file + '/png_FLAIR'
    image_dir_t1 = data_file + '/png_T1'
    label_dir = data_file + '/png_FLAIR'
    output_dir = data_file + '/test_npz'
    data = test_load_images_and_labels(image_dir, image_dir_t1, label_dir)
    test_save_data_as_npz(data, output_dir)

    '''make h5'''
    input_folder = data_file + "/test_npz"
    output_folder = data_file + "/test_vol_h5"
    make_h5(input_folder, output_folder)

    "make list"
    directory = data_file + '/lists/lists_wmh'
    os.makedirs(directory, exist_ok=True)
    folder_path = data_file + '/test_vol_h5'
    file_names_with_extension = os.listdir(folder_path)
    with open(os.path.join(directory, 'test_vol.txt'), 'w') as file:
        for name_with_extension in file_names_with_extension:
            file_name_without_extension = name_with_extension[:-7]
            file.write(file_name_without_extension + '\n')
