## 0. Environment Setup
Use the provided environment configuration file to set up the environment:  `vmunet_env.yml`

## 1. Data Preparation
Please prepare the following two types of MRI images for each patient:  
FLAIR images (.nii.gz format)  
T1w images (.nii.gz format)

- **FLAIR images**  
  Should be placed in: `data/FLAIR`  
  Example filename: `05_1_LJL_FLAIR_20190315_PR.nii.gz`  
  Naming structure: `Index_Group_PatientInitials_FLAIR_ScanDate_PR.nii.gz`  

- **T1w images**  
  Should be placed in: `data/T1`  
  Example filename: `05_1_LJL_FLAIR_20190315_PR_T1.nii.gz`  
  Naming structure: `Index_Group_PatientInitials_FLAIR_ScanDate_PR_T1.nii.gz`  

> **Important Notes**
> - For the same patient, FLAIR and T1w images must share the same core filename (only differing by the "_T1" suffix).  
> - If index, group, or scan date is unavailable, use arbitrary numbers as substitutes. **No fields may be omitted**.  


## 2. Data Preprocessing
Run the preprocessing script:  ```python _1_Data_Processing.py```

The preprocessed files will be automatically saved to the corresponding folders under the `data` directory.

## 3. Inference Procedure
Run the inference script:  ```python _2_Inference.py```

#### Results Description
- All results will be saved in the `results` folder  
- Each inference run generates an independent folder  
- Segmentation results are stored in the `outputs` subfolder  


#### Result Files Explanation
- `xxx_img.nii.gz`: Original MRI image  
- `xxx_pred.nii.gz`: Three-class segmentation labels  


#### Visualization Method
1. Open the files using **ITK-SNAP** software  
2. Load `xxx_img.nii.gz` as the main image  
3. Load `xxx_pred.nii.gz` as the segmentation layer  
4. The segmentation results will be displayed
