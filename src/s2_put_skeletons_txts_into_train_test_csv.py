import numpy as np
import csv
import utils.lib_commons as lib_commons

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
    

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings
cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s2_put_skeletons_txts_into_train_test_csv.py"]

SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

CLASSES = np.array(cfg_all["classes"])

# Action recognition
WINDOW_SIZE = int(cfg_all["features"]["window_size"]) # number of frames used to extract features.

# Input and output
SRC_DETECTED_SKELETONS_TRAIN_FOLDER = par(cfg["input"]["detected_skeletons_train_folder"])
SRC_DETECTED_SKELETONS_TEST_FOLDER = par(cfg["input"]["detected_skeletons_test_folder"])

DST_TRAINING_SET = par(cfg["output"]["training_set"])
DST_TEST_SET = par(cfg["output"]["test_set"])


def read_skeletons_from_ith_txt(i, skeletons_folder):
    filename = skeletons_folder + \
        SKELETON_FILENAME_FORMAT.format(i)
    skeletons_in_ith_txt = lib_commons.read_listlist(filename)
    return skeletons_in_ith_txt

    
def write_on_csv(num_skeletons, csvpath, skeletons_folder):
    with open(csvpath, 'w',newline='') as f:
        csv_columns = ['classID','videoID','frameID','class','frame_name','joint0_x',
            'joint0_y','joint1_x','joint1_y','joint2_x','joint2_y','joint3_x','joint3_y',
            'joint4_x','joint4_y','joint5_x','joint5_y','joint6_x','joint6_y','joint7_x',
            'joint7_y','joint8_x','joint8_y','joint9_x','joint9_y','joint10_x','joint10_y',
            'joint11_x','joint11_y','joint12_x','joint12_y','joint13_x','joint13_y',
            'joint14_x','joint14_y','joint15_x','joint15_y','joint16_x','joint16_y',
            'joint17_x','joint17_y']
        # create the csv writer
        writer = csv.writer(f)
        #write columns name on the csv file
        writer.writerow(csv_columns)
        for i in range(num_skeletons):
            # Read skeletons from a txt
            skeletons = read_skeletons_from_ith_txt(i, skeletons_folder)
            if not skeletons:  # If empty, discard this image
                continue
            # write data on the csv file
            writer.writerow(skeletons[0])

# -- Main
def main():
    filepaths_train = lib_commons.get_filenames(SRC_DETECTED_SKELETONS_TRAIN_FOLDER,
                                          use_sort=True, with_folder_path=True)
    num_skeletons_train = len(filepaths_train)

    filepaths_test = lib_commons.get_filenames(SRC_DETECTED_SKELETONS_TEST_FOLDER,
                                          use_sort=True, with_folder_path=True)
    num_skeletons_test = len(filepaths_test)

    #creating csv files for training and test set
    os.makedirs(os.path.dirname(DST_TRAINING_SET), exist_ok=True)
    os.makedirs(os.path.dirname(DST_TEST_SET), exist_ok=True)

    write_on_csv(num_skeletons_train, DST_TRAINING_SET, SRC_DETECTED_SKELETONS_TRAIN_FOLDER)
    print("\nSaved training set at the following path: " + DST_TRAINING_SET)

    write_on_csv(num_skeletons_test, DST_TEST_SET, SRC_DETECTED_SKELETONS_TEST_FOLDER)
    print("\nSaved test set at the following path: " + DST_TEST_SET)

if __name__ == "__main__":
    main()
