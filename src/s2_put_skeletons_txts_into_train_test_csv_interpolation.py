import numpy as np

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import csv
    import utils.lib_commons as lib_commons
    from scipy import interpolate

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

def get_directory(framepath):
    directory = ""
    for letter in framepath:
        if(letter != '/'):
            directory += letter
        else:
            return directory

def scan_csv_based_on_directory(data, directory, index_col):
    idx_rows_known_value = []
    idx_rows_missing_value = []
    index = 0
    flag = False
    while(index < len(data)):
        if(get_directory(data[index][4]) == directory):
            flag = True
            if(data[index][index_col] == 0):
                idx_rows_missing_value.append(index)
            else:
                idx_rows_known_value.append(index)
        elif(flag == True):
            break
        index += 1
    return idx_rows_missing_value, idx_rows_known_value

def interpolation(data):
    scanned_columns = []
    scanned_directories = []
    count_interpolation = 0
    count_not_int = 0
    index_row = 0
    while(index_row < len(data)):
        #index_row = 5 = index where joints coordinates starts in a row
        index_col = 5
        while(index_col < len(data[index_row])):
            if(data[index_row][index_col] == 0):
                # 4: index in a row where the directory is specified
                directory = get_directory(data[index_row][4])
                if (directory in scanned_directories):
                    if(index_col in scanned_columns):
                        index_col += 1
                        continue
                else:
                    scanned_directories.append(directory)
                    if(len(scanned_columns) != 0):
                        scanned_columns.clear()
                scanned_columns.append(index_col)
                # saving idx of rows that contains or not a value (missing value => value = 0)
                idx_rows_missing_value, idx_rows_known_value = scan_csv_based_on_directory(data,directory, index_col)
                #saving values that are not zeros
                known_values = []
                for idx_row in idx_rows_known_value:
                    known_values.append(data[idx_row][index_col])
                #calculate interpolation function
                len_rows_missing_value = len(idx_rows_missing_value)
                len_rows_known_value = len(idx_rows_known_value)
                percentage = len_rows_known_value*100/(len_rows_known_value + len_rows_missing_value)
                print("length of idx_rows_missing_value = " + str(len_rows_missing_value))
                print("length of idx_rows_known_value = " + str(len_rows_known_value))
                print("length of known_values = " + str(len(known_values)))
                print("percentage = " + str(percentage) + "%")
                if(percentage >= 60):
                    print("Interpolation phase")
                    count_interpolation += 1
                    f_interpolation = interpolate.interp1d(idx_rows_known_value, known_values, kind="linear")
                    #update data with interpolate values
                    for idx_row in idx_rows_missing_value:
                        if(idx_row >= idx_rows_known_value[0] and idx_row <= idx_rows_known_value[-1]):
                            data[idx_row][index_col] = f_interpolation(idx_row)
                else:
                    count_not_int += 1
                    print("No interpolation phase needed")
            index_col += 1
        index_row +=1
    print("count of interpolation = " + str(count_interpolation))
    print("count of no interpolation = " + str(count_not_int))
    print("percentage of interpolation = " + str(count_interpolation*100/(count_not_int + count_interpolation)))
    return data

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
        data = []
        for i in range(num_skeletons):
            # Read skeletons from a txt
            skeletons = read_skeletons_from_ith_txt(i, skeletons_folder)
            if not skeletons:  # If empty, discard this image
                continue
            data.append(skeletons[0])
        data = interpolation(data)
        for row in data:
            # write data on the csv file
            writer.writerow(row)

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
