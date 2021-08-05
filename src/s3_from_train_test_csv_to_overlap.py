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
cfg = cfg_all["s3_from_train_test_csv_to_overlap.py"]

CLASSES = np.array(cfg_all["classes"])
TIMESTEPS = cfg_all["timesteps"]

# Input and output
DST_TRAINING_SET = par(cfg["input"]["training_set"])
DST_TEST_SET = par(cfg["input"]["test_set"])

DST_TRAINING_SET_OVERLAP = par(cfg["output"]["training_set_overlap"])
DST_TEST_SET_OVERLAP = par(cfg["output"]["test_set_overlap"])
DST_Y_TRAIN = par(cfg["output"]["y_train"])
DST_Y_TEST = par(cfg["output"]["y_test"])

def get_directory(framepath):
    directory = ""
    for letter in framepath:
        if(letter != '/'):
            directory += letter
        else:
            return directory

def get_matrix(data, idx_row):
    matrix_directory = get_directory(data[idx_row][4])
    matrix = []
    bound = idx_row + TIMESTEPS
    while(idx_row < len(data) and idx_row < bound):
        if(get_directory(data[idx_row][4]) == matrix_directory):
            matrix.append(data[idx_row])
            idx_row += 1
        else:
            break

    #descard frames info (result: 36 columns)
    mtx_size = len(matrix)
    idx_row_mtx = 0
    while(idx_row_mtx < mtx_size):
        matrix[idx_row_mtx] = matrix[idx_row_mtx][5:]
        idx_row_mtx += 1

    #fill the incomplete matrix with zeros to reach the size( 60 rows x 36 columns )
    if(mtx_size != TIMESTEPS):
        zeros = np.repeat(0, 36)
        while(mtx_size < TIMESTEPS):
            matrix.append(zeros)
            mtx_size += 1
    return matrix

def overlap(data):
    data_overlap = []
    idx_row = 1
    while(idx_row < len(data)):
        matrix = get_matrix(data, idx_row)
        for row in matrix:
            data_overlap.append(row)
        idx_row = int(idx_row + (TIMESTEPS/2)) 
        print("rows done: " + str(idx_row) + " (" + str(int(idx_row*100/len(data))) + "%)")
    return data_overlap

def convert_class_to_id(row_class):
    id_class = 0
    for config_class in CLASSES:
        if (row_class == config_class):
            break
        else:
            id_class += 1
    return id_class

def get_id_classes(data_overlap):
    id_classes = []
    idx_row = 0
    while(idx_row < len(data_overlap)):
        row_class = data_overlap[idx_row][3]
        id_class = convert_class_to_id(row_class)
        count_rows = 0
        while(count_rows < TIMESTEPS):
            id_classes.append([id_class])
            count_rows += 1
        idx_row += TIMESTEPS
    return id_classes

def main():
    print("timesteps =" + str(TIMESTEPS))
    #creating csv files for overlap training and test set
    os.makedirs(os.path.dirname(DST_TRAINING_SET_OVERLAP), exist_ok=True)
    os.makedirs(os.path.dirname(DST_TEST_SET_OVERLAP), exist_ok=True)

    #creating csv files for classes
    os.makedirs(os.path.dirname(DST_Y_TRAIN), exist_ok=True)
    os.makedirs(os.path.dirname(DST_Y_TEST), exist_ok=True)

    #load training data from csv file
    with open(DST_TRAINING_SET) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        train_data = []
        for row in csv_reader:
            train_data.append(row)

    #elaborate overlapping on training data and save the result in new csv file
    data_overlap = overlap(train_data)
    with open(DST_TRAINING_SET_OVERLAP, 'w',newline='') as csv_X:
        writer = csv.writer(csv_X)
        for row in data_overlap:
            writer.writerow(row)

    #save training id classes in a csv file
    data_y = get_id_classes(train_data)
    with open(DST_Y_TRAIN, 'w',newline='') as csv_y:
        writer = csv.writer(csv_y)
        for row in data_y:
            writer.writerow(row)
    
    #load test data from csv file
    with open(DST_TEST_SET) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        test_data = []
        for row in csv_reader:
            test_data.append(row)

    #elaborate overlapping on test data and save the result in new csv file
    data_overlap = overlap(test_data)
    with open(DST_TEST_SET_OVERLAP, 'w',newline='') as csv_X:
        writer = csv.writer(csv_X)
        for row in data_overlap:
            writer.writerow(row)

    #save test id classes in a csv file
    data_y = get_id_classes(test_data)
    with open(DST_Y_TEST, 'w',newline='') as csv_y:
        writer = csv.writer(csv_y)
        for row in data_y:
            writer.writerow(row)

if __name__ == "__main__":
    main()