if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import csv
    import utils.lib_commons as lib_commons
    # lstm model for the har dataset
    from numpy import mean
    #from numpy import std
    from numpy import dstack
    from pandas import read_csv
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import LSTM
    from tensorflow.keras.utils import to_categorical

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings
cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s4_evaluation_model_performance.py"]

TIMESTEPS = cfg_all["timesteps"]

# Input
DST_TRAINING_SET_OVERLAP = par(cfg["input"]["training_set_overlap"])
DST_TEST_SET_OVERLAP = par(cfg["input"]["test_set_overlap"])
DST_Y_TRAIN = par(cfg["input"]["y_train"])
DST_Y_TEST = par(cfg["input"]["y_test"])


def load_csv_X_files(train_filepath, test_filepath):
    trainX = []
    with open(train_filepath, 'r',newline='') as train_file:
        csv_reader = csv.reader(train_file)
        for row in csv_reader:
            trainX.append(row)

    testX = []
    with open(test_filepath, 'r',newline='') as test_file:
        csv_reader = csv.reader(test_file)
        for row in csv_reader:
            testX.append(row)

    trainX = dstack(trainX)
    testX = dstack(testX)

    return trainX , testX

def load_csv_y_files(train_filepath, test_filepath):
    trainy = []
    with open(train_filepath, 'r',newline='') as train_file:
        csv_reader = csv.reader(train_file)
        for row in csv_reader:
            trainy.append(row)

    testy = []
    with open(test_filepath, 'r',newline='') as test_file:
        csv_reader = csv.reader(test_file)
        for row in csv_reader:
            testy.append(row)
    
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)

    return trainy, testy


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):

    verbose, epochs, batch_size = 1, 50, 32
    #-----------------------------------------------------------------
    n_features, n_outputs = len(trainX[0]), len(trainy) #1

    model = Sequential()
    model.add(LSTM(100, input_shape=(TIMESTEPS,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(trainX[0][0], trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # evaluate model
    _, accuracy = model.evaluate(testX[0][0], testy, batch_size=batch_size, verbose=0)

    return accuracy

def main():
    # load data
    trainX, testX = load_csv_X_files(DST_TRAINING_SET_OVERLAP, DST_TEST_SET_OVERLAP)
    trainy, testy = load_csv_y_files(DST_Y_TRAIN, DST_Y_TEST)
    print("data loaded")
    #elaborate model accuracy
    score = evaluate_model(trainX, trainy, testX, testy)
    score = score * 100.0
    print("Accuracy: " + str(score))

if __name__ == "__main__":
    main()
