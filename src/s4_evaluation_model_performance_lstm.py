if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import csv
    import utils.lib_commons as lib_commons
    import matplotlib.pyplot as plt
    import pickle
    # lstm model for the har dataset
    from numpy import mean
    from numpy import array
    from numpy import reshape
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

#Output
DST_MODEL_PATH = par(cfg["output"]["model_path"])
DST_CHART_PATH = par(cfg["output"]["chart_path"])

def load_csv_X_files(train_filepath, test_filepath):

    dataframe_trainX = read_csv(train_filepath, header=None, delim_whitespace=True)
    dataframe_testX = read_csv(test_filepath, header=None, delim_whitespace=True)

    trainX = array(dataframe_trainX.values).reshape(dataframe_trainX.shape[0], 1, dataframe_trainX.shape[1])
    testX = array(dataframe_testX.values).reshape(dataframe_testX.shape[0], 1, dataframe_testX.shape[1])

    return trainX, testX

def load_csv_y_files(train_filepath, test_filepath):
    dataframe_trainy = read_csv(train_filepath, header=None, delim_whitespace=True)
    dataframe_testy = read_csv(test_filepath, header=None, delim_whitespace=True)

    trainy = to_categorical(dataframe_trainy.values)
    testy = to_categorical(dataframe_testy.values)

    return trainy, testy


def evaluate_model(trainX, trainy, testX, testy):

    verbose, epochs, batch_size = 1, 100, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.title('accuracy and loss chart in epochs')
    plt.xlabel('epoch')
    plt.legend(['loss', 'accuracy'], loc='upper right')
    plt.show()
    #save plot
    plt.savefig(DST_CHART_PATH)

    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    # -- Save model
    print("Save model to " + DST_MODEL_PATH)
    with open(DST_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Saving phase complete")

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
