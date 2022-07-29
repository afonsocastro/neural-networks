#!/usr/bin/env python3

import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
import matplotlib.pyplot as plt
import json


if __name__ == '__main__':

    model_config = json.load(open('model_config.json'))

    validation_split = 0.3

    all_data = np.load('../data/learning_data_training.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                       encoding='ASCII')

    validation_n = len(all_data) * validation_split

    all_data = np.array(all_data)

    model = Sequential()

    for layer in model_config["layers"]:
        if layer["id"] == 0:
            model.add(Dense(units=layer["neurons"], input_shape=(650,), activation=layer["activation"],
                            kernel_regularizer='l1'))
            # model.add(Dropout(layer["dropout"]))
        elif layer["id"] == model_config["n_layers"] - 1:
            model.add(Dense(units=layer["neurons"], activation=layer["activation"], kernel_regularizer='l1'))
            model.add(Dropout(model_config["dropout"]))
        else:
            model.add(Dense(units=layer["neurons"], activation=layer["activation"], kernel_regularizer='l1'))
            # model.add(Dropout(layer["dropout"]))

    model.add(Dense(units=4, activation='softmax'))

    # `rankdir='LR'` is to make the graph horizontal.
    # keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    keras.utils.plot_model(model, show_shapes=True)

    model.summary()

    if model_config["optimizer"] == "Adam":
        model.compile(optimizer=Adam(learning_rate=model_config["lr"]), loss=model_config["loss"],
                      metrics=['accuracy'])
    elif model_config["optimizer"] == "SGD":
        model.compile(optimizer=SGD(learning_rate=model_config["lr"]), loss=model_config["loss"],
                      metrics=['accuracy'])
    elif model_config["optimizer"] == "Nadam":
        model.compile(optimizer=Nadam(learning_rate=model_config["lr"]), loss=model_config["loss"],
                      metrics=['accuracy'])
    elif model_config["optimizer"] == "RMSprop":
        model.compile(optimizer=RMSprop(learning_rate=model_config["lr"]), loss=model_config["loss"],
                      metrics=['accuracy'])

    # callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=model_config["early_stop_patience"])

    fit_history = model.fit(x=all_data[:, :-1], y=all_data[:, -1], validation_split=validation_split,
                            batch_size=model_config["batch_size"],
                            shuffle=True, epochs=model_config["epochs"], verbose=2, callbacks=[callback])

    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(fit_history.history['accuracy'])
    plt.plot(fit_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.show()

    print("\n")
    print("Using %d samples for training and %d for validation" % (len(all_data) - validation_n, validation_n))
    print("\n")

    model.save("myModel")

