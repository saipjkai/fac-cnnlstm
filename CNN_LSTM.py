import keras
from keras import layers
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, Input, TimeDistributed
from keras.applications import ResNet50
from keras.applications.vgg16 import VGG16


def cnn_lstm_model(input_shape=(50, 128, 224, 3), output_classes=3, learning_rate=1e-4, arch="resnet"):
    if arch == "resnet":
        backend_model = ResNet50(weights="imagenet", input_shape=input_shape[1:], include_top=False)
    elif arch == "vgg":
        backend_model = VGG16(weights="imagenet", input_shape=input_shape[1:], include_top=False)

    # freeze all layers except last one
    for layer in backend_model.layers[:-1]:
        layer.trainable=False

    model = Sequential()
    input_layer = Input(shape=input_shape)
    model = TimeDistributed(backend_model)(input_layer) 
    model = TimeDistributed(Flatten())(model)
    model = LSTM(128, return_sequences=False)(model)
    model = Dropout(.5)(model)
    output_layer = Dense(output_classes, activation='softmax')(model)

    model = Model(input_layer, output_layer)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['accuracy'])

    return model