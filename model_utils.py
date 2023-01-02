import keras


def run_inference(model, X, y, dims):
    y_actual = []
    y_prediction = []
    y_prediction_probabilities = []
    for count, x in enumerate(X):
        # groundtruths
        y_actual.append(np.argmax(y[count]))

        # predictions
        prediction = model.predict(x.reshape(-1, *dims))
        y_prediction_probabilities.append(prediction[0].tolist())
        y_prediction.append(np.argmax(prediction[0]))

    return y_actual, y_prediction_probabilities, y_prediction


def load_model_from_weights(WEIGHTS_PATH):
    model = keras.models.load_model(WEIGHTS_PATH)
    return model