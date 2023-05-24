from tensorflow import keras


def vgg19_model(image_size, classes=2, activation="softmax"):
    return keras.applications.VGG19(
        weights=None,
        input_shape=image_size,
        classes=classes,
        classifier_activation=activation,
    )


def xception_model(image_size, classes=2, activation="softmax"):
    return keras.applications.Xception(
        weights=None,
        input_shape=image_size,
        classes=classes,
        classifier_activation=activation,
    )
