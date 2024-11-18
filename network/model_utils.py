from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential


# Функція для побудови згорткової нейронної мережі
def build_model(input_shape, num_class):
    # Ініціалізація моделі послідовного типу (має лінійний стек шарів)
    model = Sequential()

    # Стек 1
    # Згорткові шари (convolution)
    model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
    # Агрегувальні шари (pooling)
    model.add(MaxPool2D(pool_size=2, strides=2))
    # Виключення з'єднань (dropout)
    model.add(Dropout(0.2))

    # Стек 2
    # Згорткові шари (convolution)
    model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
    # Агрегувальні шари (pooling)
    model.add(MaxPool2D(pool_size=2, strides=2))
    # Виключення з'єднань (dropout)
    model.add(Dropout(0.5))

    # Стек 3
    # Згорткові шари (convolution)
    model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
    # Агрегувальні шари (pooling)
    model.add(MaxPool2D(pool_size=2, strides=2))
    # Виключення з'єднань (dropout)
    model.add(Dropout(0.5))

    # Згладжування/розрівняння (flattening)
    model.add(Flatten())
    # Повноз'єднані шари (fully connected layers)
    model.add(Dense(units=1000, activation="relu"))
    # Виключення з'єднань (dropout)
    model.add(Dropout(0.5))
    # Повноз'єднані шари (fully connected layers)
    model.add(Dense(units=1000, activation="relu"))
    # Виключення з'єднань (dropout)
    model.add(Dropout(0.5))
    # Вихідний шар (output layer)
    model.add(Dense(units=num_class, activation="softmax"))
    model.summary()
    return model
