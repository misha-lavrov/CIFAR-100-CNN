import keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

from network.data_utils import get_raw_data_set, preprocess_data
from network.model_utils import build_model

##### Оголошення змінних та початкових даних

# Кількість класів у наборі даних CIFAR-100
num_class = 100
# Кількість епох для навчання мережі
epochs = 100
# Розмір партії під час навчання
batch_size = 64

# завантаження даних для тренування, тестування та метадані
trainData = get_raw_data_set('data/train')
testData = get_raw_data_set('data/test')
metaData = get_raw_data_set('data/meta')

# ініціалізація даних у потрібному для навчання моделі форматі
# х - дані про малюнок
# y - дані про відповідний клас (ярлик) малюнку
x_train, y_train = preprocess_data(trainData, num_class)
x_test, y_test = preprocess_data(trainData, num_class)

# створення моделі нейронної мережі
model = build_model(x_train.shape[1:], num_class)

##### Навчання згорткової нейронної мережі

# створення оптимізатора
optimizer = keras.optimizers.Adam(lr=0.0001)

# компіляція моделі
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# рання зупинка для моніторингу втрат підтвердження та уникнення переобладнання
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# збереження контрольної точки моделі для найкращої моделі
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

# підключення даних до моделі та початок навчання з відображенням історії
model_history = model.fit(x=x_train,
                          y=y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          callbacks=[early_stop, model_checkpoint],
                          validation_split=0.2,
                          steps_per_epoch=40000 // batch_size,
                          validation_steps=10000 // batch_size,
                          validation_batch_size=batch_size)

##### Візуалізація втрат і точності
# Графік для візуалізації втрат і точності залежно від кількості епох
plt.figure(figsize=(18, 8))
plt.suptitle('Loss and Accuracy Plots', fontsize=18)
plt.subplot(1, 2, 1)
plt.plot(model_history.history['loss'], label='Training Loss')
plt.plot(model_history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Number of epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.subplot(1, 2, 2)
plt.plot(model_history.history['accuracy'], label='Train Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Number of epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.show()
