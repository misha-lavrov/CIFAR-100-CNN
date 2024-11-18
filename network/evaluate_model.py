import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from pylab import rcParams
from sklearn.metrics import confusion_matrix, classification_report

from network.data_utils import get_raw_data_set, preprocess_data

##### Оголошення змінних та початкових даних

# Кількість класів у наборі даних CIFAR-100
num_class = 100
# Розмір партії для розпізнання
batch_size = 64
# Розмір масиву даних для розпізнання
# повинен бути меншим за 10000 для test, або меншим за 50000 для train)
data_set_size = 10000

# завантаження даних з файлу
testData = get_raw_data_set('data/test')
metaData = get_raw_data_set('data/meta')

# ініціалізація даних у потрібному для моделі форматі
# х - дані про малюнок
# y - дані про відповідний клас (ярлик) малюнку
x_test_full, y_test_full = preprocess_data(testData, num_class)

# завантаження моделі з файлу
model = load_model('models/best_model.h5')

# завантаження категорій (класів) зображень
category = pd.DataFrame(metaData['coarse_label_names'], columns=['SuperClass'])
subCategory = pd.DataFrame(metaData['fine_label_names'], columns=['SubClass'])

# беремо частину даних з масиву
x_test = x_test_full[:data_set_size]
y_test = y_test_full[:data_set_size]
fine_labels = testData['fine_labels'][:data_set_size]

# визначення точності розпізнання даних
test_loss, test_accuracy = model.evaluate(x=x_test,
                                          y=y_test,
                                          batch_size=batch_size,
                                          steps=data_set_size // batch_size)

print('Accuracy: ', round((test_accuracy * 100), 2), "%")
print('Loss: ', round(test_loss, 2))

# запускаємо модель для розпізнання даних
y_pred = model.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(cm)

# звіт, щоб побачити, яку категорію було передбачено неправильно, а яку – правильно
target = ["Category {}".format(i) for i in range(num_class)]
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=target))

##### Візуалізація прогнозів
prediction = np.argmax(y_pred, axis=1)
prediction = pd.DataFrame(prediction)

# генерування випадкового числа для відображення випадкового зображення з набору даних разом із істинною та прогнозованою міткою
# 16 випадкових зображень для одночасного відображення разом із їхніми справжніми та випадковими мітками
rcParams['figure.figsize'] = 12, 15

num_row = 4
num_col = 4

imageId = np.random.randint(0, len(x_test), num_row * num_col)

fig, axes = plt.subplots(num_row, num_col)

for i in range(0, num_row):
    for j in range(0, num_col):
        k = (i * num_col) + j
        axes[i, j].imshow(x_test[imageId[k]])
        axes[i, j].set_title("True: " + str(subCategory.iloc[fine_labels[imageId[k]]][0]).capitalize()
                             + "\nPredicted: " + str(subCategory.iloc[prediction.iloc[imageId[k]]]).split()[
                                 2].capitalize(),
                             fontsize=14)
        axes[i, j].axis('off')
        fig.suptitle("Images with True and Predicted Labels", fontsize=18)

plt.show()
print()
