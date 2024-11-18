import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model

from network.data_utils import get_raw_data_set

# модель та категорії (класи) зображень
model = None
subCategory = None


# функція для зміни розміру зображення
def resize_test_image(test_img):
    img = cv2.imread(test_img)
    # plt.imshow(img)
    # plt.show()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img_RGB)
    # plt.show()
    resized_img = cv2.resize(img_RGB, (32, 32))
    # plt.imshow(resized_img)
    # plt.show()
    resized_img = resized_img / 255.
    # plt.imshow(resized_img)
    # plt.show()
    return resized_img


# функція для отримання прогнозу для тестового зображення з моделі
def predict_test_image(test_img):
    resized_img = resize_test_image(test_img)
    prediction = model.predict(np.array([resized_img]))
    return prediction


# функція для отримання відсортованого прогнозу
def sort_prediction_test_image(test_img):
    prediction = predict_test_image(test_img)
    index = np.arange(0, 100)
    for i in range(100):
        for j in range(100):
            if prediction[0][index[i]] > prediction[0][index[j]]:
                temp = index[i]
                index[i] = index[j]
                index[j] = temp
    return index


# функція для отримання фрейму даних для 5 найкращих прогнозів
def df_top5_prediction_test_image(test_img):
    sorted_index = sort_prediction_test_image(test_img)
    prediction = predict_test_image(test_img)
    subCategory_name = []
    prediction_score = []
    k = sorted_index[:6]
    for i in range(len(k)):
        subCategory_name.append(subCategory.iloc[k[i]][0])
        prediction_score.append(round(prediction[0][k[i]], 2))

    df = pd.DataFrame(list(zip(subCategory_name, prediction_score)), columns=['Label', 'Probability'])
    return df


# функція для отримання графіку для 5 найкращих прогнозів
def plot_top5_prediction_test_image(test_img):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle("Prediction", fontsize=18)
    new_img = plt.imread(test_img)
    axes[0].imshow(new_img)
    axes[0].axis('off')
    data = df_top5_prediction_test_image(test_img)
    x = data['Label']
    y = data['Probability']
    axes[1] = sns.barplot(x=x, y=y, data=data, color="green")
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.0)
    axes[1].grid(False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    plt.show()


# завантаження моделі з файлу
model = load_model('models/best_model.h5')

# завантаження мета даних з файлу та назв категорій (класів) зображень
metaData = get_raw_data_set('data/meta')
subCategory = pd.DataFrame(metaData['fine_label_names'], columns=['SubClass'])

# розпізнання зображення та показ результатів
plot_top5_prediction_test_image('img/orange.png')
plot_top5_prediction_test_image('img/orchid.png')
plot_top5_prediction_test_image('img/cat.png')
plot_top5_prediction_test_image('img/lion.png')
plot_top5_prediction_test_image('img/clock.jpg')
plot_top5_prediction_test_image('img/bottle.jpg')
