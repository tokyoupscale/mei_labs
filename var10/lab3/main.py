import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# цель - узнать линейную зависимость цены от года выпуска автомобиля, вметсимости двигателя и пробега
df = pd.read_csv("adsales.csv", delimiter=",")
df = df.drop('Unnamed: 0', axis=1)

# смотрим датасетик 
print(df.head(), 
    df.shape, 
    df.isna().sum()
    )

# columns = [
#     'TV Ad Budget ($)',
#     'Radio Ad Budget ($)',
#     'Newspaper Ad Budget ($)',
#     'Sales ($)'
# ]

# linear (y = a + bX)

# x - все признаки
# y - ключевая переменная 

x = df[['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']]
y = df['Sales ($)']

# # распределение переменных
df.hist(bins=42, edgecolor='black')
plt.suptitle("распределения переменных датасета", fontsize=18)
plt.show()

# парная корелляция
sns.pairplot(data=df, diag_kind='kde')
plt.suptitle("парная корелляция", fontsize=16)
plt.show()

#хитмап
corellation = df.corr()
sns.heatmap(corellation, annot=True, cmap='coolwarm', center=0)
plt.title("корреляционная матрица", fontsize=16)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("количество данных для обучения:", x_train.shape)
print("количество данных для теста:", x_test.shape)

model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

mae = mean_absolute_error(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
r2 = r2_score(y_test, y_predict)
print(f"MAE:  {mae:.4f}") # 1.4608
print(f"RMSE: {rmse:.4f}") # 1.7816
print(f"R2:   {r2:.4f}") # 0.8994

for feature in x_test.columns:
    plt.figure(figsize=(10, 6))

    if x_test[feature].dtype in ['float64', 'int64']:
        plt.scatter(x_test[feature], y_test, color='blue', alpha=0.5, label='фактические данные')

        plt.scatter(x_test[feature], y_predict, color='red', alpha=0.5, label='предсказанные данные')

        plt.xlabel(feature)
        plt.ylabel('количество продаж')
        plt.title(f'зависимость продаж от {feature}')
        plt.legend()
        plt.show()

residuals = y_test - y_predict

sns.histplot(residuals, kde=True)

# Заголовок графика
plt.title('распределение ошибок')
plt.xlabel('ошибки')
plt.ylabel('частота')
plt.grid(True)
plt.show()
