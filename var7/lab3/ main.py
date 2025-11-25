# Air Quality Dataset: 
# прогнозирование уровня загрязнения воздуха на основе метеоданных и времени

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# y = CO(GT)
# x = T/RH/AH

# цель - узнать линейную зависимость качества воздуха от температуры etc
df = pd.read_csv("AirQuality.csv", delimiter=";", na_values=-200)
print(df.head(), df.shape, df.isna().sum())

df['DateTime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'],
    format='%d/%m/%Y %H.%M.%S'
)

for col in ['RH', 'T', 'AH', 'CO(GT)', 'C6H6(GT)']: 
    df[col] = (
        df[col]
        .replace(',', '.', regex=True)
        .astype(float)
    )

df = df.set_index('DateTime')
df = df.drop(columns=['Date', 'Time'])

df = df.replace(-200, np.nan)
df = df.dropna(how="all", axis=1)
df = df.dropna(how="all", axis=0)

def get_season(month):
    if month in [12,1,2]: return 0
    if month in [3,4,5]: return 1
    if month in [6,7,8]: return 2
    return 3

df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
df['season'] = df.index.month.map(get_season)

df_model = df[['CO(GT)', 'T', 'RH', 'AH', 'hour', 'weekday', 'season']].dropna()

X = df_model[['T', 'RH', 'AH', 'hour', 'weekday', 'season']]
y = df_model['CO(GT)']

# я хз че это делает vibecoded ngl
df.hist(bins=30, figsize=(16, 12), color='skyblue', edgecolor='black')
plt.suptitle("распределения переменных датасета", fontsize=18)
plt.show()

sns.pairplot(df[['CO(GT)', 'T', 'RH', 'AH']], diag_kind='kde')
plt.suptitle("парная корелляция", y=1.02, fontsize=16)
plt.show()

corr = df[['CO(GT)', 'T', 'RH', 'AH']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("корреляционная матрица", fontsize=16)
plt.show()

print("корреляции CO(GT) с признаками:\n")
print(corr['CO(GT)'].sort_values(ascending=False))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("количество данных для обучения:", X_train.shape)
print("количество данных для теста:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# график как модель предсказывает co(gt)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("истинные CO(GT)")
plt.ylabel("предсказанные CO(GT)")
plt.title("сравнение истинных и предсказанных CO(GT)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# ошибки модели
errors = y_test - y_pred

plt.figure(figsize=(8, 6))
sns.histplot(errors, kde=True, bins=30, color='purple')
plt.xlabel("ошибка предсказания")
plt.title("распределение ошибок")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:200], label='true')
plt.plot(y_pred[:200], label='Предсказания')
plt.xlabel("индекс наблюдения")
plt.ylabel("CO(GT)")
plt.title("истинные/предсказанные (200 наблюдений)")
plt.legend()
plt.show()

# MAE:  0.9881 
# RMSE: 1.2646
# R2:   0.1806 чето очень мало гпт сказал это датасет говно
