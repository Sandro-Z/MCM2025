import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


athletes = pd.read_csv('summerOly_athletes.csv', encoding='ISO-8859-1')
medal_counts = pd.read_csv('summerOly_medal_counts.csv', encoding='ISO-8859-1')


medal_summary = medal_counts.groupby(['Year', 'NOC']).agg(
    {'Gold': 'sum', 'Silver': 'sum', 'Bronze': 'sum', 'Total': 'sum'}).reset_index()


athlete_summary = athletes.groupby(['Year', 'NOC']).size().reset_index(name='Athletes')


data = pd.merge(medal_summary, athlete_summary, on=['Year', 'NOC'], how='left')
data = data.fillna(0)

# 4. 添加历史奖牌特征
# 计算每个国家过去3届奥运会的平均奖牌数和总奖牌数
data['Past_Gold_Avg'] = data.groupby('NOC')['Gold'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).mean())
data['Past_Total_Avg'] = data.groupby('NOC')['Total'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).mean())
data['Past_Gold_Sum'] = data.groupby('NOC')['Gold'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).sum())
data['Past_Total_Sum'] = data.groupby('NOC')['Total'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).sum())


data[['Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum']] = data[[
    'Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum']].fillna(0)


features = ['Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum']
target_gold = 'Gold'
target_total = 'Total'


# 构建模型
def build_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Model for {target}: MSE = {mse}')

    return model


# 训练模型
model_gold = build_model(data, features, target_gold)
model_total = build_model(data, features, target_total)

# 准备2028年的数据
unique_nocs = data['NOC'].unique()
future_data = pd.DataFrame({
    'NOC': unique_nocs,
    'Past_Gold_Avg': data[data['Year'] == 2024].groupby('NOC')['Past_Gold_Avg'].mean().reindex(unique_nocs).fillna(
        0).values,
    'Past_Total_Avg': data[data['Year'] == 2024].groupby('NOC')['Past_Total_Avg'].mean().reindex(unique_nocs).fillna(
        0).values,
    'Past_Gold_Sum': data[data['Year'] == 2024].groupby('NOC')['Past_Gold_Sum'].mean().reindex(unique_nocs).fillna(
        0).values,
    'Past_Total_Sum': data[data['Year'] == 2024].groupby('NOC')['Past_Total_Sum'].mean().reindex(unique_nocs).fillna(
        0).values
})

# 预测奖牌数
future_data['Predicted_Gold'] = model_gold.predict(future_data[features])
future_data['Predicted_Total'] = model_total.predict(future_data[features])


future_data['Predicted_Gold'] = future_data['Predicted_Gold'].clip(lower=0).round().astype(int)
future_data['Predicted_Total'] = future_data['Predicted_Total'].clip(lower=0).round().astype(int)


gold_std = np.std(model_gold.predict(data[features]) - data[target_gold])
total_std = np.std(model_total.predict(data[features]) - data[target_total])

# 确保预测区间为非负整数
future_data['Gold_Lower'] = np.ceil(future_data['Predicted_Gold'] - 2 * gold_std).clip(lower=0).astype(int)
future_data['Gold_Upper'] = np.floor(future_data['Predicted_Gold'] + 2 * gold_std).clip(lower=0).astype(int)
future_data['Total_Lower'] = np.ceil(future_data['Predicted_Total'] - 2 * total_std).clip(lower=0).astype(int)
future_data['Total_Upper'] = np.floor(future_data['Predicted_Total'] + 2 * total_std).clip(lower=0).astype(int)

# 输出预测结果
future_data = future_data.sort_values(by='Predicted_Total', ascending=False)
print("Predicted Medal Table for 2028 Los Angeles Olympics:")
print(
    future_data[['NOC', 'Predicted_Gold', 'Gold_Lower', 'Gold_Upper', 'Predicted_Total', 'Total_Lower', 'Total_Upper']])

# 保存结果到CSV文件
future_data.to_csv('Predicted_Medal_Table_2028.csv', index=False)
print("Results saved to 'Predicted_Medal_Table_2028.csv'")

'''
# 可视化预测的奖牌榜
plt.figure(figsize=(12, 8))
sns.barplot(x='Predicted_Total', y='NOC', data=future_data.head(10), palette='viridis')
plt.title('Predicted Medal Table for 2028 Los Angeles Olympics')
plt.xlabel('Predicted Total Medals')
plt.ylabel('Country')
plt.show()
'''

# 计算历史和预测的奖牌数差异
future_data['Gold_History_Avg'] = data.groupby('NOC')['Past_Gold_Avg'].mean().reindex(unique_nocs).fillna(0).values
future_data['Total_History_Avg'] = data.groupby('NOC')['Past_Total_Avg'].mean().reindex(unique_nocs).fillna(0).values

# 计算进步/退步
future_data['Gold_Progress'] = future_data['Predicted_Gold'] - future_data['Gold_History_Avg']
future_data['Total_Progress'] = future_data['Predicted_Total'] - future_data['Total_History_Avg']

# 标记进步和退步的国家
future_data['Gold_Trend'] = np.where(future_data['Gold_Progress'] > 0, 'Progress', 'Decline')
future_data['Total_Trend'] = np.where(future_data['Total_Progress'] > 0, 'Progress', 'Decline')

# 按照金牌数的进步排序
progress_gold = future_data[['NOC', 'Predicted_Gold', 'Gold_Progress', 'Gold_Trend']].sort_values(by='Gold_Progress', ascending=False)
decline_gold = future_data[['NOC', 'Predicted_Gold', 'Gold_Progress', 'Gold_Trend']].sort_values(by='Gold_Progress', ascending=True)

# 按照总奖牌数的进步排序
progress_total = future_data[['NOC', 'Predicted_Total', 'Total_Progress', 'Total_Trend']].sort_values(by='Total_Progress', ascending=False)
decline_total = future_data[['NOC', 'Predicted_Total', 'Total_Progress', 'Total_Trend']].sort_values(by='Total_Progress', ascending=True)

# 输出最有可能进步的国家
print("Countries Most Likely to Improve in Gold Medals:")
print(progress_gold[['NOC', 'Predicted_Gold', 'Gold_Progress', 'Gold_Trend']].head(10))

print("\nCountries Most Likely to Decline in Gold Medals:")
print(decline_gold[['NOC', 'Predicted_Gold', 'Gold_Progress', 'Gold_Trend']].head(10))

# 输出最有可能进步的国家（按总奖牌数）
print("\nCountries Most Likely to Improve in Total Medals:")
print(progress_total[['NOC', 'Predicted_Total', 'Total_Progress', 'Total_Trend']].head(10))

print("\nCountries Most Likely to Decline in Total Medals:")
print(decline_total[['NOC', 'Predicted_Total', 'Total_Progress', 'Total_Trend']].head(10))

# 可视化金牌数进步的国家
plt.figure(figsize=(12, 8))
sns.barplot(x='Gold_Progress', y='NOC', data=progress_gold.head(10), palette='Blues')
plt.title('Top 10 Countries Likely to Improve in Gold Medals by 2028')
plt.xlabel('Gold Medal Progress')
plt.ylabel('Country')
plt.show()

# 可视化金牌数退步的国家
plt.figure(figsize=(12, 8))
sns.barplot(x='Gold_Progress', y='NOC', data=decline_gold.head(10), palette='Reds')
plt.title('Top 10 Countries Likely to Decline in Gold Medals by 2028')
plt.xlabel('Gold Medal Decline')
plt.ylabel('Country')
plt.show()

# 可视化总奖牌数进步的国家
plt.figure(figsize=(12, 8))
sns.barplot(x='Total_Progress', y='NOC', data=progress_total.head(10), palette='Blues')
plt.title('Top 10 Countries Likely to Improve in Total Medals by 2028')
plt.xlabel('Total Medal Progress')
plt.ylabel('Country')
plt.show()

# 可视化总奖牌数退步的国家
plt.figure(figsize=(12, 8))
sns.barplot(x='Total_Progress', y='NOC', data=decline_total.head(10), palette='Reds')
plt.title('Top 10 Countries Likely to Decline in Total Medals by 2028')
plt.xlabel('Total Medal Decline')
plt.ylabel('Country')
plt.show()

