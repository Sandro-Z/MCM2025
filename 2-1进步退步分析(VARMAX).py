import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# 加载数据
athletes = pd.read_csv('summerOly_athletes.csv', encoding='ISO-8859-1')
medal_counts = pd.read_csv('summerOly_medal_counts.csv', encoding='ISO-8859-1')

# 数据预处理
medal_summary = medal_counts.groupby(['Year', 'NOC']).agg(
    {'Gold': 'sum', 'Silver': 'sum', 'Bronze': 'sum', 'Total': 'sum'}).reset_index()
athlete_summary = athletes.groupby(['Year', 'NOC']).size().reset_index(name='Athletes')

data = pd.merge(medal_summary, athlete_summary, on=['Year', 'NOC'], how='left')
data = data.fillna(0)

# 添加历史奖牌特征
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

# 替换线性回归为 ARIMA 模型
def build_model(data, features, target):
    unique_nocs = data['NOC'].unique()
    predictions = []

    # 遍历每个国家，构建时间序列模型
    for noc in unique_nocs:
        country_data = data[data['NOC'] == noc][['Year', target]].set_index('Year')

        # 转换时间索引为 DateTime 格式并设置频率
        country_data.index = pd.to_datetime(country_data.index, format='%Y')
        country_data = country_data.asfreq('4YS')  # 奥运会每4年一次

        # 检查数据长度，确保有足够的时间点构建模型
        if len(country_data) < 3:
            predictions.append(0)  # 数据不足返回0预测
            continue

        # 构建 ARIMA 模型
        try:
            series = country_data[target]
            #model=ARIMA(series,order=(5,2,0))
            model = VARMAX(series, order=(1, 1))  # 简单 ARIMA(1,1,0) 模型
            model_fit = model.fit(maxiter=1000,disp=False)
            print(model_fit.summary())
            forecast = model_fit.forecast(steps=1)  # 预测未来1步
            predictions.append(forecast[0])  # 保存预测结果
        except Exception as e:
            print(f"Error processing NOC {noc}: {e}")
            predictions.append(0)  # 如果建模失败，返回0预测

    return np.array(predictions)

# 替换线性回归模型部分
unique_nocs = data['NOC'].unique()
future_data = pd.DataFrame({
    'NOC': unique_nocs,
    'Past_Gold_Avg': data[data['Year'] == 2024].groupby('NOC')['Past_Gold_Avg'].mean().reindex(unique_nocs).fillna(0).values,
    'Past_Total_Avg': data[data['Year'] == 2024].groupby('NOC')['Past_Total_Avg'].mean().reindex(unique_nocs).fillna(0).values,
    'Past_Gold_Sum': data[data['Year'] == 2024].groupby('NOC')['Past_Gold_Sum'].mean().reindex(unique_nocs).fillna(0).values,
    'Past_Total_Sum': data[data['Year'] == 2024].groupby('NOC')['Past_Total_Sum'].mean().reindex(unique_nocs).fillna(0).values
})

future_data['Predicted_Gold'] = build_model(data, features, 'Gold')
future_data['Predicted_Total'] = build_model(data, features, 'Total')

# 处理 NaN 和 inf 值后转换为整数
future_data['Predicted_Gold'] = future_data['Predicted_Gold'].fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0).round().astype(int)
future_data['Predicted_Total'] = future_data['Predicted_Total'].fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0).round().astype(int)

# 计算预测区间
gold_std = np.std(future_data['Predicted_Gold'])
total_std = np.std(future_data['Predicted_Total'])

future_data['Gold_Lower'] = np.ceil(future_data['Predicted_Gold'] - 2 * gold_std).clip(lower=0).astype(int)
future_data['Gold_Upper'] = np.floor(future_data['Predicted_Gold'] + 2 * gold_std).clip(lower=0).astype(int)
future_data['Total_Lower'] = np.ceil(future_data['Predicted_Total'] - 2 * total_std).clip(lower=0).astype(int)
future_data['Total_Upper'] = np.floor(future_data['Predicted_Total'] + 2 * total_std).clip(lower=0).astype(int)

# 输出预测结果
future_data = future_data.sort_values(by='Predicted_Total', ascending=False)
print("Predicted Medal Table for 2028 Los Angeles Olympics:")
print(
    future_data[['NOC', 'Predicted_Gold', 'Gold_Lower', 'Gold_Upper', 'Predicted_Total', 'Total_Lower', 'Total_Upper']])

# 保存结果到 CSV 文件
future_data.to_csv('Predicted_Medal_Table_2028.csv', index=False)
print("Results saved to 'Predicted_Medal_Table_2028.csv'")
'''
plt.figure(figsize=(12, 8))
sns.barplot(x='Total_Progress', y='NOC', data=progress_total.head(10), palette='Blues')
plt.title('Top 10 Countries Likely to Improve in Total Medals by 2028')
plt.xlabel('Total Medal Progress')
plt.ylabel('Country')
plt.show()
'''
file_path = 'Predicted_Medal_Table_2028.csv'
future_data = pd.read_csv(file_path)


# 计算进步与退步
future_data['Gold_Progress'] = future_data['Predicted_Gold'] - future_data['Past_Gold_Avg']
future_data['Total_Progress'] = future_data['Predicted_Total'] - future_data['Past_Total_Avg']

# 标记进步和退步的国家
future_data['Gold_Trend'] = future_data['Gold_Progress'].apply(lambda x: 'Progress' if x > 0 else 'Decline')
future_data['Total_Trend'] = future_data['Total_Progress'].apply(lambda x: 'Progress' if x > 0 else 'Decline')

# 按金牌进步最多排序
progress_gold = future_data[['NOC', 'Predicted_Gold', 'Gold_Progress', 'Gold_Trend']].sort_values(
    by='Gold_Progress', ascending=False)
decline_gold = future_data[['NOC', 'Predicted_Gold', 'Gold_Progress', 'Gold_Trend']].sort_values(
    by='Gold_Progress', ascending=True)

# 按总奖牌进步最多排序
progress_total = future_data[['NOC', 'Predicted_Total', 'Total_Progress', 'Total_Trend']].sort_values(
    by='Total_Progress', ascending=False)
decline_total = future_data[['NOC', 'Predicted_Total', 'Total_Progress', 'Total_Trend']].sort_values(
    by='Total_Progress', ascending=True)

file_path = 'Predicted_Medal_Table_2028.csv'
f=open(file_path,'r')
txt=f.readlines()
f.close()
forbidden_country=['Soviet Union','Yugoslavia','West Germany','East Germany','Â ','Mixed team','Russia','Â',' ']
for line in txt:
    for country in forbidden_country:
        if country in line:
            txt.remove(line)
g=open(file_path,'w')
for line in txt:
    g.write(line)
g.close()

file_path = 'Predicted_Medal_Table_2028.csv'
future_data = pd.read_csv(file_path)

# 计算进步与退步
future_data['Gold_Progress'] = future_data['Predicted_Gold'] - future_data['Past_Gold_Avg']
future_data['Total_Progress'] = future_data['Predicted_Total'] - future_data['Past_Total_Avg']

# 标记进步和退步的国家
future_data['Gold_Trend'] = future_data['Gold_Progress'].apply(lambda x: 'Progress' if x > 0 else 'Decline')
future_data['Total_Trend'] = future_data['Total_Progress'].apply(lambda x: 'Progress' if x > 0 else 'Decline')

# 按金牌进步最多排序
progress_gold = future_data[['NOC', 'Predicted_Gold', 'Gold_Progress', 'Gold_Trend']].sort_values(
    by='Gold_Progress', ascending=False)
decline_gold = future_data[['NOC', 'Predicted_Gold', 'Gold_Progress', 'Gold_Trend']].sort_values(
    by='Gold_Progress', ascending=True)

# 按总奖牌进步最多排序
progress_total = future_data[['NOC', 'Predicted_Total', 'Total_Progress', 'Total_Trend']].sort_values(
    by='Total_Progress', ascending=False)
decline_total = future_data[['NOC', 'Predicted_Total', 'Total_Progress', 'Total_Trend']].sort_values(
    by='Total_Progress', ascending=True)

print(progress_gold)
print(decline_gold)
print(progress_total)
print(decline_total)
# 绘图功能

# 金牌数进步最多的国家
plt.figure(figsize=(12, 8))

sns.barplot(x=-decline_gold['Gold_Progress'].head(10), y='NOC', data=decline_gold.head(10), palette='Blues_r')
plt.title('Top 10 Countries Likely to Improve in Gold Medals by 2028(VARMAX)')
plt.xlabel('Gold Medal Progress')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('2-5.jpg')
plt.show()

# 金牌数退步最多的国家
plt.figure(figsize=(12, 8))
sns.barplot(x=progress_gold['Gold_Progress'].head(10), y='NOC', data=progress_gold.head(10), palette='Reds_r')
plt.title('Top 10 Countries Likely to Decline in Gold Medals by 2028')
plt.xlabel('Gold Medal Decline')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('2-2.jpg')
plt.show()

# 总奖牌数进步最多的国家
plt.figure(figsize=(12, 8))
sns.barplot(x=-decline_total['Total_Progress'].head(10), y='NOC', data=decline_total.head(10), palette='Blues_r')
plt.title('Top 10 Countries Likely to Improve in Total Medals by 2028(VARMAX)')
plt.xlabel('Total Medal Progress')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('2-6.jpg')
plt.show()

# 总奖牌数退步最多的国家
plt.figure(figsize=(12, 8))
sns.barplot(x=progress_total['Total_Progress'].head(10), y='NOC', data=progress_total.head(10), palette='Reds')
plt.title('Top 10 Countries Likely to Decline in Total Medals by 2028')
plt.xlabel('Total Medal Decline')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('2-4.jpg')
plt.show()
