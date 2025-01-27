import pandas as pd

# 加载数据文件 (假设文件名为 'summerOly_medal_counts.csv')
data = pd.read_csv('summerOly_medal_counts.csv')

# 统计每个国家在历届奥运会中没有金牌的次数
no_gold_counts = data[data['Gold'] == 0].groupby('NOC').size()

# 输出没有金牌的国家及其次数
print(no_gold_counts)

# 保存结果到新的CSV文件
no_gold_counts.to_csv('no_gold_counts.csv', header=['No_Gold_Counts'], index=True)

# 读取数据
medal_counts_df = pd.read_csv('summerOly_medal_counts.csv', encoding='ISO-8859-1')

# 筛选出未获得金牌的国家
countries_no_gold = medal_counts_df[medal_counts_df['Gold'] == 0]['NOC'].unique()

# 筛选出这些国家的历年数据
no_gold_data = medal_counts_df[medal_counts_df['NOC'].isin(countries_no_gold)].copy()

# 计算这些国家历年的奖牌趋势
no_gold_data.loc[:, 'trend'] = no_gold_data.groupby('NOC')['Total'].diff().fillna(0)


gold_probabilities = no_gold_data.groupby('NOC')['trend'].apply(lambda x: np.mean(x > 0))


gold_probabilities = gold_probabilities.clip(0, 1)


gold_probabilities = gold_probabilities.sort_values(ascending=False)


top_countries = gold_probabilities.head(10)


plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='skyblue', edgecolor='black')

# 设置标题和标签
plt.title("Top 10 Countries Likely to Win Gold Medals in 2028", fontsize=14)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Probability", fontsize=12)

# 显示图形
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
