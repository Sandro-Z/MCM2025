import pandas as pd

# 读取CSV文件
file_path = 'summerOly_athletes.csv'  # 这里替换成你的文件路径
df = pd.read_csv(file_path)

# 查看数据的前几行，检查数据格式
print(df.head())

# 提取相关列：年份、运动员、奖牌、国家
# 假设第1列是运动员名字，第四列是国家，第N列为奖牌信息（可能是金、银、铜或者No Medal）
df = df[['Name', 'Year', 'NOC', 'Medal']]  # 请根据实际列名修改
df['Medal'] = df['Medal'].fillna('No Medal')  # 将空值填充为"No Medal"

# 创建一个新列表示金、银、铜奖牌
df['Gold'] = df['Medal'].apply(lambda x: 1 if x == 'Gold' else 0)
df['Silver'] = df['Medal'].apply(lambda x: 1 if x == 'Silver' else 0)
df['Bronze'] = df['Medal'].apply(lambda x: 1 if x == 'Bronze' else 0)

# 按年份、国家、运动员进行分组，并统计每位运动员获得的金、银、铜奖牌数
result = df.groupby(['Year', 'NOC', 'Name']).agg(
    Gold=('Gold', 'sum'),
    Silver=('Silver', 'sum'),
    Bronze=('Bronze', 'sum')
).reset_index()

# 添加总奖牌数列
result['Total Medals'] = result['Gold'] + result['Silver'] + result['Bronze']

# 按照要求的列顺序重新排序
result = result[['Year', 'NOC', 'Name', 'Gold', 'Silver', 'Bronze', 'Total Medals']]

# 输出处理后的结果
print(result)

# 保存到新的CSV文件
result.to_csv('processed_athletes_data.csv', index=False)
