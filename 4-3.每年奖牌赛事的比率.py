import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
programs_file_path = 'summerOly_programs.csv'
medal_counts_file_path = 'summerOly_medal_counts.csv'

programs_df = pd.read_csv(programs_file_path, encoding='ISO-8859-1')
medal_counts_df = pd.read_csv(medal_counts_file_path, encoding='ISO-8859-1')

# 清理数据，去除非数字列
programs_df_cleaned = programs_df.loc[:, ~programs_df.columns.str.contains("Code|Sport|Discipline|Sports Governing Body")]

# 去除特殊字符年份列，例如 "1906*"
programs_df_cleaned = programs_df_cleaned.rename(columns=lambda x: x.strip().replace('*', '') if isinstance(x, str) else x)

# 将所有年份列转换为数字类型
programs_df_cleaned = programs_df_cleaned.apply(pd.to_numeric, errors='coerce')

# 重新整理数据为"年份"和"每个运动项目的赛事数"
programs_melted_cleaned = programs_df_cleaned.melt(var_name='Year', value_name='Number_of_Events')

# 将年份列转换为整数类型
programs_melted_cleaned['Year'] = pd.to_numeric(programs_melted_cleaned['Year'], errors='coerce', downcast='integer')

# 计算每年每个国家的金牌数量
medal_counts_per_year = medal_counts_df.groupby(['Year', 'NOC'])['Gold'].sum().reset_index()

# 合并每年各国金牌数量与项目数量
merged_data_cleaned = pd.merge(medal_counts_per_year, programs_melted_cleaned, how='left', on='Year')

# 计算每年金牌数与项目数的比例
yearly_summary = merged_data_cleaned.groupby('Year').agg({
    'Number_of_Events': 'sum',  # 每年项目数的总和
    'Gold': 'sum'               # 每年金牌数的总和
}).reset_index()

# 计算每年金牌与赛事的比率
yearly_summary['Gold_to_Event_Ratio'] = yearly_summary['Gold'] / yearly_summary['Number_of_Events']

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(yearly_summary['Year'], yearly_summary['Gold_to_Event_Ratio'], marker='o', color='purple')

# 设置标题和标签
plt.title("Gold/Event Ratio (1896-2024)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Gold to Event Ratio", fontsize=12)
plt.savefig('4-2.jpg')
# 显示图表
plt.tight_layout()
plt.show()
