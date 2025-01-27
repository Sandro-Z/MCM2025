import pandas as pd
import matplotlib.pyplot as plt


programs_file_path = 'summerOly_programs.csv'
medal_counts_file_path = 'summerOly_medal_counts.csv'
programs_df = pd.read_csv(programs_file_path, encoding='ISO-8859-1')
medal_counts_df = pd.read_csv(medal_counts_file_path, encoding='ISO-8859-1')
programs_df_cleaned = programs_df.loc[:, ~programs_df.columns.str.contains("Code|Sport|Discipline|Sports Governing Body")]
programs_df_cleaned = programs_df_cleaned.rename(columns=lambda x: x.strip().replace('*', '') if isinstance(x, str) else x)
programs_df_cleaned = programs_df_cleaned.apply(pd.to_numeric, errors='coerce')
programs_melted_cleaned = programs_df_cleaned.melt(var_name='Year', value_name='Number_of_Events')
programs_melted_cleaned['Year'] = pd.to_numeric(programs_melted_cleaned['Year'], errors='coerce', downcast='integer')
medal_counts_per_year = medal_counts_df.groupby(['Year', 'NOC'])['Gold'].sum().reset_index()
merged_data_cleaned = pd.merge(medal_counts_per_year, programs_melted_cleaned, how='left', on='Year')
yearly_summary = merged_data_cleaned.groupby('Year').agg({
    'Number_of_Events': 'sum',  # 每年项目数的总和
    'Gold': 'sum'               # 每年金牌数的总和
}).reset_index()

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(yearly_summary['Year'], yearly_summary['Number_of_Events']/50, label="Number of Events", marker='o', color='b')
plt.plot(yearly_summary['Year'], yearly_summary['Gold']/50, label="Gold Medals", marker='o', color='g')

# 设置标题和标签
plt.title("Gold medals and Events", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(title="Legend")
plt.savefig('4-1.jpg')
# 显示图表
plt.tight_layout()
plt.show()
