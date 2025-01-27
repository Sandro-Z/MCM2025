import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
programs_file_path = 'summerOly_programs.csv'
medal_counts_file_path = 'summerOly_medal_counts.csv'

programs_df = pd.read_csv(programs_file_path, encoding='ISO-8859-1')
medal_counts_df = pd.read_csv(medal_counts_file_path, encoding='ISO-8859-1')

# 清理数据，去除无关列
programs_df_cleaned = programs_df.loc[:, ~programs_df.columns.str.contains("Code|Sports Governing Body")]

# 去除特殊字符年份列，例如 "1906*"
programs_df_cleaned = programs_df_cleaned.rename(columns=lambda x: x.strip().replace('*', '') if isinstance(x, str) else x)

# 将所有年份列转换为数字类型
programs_df_cleaned = programs_df_cleaned.apply(pd.to_numeric, errors='coerce')

# 重新整理数据为"年份"和"每个运动项目的赛事数"
programs_melted_cleaned = programs_df_cleaned.melt(var_name='Year', value_name='Number_of_Events')

# 将年份列转换为整数类型
programs_melted_cleaned['Year'] = pd.to_numeric(programs_melted_cleaned['Year'], errors='coerce', downcast='integer')

# 汇总每年每个国家的金牌数量
medal_counts_per_year = medal_counts_df.groupby(['Year', 'NOC'])['Gold'].sum().reset_index()
print(medal_counts_per_year)
# 将每年每个国家的金牌数量与项目数量数据合并
merged_data_cleaned = pd.merge(medal_counts_per_year, programs_melted_cleaned, how='left', on='Year')

# 还原每个项目的类别信息
programs_with_sport = programs_df[['Sport']].drop_duplicates()
programs_with_sport['index'] = programs_with_sport.index

# 合并运动项目的类别信息
merged_data_cleaned = pd.merge(merged_data_cleaned, programs_with_sport, how='left', left_index=True, right_on='index')
# 汇总每个项目的金牌数和赛事数量
project_summary = merged_data_cleaned.groupby('Sport').agg({
    'Number_of_Events': 'sum',  # 每个运动项目的赛事数
    'Gold': 'sum'               # 每个运动项目的金牌数
}).reset_index()

# 按金牌数量对项目排序
project_summary_sorted = project_summary.sort_values(by='Gold', ascending=False)
# 绘制金牌数量的排序条形图
plt.figure(figsize=(14, 8))
bars = plt.barh(project_summary_sorted['Sport'], project_summary_sorted['Gold'], color=plt.cm.viridis(project_summary_sorted['Gold'] / max(project_summary_sorted['Gold'])))
plt.xlabel("Gold Medals", fontsize=14)
plt.ylabel("Sport", fontsize=14)
plt.title("Gold Medals by Sport (Sorted)", fontsize=16)

# 增加条形图中的色彩渐变
for bar in bars:
    bar.set_edgecolor('black')

plt.tight_layout()
plt.show()

# 绘制气泡图，气泡大小表示金牌数量，颜色表示赛事数
plt.figure(figsize=(14, 8))
plt.scatter(project_summary['Number_of_Events'], project_summary['Gold'], s=project_summary['Gold'] * 500,  # 增大气泡大小
            c=project_summary['Number_of_Events'], cmap='plasma', alpha=0.6, edgecolors="w", linewidth=2)

# 设置标题和标签
plt.title("Bubble Chart: Relationship between Number of Events and Gold Medals by Sport", fontsize=16)
plt.xlabel("Number of Events", fontsize=14)
plt.ylabel("Gold Medals", fontsize=14)

# 添加色条
plt.colorbar(label="Number of Events")
plt.tight_layout()
plt.show()
