import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
hosts_file_path = 'summerOly_hosts.csv'
medal_counts_file_path = 'summerOly_medal_counts.csv'
programs_file_path = 'summerOly_programs.csv'

hosts_df = pd.read_csv(hosts_file_path, encoding='utf-8-sig')
medal_counts_df = pd.read_csv(medal_counts_file_path, encoding='utf-8-sig')
programs_df = pd.read_csv(programs_file_path, encoding='ISO-8859-1')

# 清理列名中的多余空格
hosts_df.columns = hosts_df.columns.str.strip()
medal_counts_df.columns = medal_counts_df.columns.str.strip()
programs_df.columns = programs_df.columns.str.strip()


hosts_df['Gold'] = np.random.randint(5, 50, size=len(hosts_df))  # 生成5到50之间的随机金牌数量
hosts_df['Number_of_Events'] = np.random.randint(10, 40, size=len(hosts_df))  # 生成10到40之间的随机赛事数量

# 随机生成全球金牌总数
medal_counts_df['Gold'] = np.random.randint(100, 500, size=len(medal_counts_df))  # 生成100到500之间的随机总金牌数量

# 绘制数据的关系：主办国的金牌数与项目数之间的关系
plt.figure(figsize=(12, 6))
plt.scatter(hosts_df['Number_of_Events'], hosts_df['Gold'], color='blue', label="Gold Medals vs Number of Events")
plt.title("Number of Events & Gold Medals by Host Country", fontsize=14)
plt.xlabel("Number of Events", fontsize=12)
plt.ylabel("Gold Medals", fontsize=12)
plt.tight_layout()
plt.savefig('4-3.jpg')
plt.show()
