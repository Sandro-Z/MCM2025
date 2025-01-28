import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import chardet
from pylab import mpl
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


# ========= 1. 函数定义 ==========
def detect_encoding(file_path):
    """自动检测文件的编码格式"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))  # 读取前100KB进行检测
    return result['encoding']


def read_csv_with_encoding(file_path):
    """尝试使用多种编码格式读取CSV文件"""
    encodings = ['utf-8', 'latin1', 'cp1252']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"成功使用编码 '{enc}' 读取文件: {file_path}")
            return df
        except UnicodeDecodeError:
            print(f"使用编码 '{enc}' 读取文件 {file_path} 失败。尝试下一个编码...")
    # 如果所有编码都失败，使用chardet自动检测
    detected_enc = detect_encoding(file_path)
    try:
        df = pd.read_csv(file_path, encoding=detected_enc)
        print(f"成功使用检测到的编码 '{detected_enc}' 读取文件: {file_path}")
        return df
    except Exception as e:
        print(f"使用检测到的编码 '{detected_enc}' 读取文件 {file_path} 失败。错误: {e}")
        raise e


# ========= 2. 数据加载 ==========
# 2.1 读取数据字典（可选，用于了解数据）
data_dictionary_path = 'data_dictionary.csv'
try:
    df_dict = read_csv_with_encoding(data_dictionary_path)
    print("\n数据字典已加载。形状:", df_dict.shape)
    print(df_dict.head())
except Exception as e:
    print(f"无法读取数据字典文件: {e}")

# 2.2 读取历届奥运会奖牌数数据
medal_counts_path = 'summerOly_medal_counts.csv'
try:
    df_medals = read_csv_with_encoding(medal_counts_path)
    print("\n奖牌数数据已加载。形状:", df_medals.shape)
    print(df_medals.head())
except Exception as e:
    print(f"无法读取奖牌数数据文件: {e}")

# 2.3 读取主办国数据
hosts_path = 'summerOly_hosts.csv'
try:
    df_hosts = read_csv_with_encoding(hosts_path)
    print("\n主办国数据已加载。形状:", df_hosts.shape)
    print(df_hosts.head())
except Exception as e:
    print(f"无法读取主办国数据文件: {e}")

# 2.4 读取赛事项目数据
programs_path = 'summerOly_programs.csv'
try:
    df_programs = read_csv_with_encoding(programs_path)
    print("\n赛事项目数据已加载。形状:", df_programs.shape)
    print(df_programs.head())
except Exception as e:
    print(f"无法读取赛事项目数据文件: {e}")

# 2.5 读取运动员数据
athletes_path = 'summerOly_athletes.csv'
try:
    df_athletes = read_csv_with_encoding(athletes_path)
    print("\n运动员数据已加载。形状:", df_athletes.shape)
    print(df_athletes.head())
except Exception as e:
    print(f"无法读取运动员数据文件: {e}")

# ========= 3. 数据清洗与整合 ==========
# 3.1 创建国家名称到NOC代码的映射字典
country_to_noc_map = {
    'Greece': 'GRE',
    'France': 'FRA',
    'United States': 'USA',
    'United Kingdom': 'GBR',
    'Sweden': 'SWE',
    'Belgium': 'BEL',
    'Netherlands': 'NED',
    'Germany': 'GER',
    'Finland': 'FIN',
    'Australia': 'AUS',
    'Italy': 'ITA',
    'Spain': 'ESP',
    'Soviet Union': 'URS',
    'West Germany': 'FRG',
    'East Germany': 'GDR',
    'Unified Team': 'EUN',
    'ROC': 'ROC',  # Russian Olympic Committee
    'North Korea': 'PRK',
    'South Korea': 'KOR',
    'Mixed team': 'MIX',
    'Great Britain': 'GBR',
    'Iceland': 'ISL',
    'Ghana': 'GHA',
    'Iraq': 'IRQ',
    'Malaysia': 'MAS',
    'Kuwait': 'KUW',
    'Paraguay': 'PAR',
    'Sudan': 'SUD',
    'Saudi Arabia': 'KSA',
    # 根据实际数据添加更多映射
}

# 3.2 清洗主办国数据，提取国家名称并映射到NOC
df_hosts['HostCountry'] = df_hosts['Host'].str.split(',').str[-1].str.strip()
df_hosts['Host_NOC'] = df_hosts['HostCountry'].map(country_to_noc_map)
df_hosts['Host_NOC'] = df_hosts['Host_NOC'].fillna('UNK')  # 'UNK' 表示未知

print("\n主办国数据与Host_NOC:")
print(df_hosts[['Year', 'Host', 'HostCountry', 'Host_NOC']].head())

# 3.3 汇总运动员数据中的金牌数
# 3.3.1 过滤夏季奥运会数据
if 'Season' in df_athletes.columns:
    # 如果 'Season' 列存在，则进行过滤
    df_athletes_summer = df_athletes[df_athletes['Season'] == 'Summer'].copy()
    print("\n已基于 'Season' 列过滤夏季奥运会的运动员数据。")
else:
    # 如果 'Season' 列不存在，假设所有数据都是夏季奥运会
    df_athletes_summer = df_athletes.copy()
    print("\n假设所有运动员数据均来自夏季奥运会。")

# 3.3.2 过滤出金牌
df_athletes_gold = df_athletes_summer[df_athletes_summer['Medal'] == 'Gold'].copy()

# 3.3.3 按国家、运动项目、年份汇总金牌数
df_medal_counts = df_athletes_gold.groupby(['NOC', 'Sport', 'Year']).size().reset_index(name='Gold_Count')

print("\n按国家、运动项目、年份汇总的金牌数:")
print(df_medal_counts.head())

# 3.4 合并主办国信息到奖牌数据
df_medal_counts = pd.merge(
    df_medal_counts,
    df_hosts[['Year', 'Host_NOC']],
    on='Year',
    how='left'
)

# 3.5 创建Host列：如果NOC == Host_NOC，则为1，否则为0
df_medal_counts['Host'] = np.where(df_medal_counts['NOC'] == df_medal_counts['Host_NOC'], 1, 0)

print("\n含Host标记的奖牌数据:")
print(df_medal_counts.head())

# 3.6 计算历史平均金牌数
df_historical = df_medal_counts.groupby('NOC')['Gold_Count'].mean().reset_index().rename(
    columns={'Gold_Count': 'HistoricalGold'})
print("\n每个国家的历史平均金牌数:")
print(df_historical.head())

# 3.7 合并历史金牌数到奖牌数据
df_final = pd.merge(df_medal_counts, df_historical, on='NOC', how='left')

# 3.8 处理特殊NOC
# 移除 'mixed team' 和其他特殊NOC
df_final = df_final[~df_final['NOC'].isin(['MIX'])]
print("\n移除 'mixed team' 后的最终数据:")
print(df_final.head())

# 3.9 计算每年的总赛事数
# 转换赛事项目数据为长格式
df_programs_melt = df_programs.melt(
    id_vars=['Sport', 'Discipline', 'Code', 'Sports Governing Body'],
    var_name='Year',
    value_name='Num_Events'
)

# 提取年份数字，并去除可能的非数字字符
df_programs_melt['Year'] = pd.to_numeric(df_programs_melt['Year'], errors='coerce')

# 处理 'Num_Events' 列，转换为数值型，非数值字符如 '•' 转换为 0
df_programs_melt['Num_Events'] = pd.to_numeric(df_programs_melt['Num_Events'], errors='coerce').fillna(0)

# 计算每年的Total_Events
df_total_events = df_programs_melt.groupby('Year')['Num_Events'].sum().reset_index().rename(
    columns={'Num_Events': 'Total_Events'})

print("\n每年的总赛事数:")
print(df_total_events.head())

# 3.10 合并Total_Events到df_final
df_final = pd.merge(df_final, df_total_events, on='Year', how='left')

# 填补缺失的Total_Events为0
df_final['Total_Events'] = df_final['Total_Events'].fillna(0)

print("\n合并总赛事数后的最终数据:")
print(df_final[['NOC', 'Year', 'Sport', 'Gold_Count', 'Host', 'HistoricalGold', 'Total_Events']].head())

# 3.11 识别尚未获得奖牌的国家/地区
# 定义允许的新NOCs
allowed_new_nocs = ['ISL', 'GHA', 'IRQ', 'MAS', 'KUW', 'PAR', 'SUD', 'KSA']

# 识别这些国家是否已经在奖牌数据中出现
new_nocs = [noc for noc in allowed_new_nocs if noc not in df_final['NOC'].unique()]

print("\n尚未获得奖牌的国家/地区:", new_nocs)

# ========= 4. 定义“伟大教练”名单并标注 ==========
# 4.1 定义“伟大教练”名单
# 根据您提供的信息，定义“伟大教练”的执教记录
great_coaches = [
    {
        'Coach_Name': 'Lang Ping',
        'NOC': 'CHN',
        'Sport': 'Volleyball',
        'Year': 2021  # 东京奥运会
    },
    {
        'Coach_Name': 'Lang Ping',
        'NOC': 'CHN',
        'Sport': 'Volleyball',
        'Year': 2016  # 里约奥运会
    },
    {
        'Coach_Name': 'Lang Ping',
        'NOC': 'USA',
        'Sport': 'Volleyball',
        'Year': 2008  # 北京奥运会
    },
    {
        'Coach_Name': 'Lang Ping',
        'NOC': 'CHN',
        'Sport': 'Volleyball',
        'Year': 1996  # 亚特兰大奥运会
    },
    {
        'Coach_Name': 'Béla Károlyi',
        'NOC': 'ROU',
        'Sport': 'Gymnastics',
        'Year': 1976  # 北京奥运会
    },
    {
        'Coach_Name': 'Béla Károlyi',
        'NOC': 'USA',
        'Sport': 'Gymnastics',
        'Year': 1988  # 北京奥运会
    }

]

df_coaches = pd.DataFrame(great_coaches)
print("\n伟大教练名单:")
print(df_coaches)

# 4.2 标注“伟大教练”效应
# 先初始化Coach列为0
df_final['Coach'] = 0

# 遍历每个“伟大教练”的记录，并标注相应的记录
for index, coach in df_coaches.iterrows():
    condition = (
            (df_final['NOC'] == coach['NOC']) &
            (df_final['Sport'] == coach['Sport']) &
            (df_final['Year'] == coach['Year'])
    )
    df_final.loc[condition, 'Coach'] = 1

print("\n标注“伟大教练”后的奖牌数据:")
print(df_final[['NOC', 'Sport', 'Year', 'Coach']].head(15))

# ========= 5. 构建并拟合泊松回归模型 ==========
# 5.1 添加国家、运动项目和年份的固定效应
df_final['Country_FE'] = df_final['NOC']  # 保留NOC信息，但不作为固定效应
df_final['Sport_FE'] = df_final['Sport']
df_final['Year_FE'] = df_final['Year']  # 确保 Year_FE 是数值型

# 检查 Year_FE 的数据类型
print("\nYear_FE 的数据类型:")
print(df_final[['Year_FE']].dtypes)

# 5.2 构建模型公式
# 为了让模型能够泛化到新NOCs，移除 Country_FE 作为固定效应
# 将 Year_FE 作为数值型变量处理，而非分类变量
formula = 'Gold_Count ~ Coach + C(Sport_FE) + Year_FE'

# 5.3 拟合泊松回归模型
model_poisson = smf.glm(formula=formula, data=df_final, family=sm.families.Poisson()).fit()
print("\n泊松回归模型结果:")
print(model_poisson.summary())

# 5.4 检查过度分散
mean_count = df_final['Gold_Count'].mean()
var_count = df_final['Gold_Count'].var()
dispersion = var_count / mean_count
print(f"\n均值 (Mean Count): {mean_count}")
print(f"方差 (Variance Count): {var_count}")
print(f"离散比率 (Dispersion Ratio - Variance/Mean): {dispersion}")

if dispersion > 1.5:
    print("数据存在过度分散，建议使用负二项回归模型。")
    # 5.5 拟合负二项回归模型
    model_nb = smf.glm(formula=formula, data=df_final, family=sm.families.NegativeBinomial()).fit()
    print("\n负二项回归模型结果:")
    print(model_nb.summary())
else:
    model_nb = model_poisson

# ========= 6. 模型性能评估 ==========
# 6.1 计算模型在训练数据上的预测
df_final['Predicted'] = model_nb.predict(df_final)

# 6.2 计算性能指标
mse = mean_squared_error(df_final['Gold_Count'], df_final['Predicted'])
mae = mean_absolute_error(df_final['Gold_Count'], df_final['Predicted'])
r2 = r2_score(df_final['Gold_Count'], df_final['Predicted'])

print(f"\n模型性能评估：\n均方误差 (MSE)：{mse:.2f}\n平均绝对误差 (MAE)：{mae:.2f}\n决定系数 (R²)：{r2:.2f}")

# ========= 7. 进行“伟大教练”效应预测 ==========
# 7.1 定义每个新NOC的关键运动项目
key_sports = {
    'CHN':'Volleyball',
    'USA':'Volleyball',
    'ROU':'Gymnastics',
    'ISL': 'Handball',
    'GHA': 'Athletics',
    'IRQ': 'Weightlifting',
    'MAS': 'Badminton',
    'KUW': 'Shooting',
    'PAR': 'Football',
    'SUD': 'Boxing',
    'KSA': 'Wrestling'
}

# 7.2 创建预测数据集
predictions_list_with_coach = []
predictions_list_without_coach = []

for noc, sport in key_sports.items():
    # 获取2028年的总赛事数
    if 2028 in df_total_events['Year'].values:
        total_events_2028 = df_total_events[df_total_events['Year'] == 2028]['Total_Events'].values[0]
    else:
        total_events_2028 = df_total_events['Total_Events'].mean()

    # 有“伟大教练”
    predictions_list_with_coach.append({
        'NOC': noc,
        'Sport': sport,
        'Year': 2028,
        'Coach': 1,  # 拥有“伟大教练”
        'Sport_FE': sport,
        'Year_FE': 2028,
        'Total_Events': total_events_2028
    })

    # 无“伟大教练”
    predictions_list_without_coach.append({
        'NOC': noc,
        'Sport': sport,
        'Year': 2028,
        'Coach': 0,  # 不拥有“伟大教练”
        'Sport_FE': sport,
        'Year_FE': 2028,
        'Total_Events': total_events_2028
    })

df_predict_with_coach = pd.DataFrame(predictions_list_with_coach)
df_predict_without_coach = pd.DataFrame(predictions_list_without_coach)

print("\n2028年预测数据（有伟大教练）:")
print(df_predict_with_coach)

print("\n2028年预测数据（无伟大教练）:")
print(df_predict_without_coach)

# 7.3 确保新运动项目在训练数据中已存在
unique_sports_train = df_final['Sport'].unique()
df_predict_with_coach = df_predict_with_coach[df_predict_with_coach['Sport'].isin(unique_sports_train)]
df_predict_without_coach = df_predict_without_coach[df_predict_without_coach['Sport'].isin(unique_sports_train)]

# 7.4 进行预测
df_predict_with_coach['Predicted_Gold'] = model_nb.predict(df_predict_with_coach)
df_predict_without_coach['Predicted_Gold'] = model_nb.predict(df_predict_without_coach)

# 7.5 计算95%预测区间
# 使用 `get_prediction` 方法来获取预测区间
predictions_with_coach = model_nb.get_prediction(df_predict_with_coach)
pred_summary_with_coach = predictions_with_coach.summary_frame(alpha=0.05)  # 95%预测区间
df_predict_with_coach['Gold_Lower'] = pred_summary_with_coach['mean_ci_lower']
df_predict_with_coach['Gold_Upper'] = pred_summary_with_coach['mean_ci_upper']

predictions_without_coach = model_nb.get_prediction(df_predict_without_coach)
pred_summary_without_coach = predictions_without_coach.summary_frame(alpha=0.05)  # 95%预测区间
df_predict_without_coach['Gold_Lower'] = pred_summary_without_coach['mean_ci_lower']
df_predict_without_coach['Gold_Upper'] = pred_summary_without_coach['mean_ci_upper']

# 7.6 确保预测值合理（如不超过60金）
for df in [df_predict_with_coach, df_predict_without_coach]:
    df['Predicted_Gold'] = df['Predicted_Gold'].clip(upper=60)
    df['Gold_Lower'] = df['Gold_Lower'].clip(lower=0)
    df['Gold_Upper'] = df['Gold_Upper'].clip(upper=60)

# 7.7 计算获得第一枚金牌的概率及赔率
for df in [df_predict_with_coach, df_predict_without_coach]:
    df['Probability_First_Gold'] = 1 - np.exp(-df['Predicted_Gold'])
    df['Odds_First_Gold'] = df['Probability_First_Gold'] / (1 - df['Probability_First_Gold'])

# 7.8 合并有无“伟大教练”的预测结果
df_comparison = pd.concat([
    df_predict_with_coach.assign(Coach_Status='有伟大教练'),
    df_predict_without_coach.assign(Coach_Status='无伟大教练')
])

print("\n有无“伟大教练”的预测结果对比：")
print(df_comparison[
          ['NOC', 'Sport', 'Coach_Status', 'Predicted_Gold', 'Gold_Lower', 'Gold_Upper', 'Probability_First_Gold',
           'Odds_First_Gold']])

# ========= 8. 生成对比图 ==========
plt.figure(figsize=(6,5))

# 定义不同状态的标记和颜色
markers = {'有伟大教练': 'o', '无伟大教练': 's'}
colors = {'有伟大教练': 'blue', '无伟大教练': 'orange'}

# 遍历每个Coach_Status并绘制
for status in df_comparison['Coach_Status'].unique():
    subset = df_comparison[df_comparison['Coach_Status'] == status]
    plt.errorbar(
        subset['NOC'][0:3],
        subset['Predicted_Gold'][0:3],
        yerr=[
            subset['Predicted_Gold'][0:3] - subset['Gold_Lower'][0:3],
            subset['Gold_Upper'][0:3] - subset['Predicted_Gold'][0:3]
        ],
        fmt=markers[status],
        ecolor='black',
        capsize=5,
        capthick=2,
        label=status,
        color=colors[status],
        linestyle='None',
        mec='red'
    )

plt.xlabel('NOC', fontsize=14)
plt.ylabel('Predicted Golds', fontsize=14)
plt.title('Comparison on gold numbers', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('3,5.jpg')
plt.show()

# ========= 9. 模型结果分析 ==========
print("\n尚未获得奖牌的国家/地区预测对比：")
for index, row in df_comparison.iterrows():
    noc = row['NOC']
    sport = row['Sport']
    coach_status = row['Coach_Status']
    predicted_gold = row['Predicted_Gold']
    probability = row['Probability_First_Gold']
    odds = row['Odds_First_Gold']
    print(
        f"国家/地区: {noc}, 运动项目: {sport}, 状态: {coach_status}, 预测金牌数: {predicted_gold:.2f}, 获得第一枚金牌的概率: {probability:.2%}, 赔率: {odds:.2f}")

# 模型系数和标准误差
coefficients = {
    'Intercept': 6.5920,
    'C(Sport_FE)[T.Gymnastics]': 0.2366,
    'C(Sport_FE)[T.Volleyball]': 1.0672,
    'Coach': -0.0910,
    'Year_FE': -0.0026
}

std_errors = {
    'Intercept': 0.585,
    'C(Sport_FE)[T.Gymnastics]': 0.253,
    'C(Sport_FE)[T.Volleyball]': 0.256,
    'Coach': 0.197,
    'Year_FE': 0.000
}

# 系数的置信区间 [0.025, 0.975]
ci_lower = {
    'Intercept': 5.446,
    'C(Sport_FE)[T.Gymnastics]': -0.259,
    'C(Sport_FE)[T.Volleyball]': 0.566,
    'Coach': -0.478,
    'Year_FE': -0.003
}

ci_upper = {
    'Intercept': 7.738,
    'C(Sport_FE)[T.Gymnastics]': 0.732,
    'C(Sport_FE)[T.Volleyball]': 1.568,
    'Coach': 0.296,
    'Year_FE': -0.002
}

# 将数据整理为 DataFrame
df = pd.DataFrame({
    'Coefficient': coefficients.values(),
    'Std_Error': std_errors.values(),
    'CI_Lower': ci_lower.values(),
    'CI_Upper': ci_upper.values()
}, index=coefficients.keys())

# 灵敏度热力图
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Coefficient']], annot=True, cmap="coolwarm", cbar_kws={'label': 'Coefficient Value'})
plt.title('Sensitivity Heatmap: Coefficients')
plt.tight_layout()
plt.savefig('5-lmd.jpg')
plt.show()

# 灵敏度-不确定性图
plt.figure(figsize=(8, 6))
for feature in df.index:
    plt.plot([df.loc[feature, 'CI_Lower'], df.loc[feature, 'CI_Upper']], [feature, feature], label=feature, marker='o')
    plt.scatter(df.loc[feature, 'Coefficient'], feature, color='black')

plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.title('Sensitivity-Uncertainty Plot')
plt.grid(True)
plt.tight_layout()
plt.savefig('5-bqdx.jpg')
plt.show()
