import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 模拟线性回归模型的系数和标准误差
coefficients = {
    'Intercept': 2.8937,
    'HistoricalGold': 0.4228,
    'Host': 0,
    'Total_Events': 0,
    'Num_Athletes': 0
}

std_errors = {
    'Intercept': 0.326,
    'HistoricalGold': 0.322,
    'Host': 0,
    'Total_Events': 0,
    'Num_Athletes': 0
}

# 创建一个包含各特征的DataFrame
data = pd.DataFrame({
    'Feature': list(coefficients.keys()),
    'Coefficient': list(coefficients.values()),
    'StdError': list(std_errors.values())
})

# 计算每个特征的灵敏度
data['Sensitivity'] = data['Coefficient'] / data['StdError']
print(data)

# ----------------------
# 绘制灵敏度热力图
# ----------------------

# 设置热力图的样式
plt.figure(figsize=(6, 4))
plt.xticks([0,1])
sns.heatmap(data[['Sensitivity']].T, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Sensitivity Heatmap(Q1)')
plt.xlabel('Features')
plt.ylabel('Sensitivity')
plt.savefig('1-1-1.jpg')
plt.show()

# ------------------------
# 灵敏度-不确定性分析 (蒙特卡洛模拟)
# ------------------------

# 设置模拟次数
n_simulations = 1000

# 模拟每个特征值
np.random.seed(42)

# 假设输入特征的变化范围为其标准误差的±2倍（这个假设可以调整）
simulated_data = {
    'Intercept': np.random.normal(coefficients['Intercept'], std_errors['Intercept'], n_simulations),
    'HistoricalGold': np.random.normal(coefficients['HistoricalGold'], std_errors['HistoricalGold'], n_simulations),
    'Host': np.random.normal(coefficients['Host'], std_errors['Host'], n_simulations),
    'Total_Events': np.random.normal(coefficients['Total_Events'], std_errors['Total_Events'], n_simulations),
    'Num_Athletes': np.random.normal(coefficients['Num_Athletes'], std_errors['Num_Athletes'], n_simulations),
}

# 将模拟数据转换为DataFrame
simulated_df = pd.DataFrame(simulated_data)

# 假设模型输出（预测金牌数）是一个简单的线性组合
# 通过加权模拟数据计算模拟的输出
simulated_output = (
    simulated_df['Intercept'] +
    simulated_df['HistoricalGold'] * simulated_df['HistoricalGold']  # 使用历史金牌系数模拟金牌数
)

# 计算模拟输出的统计量
mean_output = np.mean(simulated_output)
std_output = np.std(simulated_output)

# 绘制灵敏度-不确定性图
plt.figure(figsize=(8, 6))
plt.hist(simulated_output, bins=30, alpha=0.7, color='blue', label='Simulated Output')
plt.axvline(mean_output, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_output:.2f}')
plt.axvline(mean_output - std_output, color='green', linestyle='dashed', linewidth=2, label=f'Standard Deviation: {std_output:.2f}')
plt.axvline(mean_output + std_output, color='green', linestyle='dashed', linewidth=2)
plt.title('Sensitivity-Uncertainty Plot(Q1)')
plt.xlabel('Model Output (Gold Medal Count)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('1-1-2.jpg')
plt.show()

# 打印模拟输出的均值和标准差
print(f"Simulated Output Mean: {mean_output:.2f}")
print(f"Simulated Output Standard Deviation: {std_output:.2f}")
