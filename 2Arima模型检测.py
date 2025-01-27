import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 模拟SARIMAX模型的系数和标准误差（基于SARIMAX摘要数据）
coefficients = {
    'ar.L1': -1.948e-07,
    'ar.L2': -4.591e-06,
    'ar.L3': -3.451e-06,
    'ar.L4': -3.451e-06,
    'ar.L5': -3.451e-06,
    'sigma2': 0.5000
}

std_errors = {
    'ar.L1': 4.881,
    'ar.L2': 1.41e+06,
    'ar.L3': 5.863,
    'ar.L4': 6.492,
    'ar.L5': 4.881,
    'sigma2': 7.234
}

# 创建一个包含各特征的DataFrame
data = pd.DataFrame({
    'Feature': list(coefficients.keys()),
    'Coefficient': list(coefficients.values()),
    'StdError': list(std_errors.values())
})

# 计算每个特征的灵敏度（避免除以过小的数值导致的无效结果）
# 灵敏度 = 系数 / 标准误差，但我们也避免除数过小
data['Sensitivity'] = data.apply(
    lambda row: row['Coefficient'] / row['StdError'] if row['StdError'] > 0 else 0,
    axis=1
)

# 如果灵敏度非常小，可以选择进行归一化或设置阈值
threshold = 0.01  # 灵敏度的阈值，如果小于此值则设置为0
data['Sensitivity'] = data['Sensitivity'].apply(lambda x: x if abs(x) > threshold else 0)

# 打印灵敏度结果
print(data)

# ----------------------
# 绘制灵敏度热力图
# ----------------------

# 设置热力图的样式
plt.figure(figsize=(6, 4))
sns.heatmap(data[['Sensitivity']].T, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Sensitivity Heatmap(Q2-ARIMA)')
plt.xlabel('Features')
plt.ylabel('Sensitivity')
plt.savefig('2-1-1.jpg')
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
    'ar.L1': np.random.normal(coefficients['ar.L1'], std_errors['ar.L1'], n_simulations),
    'ar.L2': np.random.normal(coefficients['ar.L2'], std_errors['ar.L2'], n_simulations),
    'ar.L3': np.random.normal(coefficients['ar.L3'], std_errors['ar.L3'], n_simulations),
    'ar.L4': np.random.normal(coefficients['ar.L4'], std_errors['ar.L4'], n_simulations),
    'ar.L5': np.random.normal(coefficients['ar.L5'], std_errors['ar.L5'], n_simulations),
    'sigma2': np.random.normal(coefficients['sigma2'], std_errors['sigma2'], n_simulations),
}

# 将模拟数据转换为DataFrame
simulated_df = pd.DataFrame(simulated_data)

# 假设模型输出（例如金牌数的预测）是一个简单的线性组合
# 通过加权模拟数据计算模拟的输出
simulated_output = (
    simulated_df['ar.L1'] +
    simulated_df['ar.L2'] +
    simulated_df['ar.L3'] +
    simulated_df['ar.L4'] +
    simulated_df['ar.L5'] +
    simulated_df['sigma2']  # 这是一个简单示例，实际模型需要更复杂的计算
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
plt.title('Sensitivity-Uncertainty Plot(Q2-ARIMA)')
plt.xlabel('Model Output (e.g., Predicted Total Medals)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('2-1-2.jpg')
plt.show()

# 打印模拟输出的均值和标准差
print(f"Simulated Output Mean: {mean_output:.2f}")
print(f"Simulated Output Standard Deviation: {std_output:.2f}")
