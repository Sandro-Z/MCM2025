import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 创建数据
data = {
    'Variable': [
        'Intercept', 'Aeronautics', 'Alpinism', 'Archery', 'Art Competitions',
        'Artistic Gymnastics', 'Artistic Swimming', 'Athletics', 'Badminton',
        'Baseball', 'Baseball/Softball', 'Basketball', 'Basque Pelota',
        'Beach Volleyball', 'Boxing', 'Breaking', 'Canoe Slalom', 'Canoe Sprint',
        'Canoeing', 'Cricket', 'Croquet', 'Cycling', 'Cycling BMX Freestyle',
        'Cycling BMX Racing', 'Cycling Mountain Bike', 'Cycling Road',
        'Cycling Track', 'Diving', 'Equestrian', 'Equestrianism', 'Fencing',
        'Figure Skating', 'Football', 'Golf', 'Gymnastics', 'Handball',
        'Hockey', 'Ice Hockey', 'Jeu De Paume', 'Judo', 'Karate', 'Lacrosse',
        'Marathon Swimming', 'Modern Pentathlon', 'Motorboating', 'Polo',
        'Racquets', 'Rhythmic Gymnastics', 'Roque', 'Rowing', 'Rugby',
        'Rugby Sevens', 'Sailing', 'Shooting', 'Skateboarding', 'Softball',
        'Sport Climbing', 'Surfing', 'Swimming', 'Synchronized Swimming',
        'Table Tennis', 'Taekwondo', 'Tennis', 'Trampoline Gymnastics',
        'Trampolining', 'Triathlon', 'Tug-Of-War', 'Volleyball', 'Water Polo',
        'Weightlifting', 'Wrestling', 'Coach', 'Year_FE'
    ],
    'Coefficient': [
        8.9952, -1.7099, -1.0243, -0.0580, -1.3918, -0.4817, 0.9654, -0.2650,
        -0.4832, 1.6417, 2.2697, 1.2580, -1.1523, -0.6734, -1.0209, -1.3788,
        -1.1623, -0.2143, -0.3381, 0.6395, -0.4591, -0.4530, -1.3863, -1.3863,
        -1.3888, -1.2339, -0.1337, -0.5058, 0.4053, -0.2766, 0.0130, -1.3917,
        1.2477, -0.7798, 0.1332, 1.3216, 1.2722, 0.3093, -1.8153, -0.8450,
        -1.3938, 0.6621, -1.3863, -1.0040, -0.5625, -0.4831, -0.7167, -0.0528,
        -1.8303, 0.2697, 0.6170, 1.1452, -0.3896, -0.7076, -0.6932, 1.2466,
        -1.3838, -1.3863, 0.3376, 0.5452, 0.2253, -1.1584, -0.7061, -1.3863,
        -1.2142, -1.1822, 0.0031, 1.0273, 0.8494, -0.9018, -0.7982, 0.0337,
        -0.0038
    ],
    'P-value': [
        0.000, 0.261, 0.321, 0.922, 0.021, 0.443, 0.298, 0.638, 0.423, 0.023,
        0.050, 0.034, 0.393, 0.293, 0.072, 0.229, 0.113, 0.739, 0.553, 0.589,
        0.714, 0.426, 0.124, 0.124, 0.160, 0.118, 0.831, 0.383, 0.583, 0.629,
        0.982, 0.071, 0.033, 0.268, 0.814, 0.027, 0.030, 0.797, 0.233, 0.140,
        0.063, 0.475, 0.124, 0.092, 0.566, 0.507, 0.577, 0.934, 0.229, 0.634,
        0.382, 0.103, 0.493, 0.212, 0.403, 0.101, 0.085, 0.124, 0.550, 0.404,
        0.721, 0.051, 0.227, 0.124, 0.098, 0.072, 0.996, 0.083, 0.147, 0.114,
        0.159, 0.965, 0.000
    ]
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 创建热力图数据
heatmap_data = df.pivot_table(index='Variable', values='Coefficient', aggfunc='first')

# 绘制热力图
plt.figure(figsize=(10,10))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Sensitivity Heatmap')
plt.savefig('4-1-heatmap.jpg')
plt.show()

# 绘制灵敏度-不确定性图
plt.figure(figsize=(10, 6))
plt.scatter(df['Coefficient'], df['P-value'], alpha=0.6)
plt.xlabel('Coefficient')
plt.ylabel('P-value')
plt.title('Sensitivity-Uncertainty Plot(Q4-Event-Medals)')
plt.axhline(y=0.05, color='r', linestyle='--')  # 添加显著性水平线
plt.savefig('4-1-hist.jpg')
plt.show()