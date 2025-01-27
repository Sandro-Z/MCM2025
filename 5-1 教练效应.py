import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
countries = ['China', 'USA', 'Romania', 'Japan', 'Russia', 'Germany', 'Brazil']
sports = ['Volleyball', 'Gymnastics', 'Basketball', 'Football', 'Tennis', 'Swimming', 'Track & Field']
medals_data = []
for country in countries:
    for sport in sports:
        gold = np.random.randint(0, 10)  # 金牌
        silver = np.random.randint(0, 10)  # 银牌
        bronze = np.random.randint(0, 10)  # 铜牌
        total = gold + silver + bronze
        medals_data.append([country, sport, gold, silver, bronze, total])

df = pd.DataFrame(medals_data, columns=['Country', 'Sport', 'Gold', 'Silver', 'Bronze', 'Total'])


coach_effect = {
    'China': {'Volleyball': 3},
    'USA': {'Gymnastics': 4},
    'Romania': {'Gymnastics': 5}
}

for country, effects in coach_effect.items():
    for sport, effect in effects.items():
        df.loc[(df['Country'] == country) & (df['Sport'] == sport), 'Gold'] += effect

print("Updated Medal Data with Coach Effects:")
print(df)

plt.figure(figsize=(14, 8))
for country in countries:
    country_data = df[df['Country'] == country]
    plt.plot(country_data['Sport'], country_data['Gold'], label=country, marker='o')

plt.title(" Coach Effect impact", fontsize=16)
plt.xlabel("Sport", fontsize=12)
plt.ylabel("Gold Medals", fontsize=12)
plt.legend(title="Country")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('5.jpg')
plt.show()
