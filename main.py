import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine


wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)


print(df.head())
print(df.describe())


features = ['alcohol', 'malic_acid', 'ash', 'flavanoids']


df[features].hist(figsize=(10, 6))
plt.tight_layout()
plt.show()


plt.scatter(df['alcohol'], df['flavanoids'])
plt.xlabel('Alcohol')
plt.ylabel('Flavanoids')
plt.title('Alcohol vs Flavanoids')
plt.show()

plt.figure(figsize=(8, 6))
sns.regplot(
    x='alcohol',
    y='flavanoids',
    data=df,
    scatter_kws={'alpha': 0.6},
    line_kws={'color': 'red'}
)

plt.title('Correlation between Alcohol and Flavanoids')
plt.xlabel('Alcohol')
plt.ylabel('Flavanoids')
plt.tight_layout()


plt.savefig('correlation.png')
plt.show()

