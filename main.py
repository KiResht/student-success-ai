import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


print("Loading student data...")
df = pd.read_csv('student-mat.csv', sep=';')

y = df['G3']
X = df.drop(['G1', 'G2', 'G3'], axis=1)

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Random Forest AI...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print(f"\nModel Accuracy:")
print(f"Mean Absolute Error: {mae:.2f} (How many grade points the AI is off by, on average)")

importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 5 Most Important Factors for Student Success:")
print(importances.nlargest(5))

print("\nGenerating Success Factor Chart...")
plt.figure(figsize=(10, 6))

top_factors = importances.nlargest(5).sort_values(ascending=True)

sns.barplot(x=top_factors.values, y=top_factors.index, palette='mako')

plt.title('Top 5 Predictors of Student Success', fontsize=14, fontweight='bold')
plt.xlabel('AI Importance Score', fontsize=12)
plt.ylabel('Social/Environmental Factor', fontsize=12)

plt.tight_layout()
plt.savefig('success_factors.png', dpi=300)
print("Graph saved successfully as 'success_factors.png'!")
