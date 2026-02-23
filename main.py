import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

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
