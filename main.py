import pandas as pd

df = pd.read_csv('student-mat.csv', sep=';')

print("Data successfully loaded!")
print(df.head())

print(df.groupby('studytime')['G3'].mean())
