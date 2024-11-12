import pandas as pd
subway = pd.read_csv('data/subwaystations2.csv')
print(subway.columns)
print(subway[['Display Name', 'population']].sort_values('population').head(15))