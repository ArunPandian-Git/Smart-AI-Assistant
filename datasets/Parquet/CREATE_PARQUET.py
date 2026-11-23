import pandas as pd
df = pd.read_csv('../CSV/products.csv')
df.to_parquet('products.parquet', index=False, engine='pyarrow')
print('Created products.parquet')
