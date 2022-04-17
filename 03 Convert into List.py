import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

director_writers = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/directors_writers.csv')

#Mengubah director_name menjadi list
director_writers['director_name'] = director_writers['director_name'].apply(lambda row: row.split(','))
director_writers['writer_name'] = director_writers['writer_name'].apply(lambda row: row.split(','))

#Tampilkan 5 data teratas
print(director_writers.head())
