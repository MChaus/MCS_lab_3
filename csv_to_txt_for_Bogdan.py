import pandas as pd
import numpy as np

data_X = pd.read_csv('X.csv', index_col=0, header=0)
data_y = pd.read_csv('y.csv', index_col=0, header=0)
data_X.to_csv(r'X.txt', index=False, header=False, sep=' ')
data_y.to_csv(r'y.txt', index=False, header=False, sep=' ')
