import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings 
warnings.filterwarnings('ignore')

#Load dataset

grating_information = {
    0: {'filename': 'Red.xlsx', 'sheet_name': 'Each', 'usecols': 'A:U'},    
    1: {'filename': 'Green.xlsx', 'sheet_name': 'Each', 'usecols': 'A:U'},    
    2: {'filename': 'Blue.xlsx', 'sheet_name': 'Each', 'usecols': 'A:U'},           
}

color = 2

filename = grating_information[color]['filename']
sheet_name = grating_information[color]['sheet_name']
usecols = grating_information[color]['usecols']

df = np.array(pd.read_excel(filename, sheet_name = sheet_name, usecols = usecols))

input = 5
output = 16


# Split training and test sets
X = df[:,range(0, input)]
y = df[:,range(input, input + output)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#Scale X and y
scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler().fit(y_train)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)
