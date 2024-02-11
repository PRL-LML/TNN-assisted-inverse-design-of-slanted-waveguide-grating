import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split

### The dataset of the lens ###
 
grating_info = {
    '00': {'filename': 'Red.xlsx', 'sheet_name': 'Peak', 'performance_name': 'Red_Peak', 'usecols': 'A:F'},    
    '01': {'filename': 'Red.xlsx', 'sheet_name': 'Ave', 'performance_name': 'Red_Ave', 'usecols': 'A:F'},
    '02': {'filename': 'Red.xlsx', 'sheet_name': 'Uniformity', 'performance_name': 'Red_Uniformity', 'usecols': 'A:F'},
    '10': {'filename': 'Green.xlsx', 'sheet_name': 'Peak', 'performance_name': 'Green_Peak', 'usecols': 'A:F'},    
    '11': {'filename': 'Green.xlsx', 'sheet_name': 'Ave', 'performance_name': 'Green_Ave', 'usecols': 'A:F'},
    '12': {'filename': 'Green.xlsx', 'sheet_name': 'Uniformity', 'performance_name': 'Green_Uniformity', 'usecols': 'A:F'},
    '20': {'filename': 'Blue.xlsx', 'sheet_name': 'Peak', 'performance_name': 'Blue_Peak', 'usecols': 'A:F'},
    '21': {'filename': 'Blue.xlsx', 'sheet_name': 'Ave', 'performance_name': 'Blue_Ave', 'usecols': 'A:F'},
    '22': {'filename': 'Blue.xlsx', 'sheet_name': 'Uniformity', 'performance_name': 'Blue_Uniformity', 'usecols': 'A:F'},  
}


### Naming rules ###
# 00 = r_peak, 01 = r_ave, 02 = r_uniformity,
# 10 = g_peak, 11 = g_ave, 12 = g_uniformity,
# 20 = b_peak, 21 = b_ave, 22 = b_uniformity

grating = '00'

## Load the dataset
filename = grating_info[grating]['filename']
sheet_name = grating_info[grating]['sheet_name']
performance_name = grating_info[grating]['performance_name']
usecols = grating_info[grating]['usecols']

df = np.array(pd.read_excel(filename, sheet_name = sheet_name, usecols = usecols))

mse_mmd = performance_name + '_MSE_MMD' + '.xlsx'
pklname = performance_name + '.pkl'

### Pre-processing ###
input = 5
output = 1
    
X = df[:,range(0, input)]
y = df[:,range(input, input + output)]

# Split the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)

# Scaling X and y
scaler_x = StandardScaler().fit(X_train)
x_train = scaler_x.transform(X_train)
x_test = scaler_x.transform(X_test)

scaler_y = StandardScaler().fit(y_train)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)

x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)


# Inverse_transform
def inverse_transform_x(x):
    actual_x = scaler_x.inverse_transform(x)
    return actual_x

def inverse_transform_y(y):
    actual_y = scaler_y.inverse_transform(y)
    return actual_y
