import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time 
import warnings

# Load dataset
df = np.array(pd.read_excel('Red.xlsx', sheet_name='Each', usecols='A:U'))

input_dim = 5
output_dim = 16

# Split training and test sets
X = df[:,range(0, input_dim)]
y = df[:,range(input_dim, input_dim + output_dim)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Scale X and y
scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler().fit(y_train)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)

start = time.perf_counter()

# Set GPU device
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define the model
k = 512

learning_rate = 0.0001  

custom_optimizer = Adam(learning_rate=learning_rate)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(output_dim)
])

# Compile the model


model.compile(optimizer=custom_optimizer, loss='mse')

# Convert the data to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


# Train the model
model.fit(X_train, y_train, epochs = 100, batch_size = 128)

# Save the model
model.save('MLP_2D_128ch_xy.h5')

end = time.perf_counter()

time_taken = end - start
print("Time:", '%.1f s'% time_taken)

# Predict on training and test data
pre_y_train = model.predict(X_train)
pre_y_test = model.predict(X_test)

# Calculate metrics
Train_R2 = r2_score(y_train, pre_y_train)
Test_R2 = r2_score(y_test, pre_y_test)
Train_MSE = mean_squared_error(pre_y_train, y_train)
Test_MSE = mean_squared_error(pre_y_test, y_test)

print("Train_R2:", Train_R2)
print("Test_R2:", Test_R2)
print("Train MSE:", Train_MSE)
print("Test MSE:", Test_MSE)
