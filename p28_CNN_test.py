import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Load and prepare the data
data_path = 'GAMI-NET_Chl.csv' # Replace with the path to your data file
data = pd.read_csv(data_path)
X = data.iloc[:, :-1].values
y = data['Chl'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the features for CNN
X_train_reshaped = X_train.reshape(X_train.shape[0], 4, 6, 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], 4, 6, 1)

# Define and compile the CNN model
model = Sequential([
    Conv2D(32, (2,2), activation='relu', input_shape=(4, 6, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1) # No activation function for regression
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=4, validation_data=(X_test_reshaped, y_test))

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
rpd = np.std(y_test) / rmse

# Print the performance metrics
print(f'R2: {r2}')
print(f'RMSE: {rmse}')
print(f'RPD: {rpd}')
