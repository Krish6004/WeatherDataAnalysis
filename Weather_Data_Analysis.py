import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "weather_data.csv"  # Update path if necessary
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Display basic info and statistics
print(df.info())
print(df.describe())

# Check for missing values
df = df.dropna()

# Visualizing data trends
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['Date'], y=df['Temperature'], label='Temperature')
sns.lineplot(x=df['Date'], y=df['Humidity'], label='Humidity')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Temperature & Humidity Trends Over Time')
plt.legend()
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df[['Temperature', 'Rainfall', 'Humidity']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Splitting data into training and testing sets
X = df[['Rainfall', 'Humidity']]
y = df['Temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")