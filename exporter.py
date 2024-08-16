import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the data
df = pd.read_csv("full_players_info_2023.csv")

# Drop columns and one-hot encode as in your Streamlit app
df = df.drop(columns=['name', 'height_in_cm'])
df = pd.get_dummies(df, columns=['country_of_citizenship', 'position', 'foot', 'current_club_domestic_competition_id', 'current_club_name'])

# Split data into features and target
X = df.drop(columns=['market_value_in_eur'])
y = df['market_value_in_eur']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
