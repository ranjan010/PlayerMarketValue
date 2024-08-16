import streamlit as st
import pandas as pd
import pickle

# Streamlit app title
st.title("Football Player Market Value Prediction")
csv_file_path = "full_players_info_2023.csv"  # Replace with your actual CSV file path

try:
    # Load the pre-trained model and scaler
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Display the first few rows of the dataframe
    st.write("### Dataset Overview:")
    st.write(df.head())

    # Make a copy of the original dataframe before one-hot encoding
    df_original = df.copy()

    # Dropping the 'name' column as it's not needed for prediction
    if 'name' in df.columns:
        df = df.drop(columns=['name'])

    # One-Hot Encode categorical variables
    df = pd.get_dummies(df, columns=['country_of_citizenship', 'position', 'foot', 'current_club_domestic_competition_id', 'current_club_name'])

    # Split data into features and target
    X = df.drop(columns=['market_value_in_eur'])

    # UI to input player attributes using the original dataframe
    st.write("### Enter Player Attributes and Details:")
    country_of_citizenship = st.selectbox("Country of Citizenship", df_original['country_of_citizenship'].unique())
    position = st.selectbox("Position", df_original['position'].unique())
    foot = st.selectbox("Foot", df_original['foot'].unique())
    height_in_cm = st.number_input("Height (in cm)", min_value=150, max_value=220, value=180)
    current_club_domestic_competition_id = st.selectbox("Club Competition ID", df_original['current_club_domestic_competition_id'].unique())
    current_club_name = st.selectbox("Current Club", df_original['current_club_name'].unique())
    yellow_cards = st.number_input("Yellow Cards", min_value=0, max_value=20, value=0)
    red_cards = st.number_input("Red Cards", min_value=0, max_value=10, value=0)
    goals = st.number_input("Goals", min_value=0, max_value=100, value=0)
    assists = st.number_input("Assists", min_value=0, max_value=50, value=0)
    minutes_played = st.number_input("Minutes Played", min_value=0, max_value=5000, value=0)
    age = st.number_input("Age", min_value=15, max_value=50, value=25)

    # Convert input to dataframe
    input_data = pd.DataFrame({
        'country_of_citizenship': [country_of_citizenship],
        'position': [position],
        'foot': [foot],
        'height_in_cm': [height_in_cm],
        'current_club_domestic_competition_id': [current_club_domestic_competition_id],
        'current_club_name': [current_club_name],
        'yellow_cards': [yellow_cards],
        'red_cards': [red_cards],
        'goals': [goals],
        'assists': [assists],
        'minutes_played': [minutes_played],
        'age': [age]
    })

    # Apply the same one-hot encoding to the input data
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)

    # Predict the market value based on user input
    if st.button("Predict Market Value"):
        prediction = model.predict(input_data_scaled)
        st.write(f"### Predicted Market Value: €{prediction[0]:,.2f}")

except Exception as e:
    st.write(f"Error loading the dataset or model: {e}")
