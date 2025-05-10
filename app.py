import pandas as pd
import numpy as np
import os
import openai
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Predict missing distances
def predict_and_fill_distances(df):
    df["comment_length"] = df["comments"].apply(len)
    df["price_bin"] = pd.qcut(df["price"], q=4, labels=False, duplicates="drop")
    df["comment_length_bin"] = pd.qcut(df["comment_length"], q=4, labels=False, duplicates="drop")

    feature_cols = ["price", "price_bin", "comment_length", "comment_length_bin"]
    X = df[feature_cols].fillna(df[feature_cols].mean())

    if df["distance_from_beach"].notna().sum() > 5:
        model_beach = RandomForestRegressor()
        model_beach.fit(X[df["distance_from_beach"].notna()], df.loc[df["distance_from_beach"].notna(), "distance_from_beach"])
        df.loc[df["distance_from_beach"].isna(), "distance_from_beach"] = model_beach.predict(X[df["distance_from_beach"].isna()])

    if df["distance_from_city_center"].notna().sum() > 5:
        model_city = RandomForestRegressor()
        model_city.fit(X[df["distance_from_city_center"].notna()], df.loc[df["distance_from_city_center"].notna(), "distance_from_city_center"])
        df.loc[df["distance_from_city_center"].isna(), "distance_from_city_center"] = model_city.predict(X[df["distance_from_city_center"].isna()])

    return df

# Load data
def load_and_clean_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Reviews")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["comments"] = df["comments"].fillna("No review")

    if "distance_from_beach" not in df.columns:
        df["distance_from_beach"] = np.nan
    if "distance_from_city_center" not in df.columns:
        df["distance_from_city_center"] = np.nan

    df = predict_and_fill_distances(df)
    return df

# Optional GPT interpretation
def get_ai_query_intent(prompt):
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            return None
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You help users find Airbnb listings."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception:
        return None

# Simple fallback logic
def local_interpret_query(query):
    query = query.lower()
    price_limit = 100
    near_beach = "beach" in query
    if "under" in query:
        try:
            price_limit = int(query.split("under")[1].split()[0].replace("$", "").strip())
        except:
            pass
    return price_limit, near_beach

# Format distances
def format_distance(value):
    return f"{value:.1f} km" if pd.notna(value) else "unknown"

# Main chatbot logic
def run_chatbot():
    st.set_page_config(page_title="Airbnb Chatbot", layout="centered")
    st.title("Airbnb Chatbot – Explore Listings in Messina")

    if "history" not in st.session_state:
        st.session_state.history = []

    df = load_and_clean_data("Airbnb_EDA_Messina.xlsx")
    user_input = st.chat_input("Ask something like 'places near the beach under $100'")

    if user_input:
        st.session_state.history.append({"user": user_input})
        gpt_response = get_ai_query_intent(user_input)

        if gpt_response:
            bot_reply = f"""GPT response:

{gpt_response}"""
        else:
            price_limit, near_beach = local_interpret_query(user_input)
            results = df[df["price"] <= price_limit]

            if near_beach:
                results = results.sort_values(by="distance_from_beach").head(5)
            else:
                results = results.head(5)

            if results.empty:
                bot_reply = "No listings match your criteria."
            else:
                bot_reply = "Here are some listings:\n\n"
                for _, row in results.iterrows():
                    bot_reply += f"- ${row['price']} | Beach: {format_distance(row['distance_from_beach'])}, Center: {format_distance(row['distance_from_city_center'])}\n"
                    bot_reply += f"  Review: \"{row['comments'][:100]}...\"\n\n"

        st.session_state.history[-1]["bot"] = bot_reply

    for chat in st.session_state.history:
        st.chat_message("user").write(chat["user"])
        if "bot" in chat:
            st.chat_message("assistant").write(chat["bot"])

# Run app
if __name__ == "__main__":
    run_chatbot()
