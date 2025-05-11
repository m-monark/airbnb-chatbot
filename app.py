import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI

# Predict missing distances using RandomForest

def predict_and_fill_distances(df):
    df["comment_length"] = df["comments"].apply(len)
    df["price_bin"] = pd.qcut(df["price"], q=4, labels=False, duplicates="drop")
    df["comment_length_bin"] = pd.qcut(df["comment_length"], q=4, labels=False, duplicates="drop")
    features = ["price", "price_bin", "comment_length", "comment_length_bin"]
    X = df[features].fillna(df[features].mean())

    if df["distance_from_beach"].notna().sum() >= 1:
        model_beach = RandomForestRegressor()
        model_beach.fit(X[df["distance_from_beach"].notna()], df.loc[df["distance_from_beach"].notna(), "distance_from_beach"])
        df.loc[df["distance_from_beach"].isna(), "distance_from_beach"] = model_beach.predict(X[df["distance_from_beach"].isna()])

    if df["distance_from_city_center"].notna().sum() >= 1:
        model_city = RandomForestRegressor()
        model_city.fit(X[df["distance_from_city_center"].notna()], df.loc[df["distance_from_city_center"].notna(), "distance_from_city_center"])
        df.loc[df["distance_from_city_center"].isna(), "distance_from_city_center"] = model_city.predict(X[df["distance_from_city_center"].isna()])

    return df

# Load and clean data

def load_and_clean_data(path):
    df = pd.read_excel(path, sheet_name="Reviews")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["comments"] = df["comments"].fillna("No review")
    if "distance_from_beach" not in df.columns:
        df["distance_from_beach"] = np.nan
    if "distance_from_city_center" not in df.columns:
        df["distance_from_city_center"] = np.nan
    return predict_and_fill_distances(df)

# Interpret query for price and beach condition

def local_interpret_query(query):
    query = query.lower()
    price_limit = 100
    near_beach = "beach" in query
    for keyword in ["under", "max", "less than"]:
        if keyword in query:
            try:
                price_limit = int(query.split(keyword)[1].split()[0].replace("$", "").strip())
                break
            except:
                pass
    return price_limit, near_beach

# Format distances

def format_km(val):
    try:
        return f"{float(val):.1f} km"
    except:
        return "Not specified"

# Run the app

def run_chatbot():
    st.set_page_config(page_title="Airbnb Chatbot", layout="centered")
    st.title("Airbnb Chatbot – Explore Listings in Messina")

    if "history" not in st.session_state:
        st.session_state.history = []

    df = load_and_clean_data("Airbnb_EDA_Messina.xlsx")
    user_input = st.chat_input("Ask something like 'family-friendly listings near the beach under $80'")

    if user_input:
        st.session_state.history.append({"user": user_input})
        price_limit, near_beach = local_interpret_query(user_input)

        filtered = df[df["price"] <= price_limit]
        if near_beach:
            filtered = filtered.sort_values(by="distance_from_beach")
        filtered = filtered.drop_duplicates(subset="name", keep="first")

        if filtered.empty:
            fallback = df.sort_values(by="distance_from_beach" if near_beach else "price")
            fallback = fallback.drop_duplicates(subset="name", keep="first").head(1)
            fallback_notice = True
            results = fallback
        else:
            fallback_notice = False
            results = filtered.head(3)

        listings_text = ""
        for _, row in results.iterrows():
            listings_text += f"- Name: {row['name']}\n"
            listings_text += f"  Price: ${row['price']:.0f} per night\n"
            listings_text += f"  Beach Distance: {format_km(row['distance_from_beach'])}\n"
            listings_text += f"  Center Distance: {format_km(row['distance_from_city_center'])}\n"
            listings_text += f"  Summary of Guest Comment: \"{row['comments'][:180]}\"\n\n"

        prompt = f"""
You are an assistant helping users discover Airbnb listings in Messina.

User query:
{user_input}

Here are the listings:

{listings_text}

Please recommend the best listing(s). Mention the name, price, distances, and comment summary.
Conclude by reminding the user to check Airbnb for live availability and pricing.
"""

        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You help users choose the best Airbnb listings in Messina."},
                    {"role": "user", "content": prompt}
                ]
            )
            bot_reply = response.choices[0].message.content
        except Exception as e:
            bot_reply = f"Sorry, GPT could not process your request.\n\nError: {e}"

        if fallback_notice:
            bot_reply = f"\ud83d\udea8 No listings were found under ${price_limit}, but here's the closest match:\n\n{bot_reply}"

        st.session_state.history[-1]["bot"] = bot_reply

    for chat in st.session_state.history:
        st.chat_message("user").write(chat["user"])
        if "bot" in chat:
            st.chat_message("assistant").write(chat["bot"])

if __name__ == "__main__":
    run_chatbot()
