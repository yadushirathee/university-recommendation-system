import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.4)),
                          url("https://assets.weforum.org/article/image/pvHwd-UDSSKJlYUP8ic24kGrlqfMaCSlTFQljAb4zjY.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the dataset
df = pd.read_csv("International_Education_Costs.csv")

# Compute total cost
df['Tuition_Cost'] = df['Tuition_USD']
df['Rent_Cost'] = df['Rent_USD'] * 12 * df['Duration_Years']
df['Insurance_Cost'] = df['Insurance_USD'] * df['Duration_Years']
df['Visa_Cost'] = df['Visa_Fee_USD']
df['Total_Cost_USD'] = df['Tuition_Cost'] + df['Rent_Cost'] + df['Insurance_Cost'] + df['Visa_Cost']

# Show app title
st.markdown(
    """
    <h1 style='text-align: center;
               color: white;
               font-size: 36px;
               text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
               margin-bottom: 30px;'>
        🎓 University Recommendation System
    </h1>
    """,
    unsafe_allow_html=True
)


# User input
st.markdown("<h4 style='color: #1F1F1F; text-shadow: 1px 1px 2px #999;'>🌍 Select Country</h4>", unsafe_allow_html=True)
country = st.selectbox("", df['Country'].unique())

available_programs = df[df['Country'] == country]['Program'].unique()
st.markdown("<h4 style='color: #1F1F1F; text-shadow: 1px 1px 2px #999;'>📘 Select Program</h4>", unsafe_allow_html=True)
program = st.selectbox("", available_programs)

st.markdown("<h4 style='color: #1F1F1F; text-shadow: 1px 1px 2px #999;'>🎓 Select Level</h4>", unsafe_allow_html=True)
level = st.selectbox("", df['Level'].unique())

st.markdown("<h4 style='color: #1F1F1F; text-shadow: 1px 1px 2px #999;'>💵 Enter Your Total Budget (USD)</h4>", unsafe_allow_html=True)
budget = st.number_input("")


if st.button("🔍 Get Recommendations"):
    # Filter dataset by user-selected country/program/level
    filtered_df = df[
        (df['Country'] == country) &
        (df['Program'] == program) &
        (df['Level'] == level)
    ].copy()

    if filtered_df.empty:
        st.warning("No universities found for your selected options.")
    else:
        # OneHotEncode the categorical features
        categorical_cols = ['Country', 'Program', 'Level']
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe_features = ohe.fit_transform(df[categorical_cols])
        df_encoded = pd.DataFrame(ohe_features, columns=ohe.get_feature_names_out(categorical_cols))

        # Scale the cost
        scaler = MinMaxScaler()
        df_scaled_cost = scaler.fit_transform(df[['Total_Cost_USD']])
        df_encoded.reset_index(drop=True, inplace=True)
        df_scaled = pd.DataFrame(df_scaled_cost, columns=["Total_Cost_scaled"])
        feature_matrix = pd.concat([df_encoded, df_scaled], axis=1)

        # Prepare user vector
        user_input = pd.DataFrame([[country, program, level]], columns=categorical_cols)
        user_encoded = ohe.transform(user_input)
        user_budget_scaled = scaler.transform([[budget]])
        user_vector = np.hstack([user_encoded, user_budget_scaled])

        # Cosine similarity
        similarities = cosine_similarity(user_vector, feature_matrix)
        top_indices = similarities[0].argsort()[-5:][::-1]
        recommendations = df.iloc[top_indices]
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <h3 style='text-align: center; color: black;'>🏆 Top 5 University Recommendations:</h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='
               overflow-x: auto;
               text-align: center;
               padding: 25px;
               background-color: rgba(255, 255, 255, 0.8);
               border-radius: 12px;
               margin-top: 25px;
        '>
        <table class='styled-table' style='margin: 0 auto; background-color: white;'>
        """ + 
        recommendations[['University', 'City', 'Country', 'Program', 'Level', 'Duration_Years',
                     'Tuition_USD', 'Rent_USD', 'Insurance_USD',
                     'Visa_Fee_USD', 'Living_Cost_Index', 'Total_Cost_USD']].to_html(
               index=False,
               escape=False,
               header=True
        ) +
        """
               </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <style>
        .styled-table {
             border-collapse: collapse;
             font-size: 16px;
             font-family: 'Segoe UI', sans-serif;
             min-width: 750px;
             box-shadow: 0 0 15px rgba(0,0,0,0.1);
             border-radius: 10px;
             overflow: hidden;
        }
        
        .styled-table thead tr {
             background-color: #007ACC;
             color: #ffffff;
             text-align: center;
        }
        
        .styled-table th, .styled-table td {
             padding: 14px 16px;
             border-bottom: 1px solid #ddd;
             text-align: center;
             color: #222222;
        }
        
        .styled-table tbody tr:nth-of-type(even) {
              background-color: white;
        }
        
        .styled-table tbody tr:hover {
             background-color: white;
             transition: 0.2s ease-in-out;
        }
        </style>
        """, unsafe_allow_html=True)

      