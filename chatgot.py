# import streamlit as st
# from sentence_transformers import SentenceTransformer
# import pandas as pd
# import numpy as np
# import faiss
# import openai

# # Load the pre-trained model for embedding
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # Load the CSV file
# df = pd.read_csv('travel_itineraries.csv')

# # Generate embeddings for the descriptions in the CSV file
# df['embedding'] = df['description'].apply(lambda x: model.encode(x))

# # Convert embeddings to a 2D numpy array for FAISS
# embeddings = np.stack(df['embedding'].values)

# # Create a FAISS index
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(embeddings)

# # OpenAI API key
# openai.api_key = "sk-proj-xwUxmOs0ZkUegvZYyBuuT3BlbkFJ4wK9AngasGlQ104aLrDb"

# st.title("Travel Itinerary Recommendation Chatbot")

# user_input = st.text_input("Describe your ideal travel experience:")

# if user_input:
#     # Generate embeddings for the user input
#     user_vector = model.encode(user_input).reshape(1, -1)

#     # Query the FAISS index for the top K most similar entries
#     K = 3
#     distances, indices = index.search(user_vector, K)

#     st.write("Here is your travel itinerary:")

#     # First show the exact match if found
#     exact_match = False
#     for idx in indices[0]:
#         itinerary = df.iloc[idx]
#         if user_input.lower() in itinerary['city'].lower():
#             st.write(f"**City:** {itinerary['city']}")
#             st.write(f"**Activities:** {itinerary['activities']}")
#             st.write(f"**Food:** {itinerary['food']}")
#             st.write(f"**Accommodation:** {itinerary['accommodation']}")
#             st.write(f"**Description:** {itinerary['description']}")
#             st.write("---")
#             exact_match = True
#             break

#     if not exact_match:
#         st.write("No exact match found. Here are some similar itineraries:")

#     # Show other recommended cities
#     st.write("Here are some other recommended cities to visit:")
#     for idx in indices[0]:
#         itinerary = df.iloc[idx]
#         if user_input.lower() not in itinerary['city'].lower():
#             st.write(f"**City:** {itinerary['city']}")
#             st.write(f"**Activities:** {itinerary['activities']}")
#             st.write(f"**Food:** {itinerary['food']}")
#             st.write(f"**Accommodation:** {itinerary['accommodation']}")
#             st.write(f"**Description:** {itinerary['description']}")
#             st.write("---")

#WORKING STYLE
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# import pandas as pd
# import numpy as np
# import faiss
# import openai

# # Custom CSS for better styling
# st.markdown(
#     """
#     <style>
#     .main {
#         background-color: #f0f0f5;
#     }
#     .title {
#         font-size: 2.5em;
#         color: #4b4b9b;
#         text-align: center;
#         margin-top: 20px;
#     }
#     .input-box {
#         font-size: 1.2em;
#         margin-top: 20px;
#     }
#     .section-title {
#         font-size: 1.8em;
#         color: #4b4b9b;
#         margin-top: 30px;
#         margin-bottom: 10px;
#     }
#     .recommendation-card {
#         background-color: #ffffff;
#         border-radius: 10px;
#         padding: 15px;
#         margin-top: 10px;
#         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
#     }
#     .recommendation-card h3 {
#         color: #4b4b9b;
#     }
#     .recommendation-card p {
#         color: #555555;
#     }
#     </style>
#     """, unsafe_allow_html=True
# )

# # Load the pre-trained model for embedding
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # Load the CSV file
# df = pd.read_csv('travel_itineraries.csv')

# # Generate embeddings for the descriptions in the CSV file
# df['embedding'] = df['description'].apply(lambda x: model.encode(x))

# # Convert embeddings to a 2D numpy array for FAISS
# embeddings = np.stack(df['embedding'].values)

# # Create a FAISS index
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(embeddings)

# # OpenAI API key
# openai.api_key = "sk-proj-xwUxmOs0ZkUegvZYyBuuT3BlbkFJ4wK9AngasGlQ104aLrDb"

# # App Title
# st.markdown('<div class="title">Travel Itinerary Recommendation Chatbot</div>', unsafe_allow_html=True)

# # User Input
# st.markdown('<div class="input-box">Describe your ideal travel experience:</div>', unsafe_allow_html=True)
# user_input = st.text_input("")

# if user_input:
#     # Generate embeddings for the user input
#     user_vector = model.encode(user_input).reshape(1, -1)

#     # Query the FAISS index for the top K most similar entries
#     K = 3
#     distances, indices = index.search(user_vector, K)

#     st.markdown('<div class="section-title">Here is your travel itinerary:</div>', unsafe_allow_html=True)

#     # First show the exact match if found
#     exact_match = False
#     for idx in indices[0]:
#         itinerary = df.iloc[idx]
#         if user_input.lower() in itinerary['city'].lower():
#             st.markdown(
#                 f"""
#                 <div class="recommendation-card">
#                     <h3>City: {itinerary['city']}</h3>
#                     <p><strong>Activities:</strong> {itinerary['activities']}</p>
#                     <p><strong>Food:</strong> {itinerary['food']}</p>
#                     <p><strong>Accommodation:</strong> {itinerary['accommodation']}</p>
#                     <p><strong>Description:</strong> {itinerary['description']}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
#             exact_match = True
#             break

#     if not exact_match:
#         st.markdown('<div class="section-title">No exact match found. Here are some similar itineraries:</div>', unsafe_allow_html=True)

#     # Show other recommended cities
#     st.markdown('<div class="section-title">Other recommended cities to visit:</div>', unsafe_allow_html=True)
#     for idx in indices[0]:
#         itinerary = df.iloc[idx]
#         if user_input.lower() not in itinerary['city'].lower():
#             st.markdown(
#                 f"""
#                 <div class="recommendation-card">
#                     <h3>City: {itinerary['city']}</h3>
#                     <p><strong>Activities:</strong> {itinerary['activities']}</p>
#                     <p><strong>Food:</strong> {itinerary['food']}</p>
#                     <p><strong>Accommodation:</strong> {itinerary['accommodation']}</p>
#                     <p><strong>Description:</strong> {itinerary['description']}</p>
#                 </div>
#                 """, unsafe_allow_html=True)

import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import openai
from langchain_openai import OpenAI

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f5;
    }
    .title {
        font-size: 2.5em;
        color: #4b4b9b;
        text-align: center;
        margin-top: 20px;
    }
    .input-box {
        font-size: 1.2em;
        margin-top: 20px;
    }
    .section-title {
        font-size: 1.8em;
        color: #4b4b9b;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .recommendation-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .recommendation-card h3 {
        color: #4b4b9b;
    }
    .recommendation-card p {
        color: #555555;
    }
    </style>
    """, unsafe_allow_html=True
)

# Function to generate mock data
def generate_mock_data():
    # Example prompt engineering to generate travel itineraries
    prompt = """
    Generate detailed travel itineraries including city, activities, food, accommodation, and a brief description for the following cities: Paris, New York, Tokyo.
    """
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Load the pre-trained model for embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the CSV file or generate mock data
try:
    df = pd.read_csv('travel_itineraries.csv')
except FileNotFoundError:
    data = generate_mock_data()
    df = pd.DataFrame([entry.split(', ') for entry in data.split('\n')], columns=['city', 'activities', 'food', 'accommodation', 'description'])
    df.to_csv('travel_itineraries.csv', index=False)

# Generate embeddings for the descriptions in the CSV file
df['embedding'] = df['description'].apply(lambda x: model.encode(x))

# Convert embeddings to a 2D numpy array for FAISS
embeddings = np.stack(df['embedding'].values)

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# OpenAI API key
openai_api_key = "sk-proj-xwUxmOs0ZkUegvZYyBuuT3BlbkFJ4wK9AngasGlQ104aLrDb"
openai.api_key = openai_api_key

# Define the OpenAI model to use with LangChain
llm = OpenAI(api_key=openai_api_key)

# App Title
st.markdown('<div class="title">Travel Itinerary Recommendation Chatbot</div>', unsafe_allow_html=True)

# User Input
st.markdown('<div class="input-box">Describe your ideal travel experience:</div>', unsafe_allow_html=True)
user_input = st.text_input("")

if user_input:
    # Generate embeddings for the user input
    user_vector = model.encode(user_input).reshape(1, -1)

    # Query the FAISS index for the top K most similar entries
    K = 3
    distances, indices = index.search(user_vector, K)

    st.markdown('<div class="section-title">Here is your travel itinerary:</div>', unsafe_allow_html=True)

    # First show the exact match if found
    exact_match = False
    for idx in indices[0]:
        itinerary = df.iloc[idx]
        if user_input.lower() in itinerary['city'].lower():
            st.markdown(
                f"""
                <div class="recommendation-card">
                    <h3>City: {itinerary['city']}</h3>
                    <p><strong>Activities:</strong> {itinerary['activities']}</p>
                    <p><strong>Food:</strong> {itinerary['food']}</p>
                    <p><strong>Accommodation:</strong> {itinerary['accommodation']}</p>
                    <p><strong>Description:</strong> {itinerary['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            exact_match = True
            break

    if not exact_match:
        st.markdown('<div class="section-title">No exact match found. Here are some similar itineraries:</div>', unsafe_allow_html=True)

    # Show other recommended cities
    st.markdown('<div class="section-title">Other recommended cities to visit:</div>', unsafe_allow_html=True)
    for idx in indices[0]:
        itinerary = df.iloc[idx]
        if user_input.lower() not in itinerary['city'].lower():
            st.markdown(
                f"""
                <div class="recommendation-card">
                    <h3>City: {itinerary['city']}</h3>
                    <p><strong>Activities:</strong> {itinerary['activities']}</p>
                    <p><strong>Food:</strong> {itinerary['food']}</p>
                    <p><strong>Accommodation:</strong> {itinerary['accommodation']}</p>
                    <p><strong>Description:</strong> {itinerary['description']}</p>
                </div>
                """, unsafe_allow_html=True)