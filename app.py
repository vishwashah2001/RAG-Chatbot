# import re
# import pandas as pd
# import streamlit as st
# from sentence_transformers import SentenceTransformer, util
# import openai

# # Set OpenAI API key
# openai.api_key = 'sk-proj-xwUxmOs0ZkUegvZYyBuuT3BlbkFJ4wK9AngasGlQ104aLrDb'

# # Load and preprocess the data
# file_path = 'spotify-2023.csv'  # Ensure the CSV file is in the same directory as the script

# # Try to load the CSV file and handle potential errors
# try:
#     music_data = pd.read_csv(file_path, encoding='latin1')
# except Exception as e:
#     st.error(f"Error loading CSV file: {e}")
#     st.stop()

# # Preprocessing function
# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     return text

# # Apply preprocessing to the track names and artist names
# music_data['track_name'] = music_data['track_name'].apply(preprocess)
# music_data['artist(s)_name'] = music_data['artist(s)_name'].apply(preprocess)

# # Combine track name and artist(s) name for a more descriptive text
# music_data['description'] = music_data['track_name'] + " by " + music_data['artist(s)_name']

# # Initialize the SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Generate embeddings for the descriptions
# music_data['embeddings'] = music_data['description'].apply(lambda x: model.encode(x))

# # Streamlit interface
# st.title("Music Recommendation Chatbot")
# user_input = st.text_input("Enter your music preferences:")

# if user_input:
#     try:
#         # Generate embedding for user input
#         query_embedding = model.encode(user_input)

#         # Calculate cosine similarity between the query and the music descriptions
#         music_data['similarity'] = music_data['embeddings'].apply(lambda x: util.pytorch_cos_sim(query_embedding, x).item())

#         # Get the top 3 recommendations based on similarity
#         top_recommendations = music_data.nlargest(3, 'similarity')

#         # Generate a response using OpenAI GPT-3.5
#         recommendations_text = "\n".join(top_recommendations['description'].tolist())
#         openai_response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful music recommendation assistant."},
#                 {"role": "user", "content": f"Based on the following music preferences: {user_input}\n\nI recommend these songs:\n{recommendations_text}"}
#             ]
#         )

#         # Display recommendations
#         st.write("Recommendations:")
#         st.write(openai_response.choices[0].message['content'].strip())
#     except Exception as e:
#         st.error(f"Error processing request: {e}")
import re
import pandas as pd
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util
import openai
import faiss

# Set OpenAI API key
openai.api_key = 'sk-proj-xwUxmOs0ZkUegvZYyBuuT3BlbkFJ4wK9AngasGlQ104aLrDb'

# Load and preprocess the data
file_path = 'spotify-2023.csv'  # Ensure the CSV file is in the same directory as the script

# Try to load the CSV file and handle potential errors
try:
    music_data = pd.read_csv(file_path, encoding='latin1')
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Apply preprocessing to the track names and artist names
music_data['track_name'] = music_data['track_name'].apply(preprocess)
music_data['artist(s)_name'] = music_data['artist(s)_name'].apply(preprocess)

# Combine track name and artist(s) name for a more descriptive text
music_data['description'] = music_data['track_name'] + " by " + music_data['artist(s)_name']

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the descriptions
music_data['embeddings'] = music_data['description'].apply(lambda x: model.encode(x))

# Prepare embeddings for FAISS
embeddings = np.vstack(music_data['embeddings'].values).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Store metadata with embeddings in a dictionary
metadata = music_data[['description']].to_dict(orient='records')

# Streamlit interface
st.title("Music Recommendation Chatbot")
user_input = st.text_input("Enter your music preferences:")

if user_input:
    try:
        # Generate embedding for user input
        query_embedding = model.encode(user_input)

        # Perform similarity search
        D, I = index.search(np.array([query_embedding]), k=3)
        
        # Retrieve top K results
        top_k_results = [metadata[i] for i in I[0]]

        # Generate a response using OpenAI GPT-3.5
        recommendations = "\n".join([result['description'] for result in top_k_results])
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful music recommendation assistant."},
                {"role": "user", "content": f"Based on the following music preferences: {user_input}\n\nI recommend these songs:\n{recommendations}"}
            ]
        )

        # Display recommendations
        st.write("Recommendations:")
        st.write(openai_response.choices[0].message['content'].strip())
    except Exception as e:
        st.error(f"Error processing request: {e}")