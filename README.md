# RAG Chatbot
# Travel Itinerary Recommendation Chatbot

This is a Travel Itinerary Recommendation Chatbot built using Streamlit, FAISS, Sentence Transformers, and OpenAI. The chatbot provides personalized travel itineraries based on user input by leveraging Retrieval-Augmented Generation (RAG).

## Features

- User-friendly interface for entering travel preferences
- Personalized travel itinerary recommendations
- Uses Sentence Transformers for embedding generation
- Utilizes FAISS for efficient similarity search
- Integrates OpenAI for enhanced responses
- Stylish and modern interface with custom CSS

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Required Libraries

Install the required libraries using pip:

```sh
pip install openai faiss-cpu streamlit sentence-transformers pandas langchain-openai langchain-community
Setup
1.	Clone the repository:
 git clone <repository_url>
 cd <repository_directory>
2.	Create a virtual environment (optional but recommended):
 python -m venv myenv
 source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
##Running the App
To run the Streamlit app, use the following command:
 streamlit run chatgot.py
This will start the Streamlit server, and you can interact with the Travel Itinerary Recommendation Chatbot in your web browser.

File Structure

	•	app.py: Main script for running the Streamlit app.
	•	travel_itineraries.csv: CSV file containing travel itinerary data (if not found, mock data will be generated).

How It Works

	1.	User Input: The user inputs their travel preferences through a text box.
	2.	Embedding Generation: The input is converted into embeddings using Sentence Transformers.
	3.	Similarity Search: FAISS is used to search for the most similar itineraries based on the user’s input.
	4.	Recommendations: The top recommendations are displayed to the user, with an emphasis on exact matches and other similar options.

Customization

You can customize the styling of the app by modifying the CSS in the st.markdown section of the app.py file. You can also update the mock data generation prompt to include more specific or different information.

Video Link - https://youtu.be/juWUw1QJy-c
License

This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements

	•	Streamlit
	•	FAISS
	•	Sentence Transformers
	•	OpenAI
	•	LangChain

