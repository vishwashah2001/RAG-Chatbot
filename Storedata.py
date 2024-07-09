import pinecone
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize Pinecone with the new method
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key="7224a99c-0b56-4449-b0f1-8427b48dc461"
)

index_name = "travel-itineraries"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='gcp',
            region='us-west1'
        )
    )

index = pc.Index(index_name)

# Load the pre-trained model for embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Read the CSV file
df = pd.read_csv('travel_itineraries.csv')

# Generate embeddings and store them in Pinecone
for i, row in df.iterrows():
    vector = model.encode(row['description']).tolist()
    index.upsert([(str(i), vector, row.to_dict())])