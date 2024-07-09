import pandas as pd
import os

# Print the current working directory
print("Current working directory:", os.getcwd())

sample_data = [
    {"city": "Paris", "activities": "Eiffel Tower, Louvre Museum, Seine River Cruise", "food": "Croissants, Baguette, Macarons", "accommodation": "Hotel Le Bristol, Hotel Lutetia", "description": "A romantic getaway with iconic landmarks and delicious French cuisine."},
    {"city": "New York", "activities": "Statue of Liberty, Central Park, Broadway Shows", "food": "Pizza, Bagels, Cheesecake", "accommodation": "The Plaza, The Ritz-Carlton", "description": "A bustling city experience with famous landmarks and diverse food options."},
    {"city": "Tokyo", "activities": "Tokyo Tower, Shinjuku Gyoen, Akihabara", "food": "Sushi, Ramen, Tempura", "accommodation": "Park Hyatt Tokyo, The Ritz-Carlton Tokyo", "description": "A mix of traditional and modern experiences with exquisite Japanese cuisine."},
    {"city": "Sydney", "activities": "Sydney Opera House, Bondi Beach, Taronga Zoo", "food": "Seafood, Meat Pies, Lamingtons", "accommodation": "Four Seasons Hotel, Shangri-La Hotel", "description": "A vibrant city with stunning beaches and world-famous landmarks."},
    {"city": "Rome", "activities": "Colosseum, Vatican Museums, Trevi Fountain", "food": "Pasta, Pizza, Gelato", "accommodation": "Hotel Hassler, Hotel de Russie", "description": "An ancient city with a rich history and mouth-watering Italian cuisine."},
    {"city": "London", "activities": "London Eye, Buckingham Palace, British Museum", "food": "Fish and Chips, Afternoon Tea, Roast Dinner", "accommodation": "The Savoy, The Ritz London", "description": "A cosmopolitan city with historical sites and diverse culinary options."},
    {"city": "Barcelona", "activities": "Sagrada Familia, Park Güell, La Rambla", "food": "Tapas, Paella, Churros", "accommodation": "Hotel Arts, W Barcelona", "description": "A city with unique architecture, vibrant culture, and delectable Spanish food."},
    {"city": "Bangkok", "activities": "Grand Palace, Wat Arun, Chatuchak Market", "food": "Pad Thai, Tom Yum, Mango Sticky Rice", "accommodation": "Mandarin Oriental, The Siam Hotel", "description": "A city with a blend of traditional and modern attractions and flavorful Thai cuisine."},
    {"city": "Dubai", "activities": "Burj Khalifa, Desert Safari, Dubai Mall", "food": "Shawarma, Falafel, Manakish", "accommodation": "Burj Al Arab, Atlantis The Palm", "description": "A luxurious destination with impressive skyscrapers and diverse food options."},
    {"city": "Cape Town", "activities": "Table Mountain, Robben Island, V&A Waterfront", "food": "Braai, Bobotie, Biltong", "accommodation": "The Silo, One&Only Cape Town", "description": "A city with stunning landscapes, rich history, and delicious South African cuisine."},
    {"city": "Istanbul", "activities": "Hagia Sophia, Blue Mosque, Grand Bazaar", "food": "Kebabs, Baklava, Meze", "accommodation": "Four Seasons Hotel Istanbul, Ciragan Palace Kempinski", "description": "A city where East meets West, offering historical sites and delightful Turkish cuisine."},
    {"city": "Rio de Janeiro", "activities": "Christ the Redeemer, Copacabana Beach, Sugarloaf Mountain", "food": "Feijoada, Churrasco, Açaí", "accommodation": "Belmond Copacabana Palace, Hotel Fasano", "description": "A vibrant city with beautiful beaches, lively festivals, and delicious Brazilian food."},
    {"city": "Singapore", "activities": "Marina Bay Sands, Gardens by the Bay, Sentosa Island", "food": "Chili Crab, Hainanese Chicken Rice, Laksa", "accommodation": "Raffles Hotel, Marina Bay Sands", "description": "A modern city with futuristic architecture, green spaces, and a variety of delicious food."},
    {"city": "Buenos Aires", "activities": "La Boca, Teatro Colon, Palermo", "food": "Asado, Empanadas, Dulce de Leche", "accommodation": "Alvear Palace Hotel, Four Seasons Hotel Buenos Aires", "description": "A city with European charm, tango dancing, and flavorful Argentine cuisine."},
    {"city": "San Francisco", "activities": "Golden Gate Bridge, Alcatraz Island, Fisherman's Wharf", "food": "Sourdough Bread, Dungeness Crab, Clam Chowder", "accommodation": "The Fairmont, Ritz-Carlton San Francisco", "description": "A city with iconic landmarks, scenic views, and a diverse food scene."},
]

print("Creating DataFrame...")
df = pd.DataFrame(sample_data)
print("Saving to CSV...")
df.to_csv('travel_itineraries.csv', index=False)
print("CSV file created successfully.")