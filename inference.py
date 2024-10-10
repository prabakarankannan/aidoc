import streamlit as st
import faiss
import numpy as np
import pandas as pd
import requests
import json
from sentence_transformers import SentenceTransformer

# 1. Load the FAISS index and embeddings model (keeps using SentenceTransformers for embeddings)
def load_faiss_index(index_path, embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings_model = SentenceTransformer(embeddings_model_name)
    index = faiss.read_index(index_path)
    return index, embeddings_model

# 2. Function to query the FAISS index
def query_faiss(faiss_index, embeddings_model, query, k=5):
    query_embedding = embeddings_model.encode(query)
    distances, indices = faiss_index.search(np.array([query_embedding]), k)
    return indices, distances

# 3. Combine user's question with the returned results from FAISS
def combine_question_with_results(user_question, faiss_results, dataset):
    combined_question = user_question + ".\nContext \nHere are related symptoms and diseases:\n"
    previous_disease = ""
    for idx in faiss_results:
        disease = dataset.iloc[idx]['Disease']
        symptoms = dataset.iloc[idx][1:].dropna().tolist()
        if previous_disease == disease:
            pass
        else: 
            combined_question += f"- Disease: {disease}, Symptoms: {', '.join(symptoms)}\n"
            previous_disease = disease
    return combined_question

# 4. Function to interact with SambaNova API
def query_sambanova(combined_question, api_key, model="Meta-Llama-3.2-3B-Instruct"):
    """
    Query the SambaNova LLM with a combined question using their API.
    """
    url = "https://api.sambanova.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "stream": False,  # You can enable streaming if needed
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": combined_question
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response_data = response.json()

    return response_data['choices'][0]['message']['content']

# Streamlit app for the chatbot
def run_chatbot_with_streamlit(index_path, dataset_path, api_key):
    # 1. Load the FAISS index and dataset
    index, embeddings_model = load_faiss_index(index_path)
    dataset = pd.read_csv(dataset_path)

    # 2. Streamlit UI
    st.title("AI Doctor Chat Assistant")
    user_question = st.text_input("Please enter your symptoms:")

    if user_question:
        # 3. Query FAISS index
        faiss_results, distances = query_faiss(index, embeddings_model, user_question)

        # 4. Combine the user's question with retrieved FAISS results
        combined_question = combine_question_with_results(user_question, faiss_results[0], dataset)
        st.write(f"Query sent to LLM: {combined_question}")
        
        # 5. Query the SambaNova model with the combined question
        if st.button("Get Diagnosis"):
            llm_response = query_sambanova(combined_question, api_key) 
            st.write("\nDoctor's Response:\n")
            st.write(llm_response)

# Run the Streamlit app
if __name__ == "__main__":
    api_key = "f9bfc63a-b8c3-4cd4-b93b-11a70d25788e"  # Replace with your actual API key
    run_chatbot_with_streamlit("vector_db/disease_symptoms_index.faiss", "dataset/symptoms_dataset_spaces.csv", api_key)
