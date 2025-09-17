import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from transformers import pipeline
import numpy as np

st.set_page_config(page_title="NLP Avanzado", layout="wide")

st.title("🔤 Procesamiento de Lenguaje Natural (NLP) con Python")

texto = st.text_area("✍️ Ingresa tu texto aquí:", "El gato duerme en el banco del parque.")

st.header("1. 🧺 Bolsa de Palabras (BoW)")

if texto:
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([texto])
    st.write("📌 Representación BoW (vector de frecuencia):")
    st.write(dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0])))

st.header("2. 🧠 Word Embeddings con Word2Vec")

example_sentences = [
    ["el", "gato", "duerme", "en", "el", "parque"],
    ["el", "perro", "corre", "por", "el", "jardín"],
    ["ella", "se", "sienta", "en", "el", "banco"],
    ["el", "banco", "abre", "a", "las", "nueve"]
]

# Entrenar un modelo Word2Vec simple
w2v_model = Word2Vec(sentences=example_sentences, vector_size=50, min_count=1, workers=1)

# Mostrar el vector de una palabra si existe
word = st.text_input("🔍 Ingresa una palabra para ver su embedding (ej: banco):", "banco")

if word in w2v_model.wv:
    st.write(f"📊 Vector para la palabra '{word}':")
    st.write(w2v_model.wv[word])
else:
    st.warning("Esa palabra no está en el vocabulario del modelo.")

st.header("3. 🤖 Transformers (usando Hugging Face)")

with st.spinner("Cargando modelo..."):
    classifier = pipeline("fill-mask", model="bert-base-uncased")

mask_text = st.text_input("🔄 Ingresa una frase con [MASK] (en inglés):", "The cat sat on the [MASK].")

if "[MASK]" in mask_text:
    results = classifier(mask_text)
    st.write("📌 Predicciones del modelo:")
    for r in results:
        st.write(f"- {r['sequence']} (score: {r['score']:.4f})")
else:
    st.warning("Agrega la palabra '[MASK]' para que el modelo funcione.")

st.info("Este demo usa BoW, Word2Vec (local) y BERT (preentrenado en inglés).")

