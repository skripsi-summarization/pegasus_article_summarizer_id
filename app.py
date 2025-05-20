import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from newspaper import Article
from googletrans import Translator
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
translator = Translator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.title("PEGASUS Indonesian Article Summarizer 🌍")
st.write("Summarizing Indonesian news using PEGASUS")

url = st.text_input("Paste the URL of the news article:")

if url:
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        st.subheader("Original Article")
        st.write(text)

        en_text = translator.translate(text, src='id', dest='en').text
        inputs = tokenizer(en_text, return_tensors="pt", truncation=True, max_length=512, padding="longest").to(device)
        summary_ids = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        en_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        id_summary = translator.translate(en_summary, src='en', dest='id').text

        st.subheader("Summary (Indonesian)")
        st.write(id_summary)

    except Exception as e:
        st.error(f"Error: {str(e)}")
