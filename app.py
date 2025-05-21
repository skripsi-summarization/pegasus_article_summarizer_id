# PEGASUS app.py

import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from newspaper import Article
from googletrans import Translator
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PegasusForConditionalGeneration.from_pretrained("skripsi-summarization-1234/pegasus-xsum-finetuned-xlsum-summarization")
model = model.to(device)
tokenizer = PegasusTokenizer.from_pretrained("skripsi-summarization-1234/pegasus-xsum-finetuned-xlsum-summarization")
translator = Translator()

# Streamlit layout improvements
st.set_page_config(page_title="PEGASUS News Summarizer", layout="centered")
st.title("📰 PEGASUS News Summarizer 🇮🇩")
st.markdown("""
This app summarizes **Indonesian news articles** using a fine-tuned [PEGASUS](https://huggingface.co/skripsi-summarization-1234/pegasus-xsum-finetuned-xlsum-summarization) model.

**Workflow:** Translate ➜ Summarize ➜ Translate Back
""")

# Input section
url = st.text_input("📎 Paste the URL of the Indonesian news article:")

if url:
    try:
        with st.spinner("📥 Downloading and parsing article..."):
            article = Article(url)
            article.download()
            article.parse()
            text = article.text

        st.subheader("📰 Original Article")
        st.write(text)

        with st.spinner("🌐 Translating to English..."):
            en_text = translator.translate(text, src='id', dest='en').text

        with st.spinner("🤖 Summarizing in English using PEGASUS..."):
            inputs = tokenizer(en_text, return_tensors="pt", truncation=True, max_length=512, padding="longest")
            summary_ids = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
            en_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        with st.spinner("🌐 Translating summary back to Indonesian..."):
            id_summary = translator.translate(en_summary, src='en', dest='id').text

        st.success("✅ Summary generated successfully!")
        st.subheader("🔎 Ringkasan Berita")
        st.write(id_summary)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
