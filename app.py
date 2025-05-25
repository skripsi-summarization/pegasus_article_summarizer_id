import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from newspaper import Article
from googletrans import Translator
import torch

# Cache model and tokenizer
@st.cache_resource
def load_summarizer():
    model = PegasusForConditionalGeneration.from_pretrained("skripsi-summarization-1234/pegasus-xsum-finetuned-xlsum-summarization")
    tokenizer = PegasusTokenizer.from_pretrained("skripsi-summarization-1234/pegasus-xsum-finetuned-xlsum-summarization")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_summarizer()
translator = Translator()

# Page setup
st.set_page_config(page_title="Indonesian News Summarizer", layout="centered")

# --- Styling ---
st.markdown("""
    <style>
    .scroll-box {
        background-color: #fff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        max-height: 300px;
        overflow-y: scroll;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .summary-box {
        background: linear-gradient(135deg, #f0f7ff, #e0efff);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #a2caff;
        font-size: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align:center; color:#0d47a1;'>📰 Indonesian News Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ringkas berita Indonesia secara otomatis hanya dengan menempelkan URL artikel.</p>", unsafe_allow_html=True)

# --- Input ---
st.markdown("### 🔗 Masukkan URL Berita")
url = st.text_input("", placeholder="https://www.cnnindonesia.com/...", label_visibility="collapsed")

col1, col2 = st.columns(2)
with col1:
    show_btn = st.button("📥 Tampilkan Artikel", use_container_width=True)
with col2:
    summarize_btn = st.button("✍️ Ringkas", use_container_width=True)

# --- Show Article ---
if show_btn:
    if url:
        try:
            article = Article(url, language='id')
            article.download()
            article.parse()
            st.session_state.article_text = article.text

            st.markdown("### 📄 Artikel Lengkap")
            st.markdown(f"<div class='scroll-box'>{st.session_state.article_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Gagal mengambil artikel: {str(e)}")
    else:
        st.warning("⚠️ Mohon masukkan URL yang valid.")

# --- Summarize ---
if summarize_btn:
    if "article_text" in st.session_state:
        try:
            with st.spinner("🔄 Memproses artikel..."):
                # Abstracting the internal translation process
                en_text = translator.translate(st.session_state.article_text, src='id', dest='en').text
                inputs = tokenizer(en_text, return_tensors="pt", truncation=True, max_length=512, padding="longest").to(device)
                summary_ids = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                en_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                id_summary = translator.translate(en_summary, src='en', dest='id').text

            st.markdown("### 🔍 Hasil Ringkasan")
            st.markdown(f"<div class='summary-box'>{id_summary}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat merangkum: {str(e)}")
    else:
        st.warning("⚠️ Silakan tampilkan artikel terlebih dahulu.")
