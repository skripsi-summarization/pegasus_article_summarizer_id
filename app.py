import streamlit as st
from newspaper import Article
from transformers import pipeline, PegasusTokenizer, PegasusForConditionalGeneration

# Load model using pipeline and cache
@st.cache_resource
def load_summarizer():
    model_name = "skripsi-summarization-1234/pegasus-xsum-finetuned-xlsum-summarization"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    return pipeline("summarization", model=model, tokenizer=tokenizer)

summarizer = load_summarizer()

# Streamlit UI
st.set_page_config(page_title="PEGASUS Indonesian News Summarizer")
st.title("📰 PEGASUS Indonesian News Summarizer")
st.write("Enter a URL from an Indonesian news site (e.g., Detik, Kompas) to summarize its content.")

# Input: URL
url = st.text_input("Paste the article URL here:")

# Fetch article text
if st.button("Show Article Text"):
    if url:
        try:
            article = Article(url, language='id')
            article.download()
            article.parse()
            st.subheader("📄 Full Article")
            st.write(article.text)
            st.session_state.article_text = article.text
        except Exception as e:
            st.error(f"❌ Failed to fetch article: {str(e)}")
    else:
        st.warning("⚠️ Please input a valid URL.")

# Summarize
if st.button("Summarize"):
    if "article_text" in st.session_state:
        with st.spinner("✍️ Summarizing..."):
            # Optional: prepend instruction if fine-tuned with one
            input_text = st.session_state.article_text
            summary = summarizer(input_text, max_length=128, min_length=40, do_sample=False)
            st.subheader("📝 Summary")
            st.success(summary[0]['summary_text'])
    else:
        st.warning("⚠️ No article text found. Please load the article first.")
