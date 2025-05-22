import streamlit as st
st.set_page_config(page_title="PEGASUS Indonesian News Summarizer")

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
tokenizer = summarizer.tokenizer

# Truncate text safely to avoid runtime errors
def truncate_text(text, tokenizer, max_tokens=512):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

# Streamlit UI
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
            try:
                raw_text = st.session_state.article_text
                input_text = truncate_text(raw_text, tokenizer, max_tokens=512)

                if len(tokenizer(raw_text)["input_ids"]) > 512:
                    st.warning("⚠️ Article too long — only the first 512 tokens were summarized.")

                summary = summarizer(input_text, max_length=128, min_length=40, do_sample=False)
                st.subheader("📝 Summary")
                st.success(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"❌ Summarization failed: {str(e)}")
    else:
        st.warning("⚠️ No article text found. Please load the article first.")
