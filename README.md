# Indonesian News Summarization (PEGASUS/mBART)

This app summarizes Indonesian news articles using a fine-tuned Transformer model.

## Features:
- Paste any Indonesian news article URL.
- Automatically scrapes the content.
- Generates a short summary:
  - PEGASUS app: Translates to English → Summarizes → Translates back.
  - mBART app: Direct multilingual summarization.

## Built With:
- Hugging Face Transformers
- Google Translate API (`googletrans`)
- `newspaper3k` for article scraping
- Streamlit for UI
