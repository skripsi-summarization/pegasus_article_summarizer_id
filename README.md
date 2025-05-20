# Indonesian News Summarization using PEGASUS

This app summarizes Indonesian news articles using a fine-tuned PEGASUS model on XL-Sum dataset.

## Features:
- Paste any Indonesian news article URL.
- Automatically scrapes the content.
- Generates a short summary:
  - PEGASUS app: Translates to English → Summarizes → Translates back.

## Built With:
- Hugging Face Transformers
- Google Translate API (`googletrans`)
- `newspaper3k` for article scraping
- Streamlit for UI
