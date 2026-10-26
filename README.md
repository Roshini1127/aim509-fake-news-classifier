# AI-Based Fake News Classification System

## Project overview
This project uses machine learning and TF-IDF text features to classify
news text using a trained scikit-learn model.

## Files
- app.py: Streamlit web application
- fake_news_model.pkl: trained model pipeline
- news_dataset.csv: dataset used in the notebook
- model_comparison.csv: model evaluation comparison
- evaluation_report.txt: test-set evaluation
- requirements.txt: Python dependencies

## Run locally
1. Install Python.
2. Open a terminal in this folder.
3. Run: pip install -r requirements.txt
4. Run: streamlit run app.py

## Limitations
This is an educational prototype. Results depend on the dataset and can
be wrong for unfamiliar topics, satire, breaking news, or misleading text.
A model prediction is not proof that a news story is true or false.