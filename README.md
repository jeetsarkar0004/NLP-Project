# Twitter Sentiment Analysis - Review Sentiment Predictor 

## Project Overview
This project implements a complete end-to-end sentiment analysis pipeline using Natural Language Processing (NLP) techniques. The system is trained on a Twitter dataset and deployed as an interactive web application using **Streamlit**. 

## Key Features
* **Real-time Sentiment Prediction:** Instantly analyze any text input.
* **Batch CSV Prediction:** Upload reviewed tweet or review exports and download scored results.
* **Interactive Web Interface:** User-friendly UI with immediate visual feedback.
* **High Accuracy:** Achieves **86.56%** accuracy on unseen test data.
* **Confidence Scoring:** Displays prediction confidence percentages.
* **Probability Breakdown:** Shows detailed probabilities for both positive and negative sentiment classes.
* **Xquik Export Compatibility:** Detects common Xquik and Twitter/X text columns such as `text`, `Tweet`, `content`, and `body`.

## Model Performance

| Metric | Negative | Positive | Overall |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.86 | 0.87 | **86.56%** |
| **Recall** | 0.87 | 0.87 | - |
| **F1-Score** | 0.86 | 0.87 | - |

## Using the Predictor
1. Enter a review or tweet in the text area provided in the web app.
2. Click the **"Predict Sentiment"** button.
3. View the results with intuitive color-coded feedback:
   * ✅ **Green:** Positive sentiment
   * ❌ **Red:** Negative sentiment
4. Review the confidence percentage and detailed probability breakdown generated below the prediction.

## Batch CSV Workflow
1. Start the app and open the **Batch CSV Prediction** section.
2. Upload a CSV file exported from Xquik or another reviewed Twitter/X source.
3. Select the text column if it is not detected automatically.
4. Download the CSV with `predicted_sentiment` and `confidence` columns.

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Acknowledgments
* **Twitter** for providing the public dataset.
* The **NLTK team** for their excellent Natural Language Processing toolkit.
* The **Scikit-learn community** for robust machine learning tools.
* The **Streamlit team** for creating an amazing web deployment framework.
