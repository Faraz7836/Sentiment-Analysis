# import streamlit as st
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# import matplotlib.pyplot as plt

# # Download VADER lexicon
# nltk.download("vader_lexicon")

# # Initialize Sentiment Analyzer
# sia = SentimentIntensityAnalyzer()

# # Streamlit UI
# st.title("Sentiment Analysis App")
# st.write("Enter a sentence to analyze its sentiment.")

# # User input
# user_input = st.text_area("Enter your text:", "")

# if st.button("Analyze Sentiment"):
#     if user_input:
#         # Perform sentiment analysis
#         sentiment_scores = sia.polarity_scores(user_input)
        
#         # Determine sentiment label
#         if sentiment_scores["compound"] >= 0.05:
#             sentiment_label = "Positive 😊"
#         elif sentiment_scores["compound"] <= -0.05:
#             sentiment_label = "Negative 😡"
#         else:
#             sentiment_label = "Neutral 😐"

#         # Display results
#         st.subheader("Sentiment Analysis Result")
#         st.write(f"Overall Sentiment:{sentiment_label}")
#         st.write(f"Positive: {sentiment_scores['pos']:.2f}, Negative: {sentiment_scores['neg']:.2f}, Neutral: {sentiment_scores['neu']:.2f}")

#         # Bar chart visualization
#         fig, ax = plt.subplots()
#         ax.bar(["Positive", "Negative", "Neutral"], 
#                [sentiment_scores["pos"], sentiment_scores["neg"], sentiment_scores["neu"]], 
#                color=["green", "red", "blue"])
#         ax.set_ylabel("Score")
#         ax.set_title("Sentiment Score Distribution")
#         st.pyplot(fig)
#     else:
#         st.warning("Please enter text for analysis!")





# import streamlit as st
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# from transformers import pipeline
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Download VADER lexicon
# nltk.download("vader_lexicon")

# # Initialize sentiment analyzers
# sia = SentimentIntensityAnalyzer()
# roberta_sentiment = pipeline("sentiment-analysis")

# # Streamlit UI Configuration
# st.set_page_config(page_title="Review Sentiment Analysis | VADER vs. RoBERTa", layout="wide")

# # App Title
# st.title("Sentiment AnalysisS")  #🔍 
# st.markdown("Compare sentiment analysis results between two Deep Learning models: VADER and RoBERTa.")

# # User Input
# user_input = st.text_area("Enter your text for analysis:")

# if st.button("Analyze Sentiment"):
#     if user_input:
#         # VADER Sentiment Analysis
#         vader_scores = sia.polarity_scores(user_input)
#         vader_label = "Positive" if vader_scores["compound"] >= 0.05 else "Negative" if vader_scores["compound"] <= -0.05 else "Neutral"

#         # RoBERTa Sentiment Analysis
#         roberta_result = roberta_sentiment(user_input)[0]
#         roberta_label = roberta_result["label"]
#         roberta_score = roberta_result["score"]

#         # Display results side by side
#         col1, col2 = st.columns(2)

#         with col1:
#             st.subheader("🔹 VADER Analysis")
#             st.write(f"**Sentiment:** {vader_label}({vader_scores['compound']:.2f})")
#             # st.write(f"🔹 Positive: {vader_scores['pos']:.2f}, Negative: {vader_scores['neg']:.2f}, Neutral: {vader_scores['neu']:.2f}")
#             st.progress(abs(vader_scores["compound"]))  # Progress bar

#         with col2:
#             st.subheader("🔹 RoBERTa Analysis")
#             st.write(f"**Sentiment:** {roberta_label} ({roberta_score:.2f})")
#             st.progress(roberta_score)  # Progress bar

#         # Visualization
#         st.subheader("📊 Sentiment Score Comparison")
#         fig, ax = plt.subplots(figsize=(7, 4))
#         sns.barplot(x=["Positive", "Negative", "Neutral"], 
#                     y=[vader_scores["pos"], vader_scores["neg"], vader_scores["neu"]], 
#                     palette=["green", "red", "blue"])
#         plt.title("VADER Sentiment Distribution")
#         st.pyplot(fig)

#     else:
#         st.warning("Please enter text for analysis!")


import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK's VADER lexicon
nltk.download("vader_lexicon")

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load RoBERTa Model & Tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to perform RoBERTa sentiment analysis
def analyze_roberta_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    scores = outputs.logits.detach().numpy()[0]
    scores = softmax(scores)  # Convert logits to probabilities
    return {"negative": scores[0], "neutral": scores[1], "positive": scores[2]}

# Streamlit UI Configuration
st.set_page_config(page_title="Sentiment Analysis | VADER vs. RoBERTa", layout="wide")

# App Title
# st.title("Review Sentiment Analysis")


st.markdown("<h1 style='text-align: center;'>Review Sentiment Analysis by Faraz</h1>", unsafe_allow_html=True)

st.markdown("Compare sentiment analysis results from **VADER** (Lexicon-based) and **RoBERTa** (Deep Learning-based).")

# User Input
user_input = st.text_area("Enter your text for analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        # VADER Sentiment Analysis
        vader_scores = sia.polarity_scores(user_input)
        vader_label = "Positive" if vader_scores["compound"] >= 0.05 else "Negative" if vader_scores["compound"] <= -0.05 else "Neutral"

        # RoBERTa Sentiment Analysis
        roberta_scores = analyze_roberta_sentiment(user_input)
        roberta_label = max(roberta_scores, key=roberta_scores.get).capitalize()

        # Display results side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔹 VADER Analysis")
            st.write(f"**Sentiment:** {vader_label}")
            st.write(f"🔹 Positive: {vader_scores['pos']:.2f}, Negative: {vader_scores['neg']:.2f}, Neutral: {vader_scores['neu']:.2f}")
            # st.progress(abs(vader_scores["compound"]))  # Progress bar
            # Calculate VADER confidence as the absolute value of the compound score
            vader_confidence = abs(vader_scores["compound"]) * 100  # Convert to percentage
            st.metric(label="VADER Confidence", value=f"{vader_confidence:.2f}%")


        with col2:
            st.subheader("🔹 RoBERTa Analysis")
            st.write(f"**Sentiment:** {roberta_label}")
            st.write(f"🔹 Positive: {roberta_scores['positive']:.2f}, Negative: {roberta_scores['negative']:.2f}, Neutral: {roberta_scores['neutral']:.2f}")
            st.metric(label="RoBERTa Confidence", value=f"{max(roberta_scores.values())*100:.2f}%")
              # Show highest confidence score

        # Visualization Section
        # st.subheader("Sentiment Score Comparison ",divider="rainbow")

        st.markdown("<h2 style='text-align: center;'>Sentiment Score Comparison</h2>", unsafe_allow_html=True)


        col3, col4 = st.columns(2)

        with col3:
            st.write("**VADER Sentiment Distribution**")
            fig_vader, ax1 = plt.subplots(figsize=(5, 3))
            sns.barplot(x=["Positive", "Negative", "Neutral"], 
                        y=[vader_scores["pos"], vader_scores["neg"], vader_scores["neu"]], 
                        palette=["green", "red", "blue"], ax=ax1)
            ax1.set_ylabel("Score")
            st.pyplot(fig_vader)

        with col4:
            st.write("**RoBERTa Sentiment Distribution**")
            fig_roberta, ax2 = plt.subplots(figsize=(5, 3))
            sns.barplot(x=["Positive", "Negative", "Neutral"], 
                        y=[roberta_scores["positive"], roberta_scores["negative"], roberta_scores["neutral"]], 
                        palette=["green", "red", "blue"], ax=ax2)
            ax2.set_ylabel("Confidence Score")
            st.pyplot(fig_roberta)

    else:
        st.warning("Please enter text for analysis!") # Call the function to analyze sentiment        
