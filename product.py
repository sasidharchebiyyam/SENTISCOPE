import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from collections import defaultdict

st.set_page_config(
    page_title="AI Product Review Intelligence System",
    layout="wide"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #0F172A,
        #111827,
        #1E293B
    );
    color: white;
}

h1 {
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    text-align: center;
}

h2, h3, h4, h5, h6 {
    color: #F8FAFC !important;
    font-weight: 600 !important;
}

p, label {
    color: #CBD5E1 !important;
}

textarea {
    font-size: 16px !important;
    border-radius: 15px !important;
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #334155 !important;
}

.stTextInput input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}

.stButton > button {

    width: 100%;
    height: 55px;

    background: linear-gradient(
        90deg,
        #6366F1,
        #8B5CF6
    );

    color: white;
    border: none;
    border-radius: 14px;

    font-size: 18px;
    font-weight: 600;

    transition: 0.3s;
}

.stButton > button:hover {

    transform: scale(1.02);

    background: linear-gradient(
        90deg,
        #7C3AED,
        #2563EB
    );

    color: white;
}

[data-testid="metric-container"] {

    background: linear-gradient(
        145deg,
        #1E293B,
        #0F172A
    );

    border: 1px solid #334155;

    padding: 25px;

    border-radius: 18px;

    box-shadow:
        0 0 15px rgba(99,102,241,0.25);

    transition: 0.3s;
}

[data-testid="metric-container"]:hover {

    transform: translateY(-5px);

    box-shadow:
        0 0 25px rgba(139,92,246,0.45);
}

.stDataFrame {

    border-radius: 15px;
    overflow: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    return sentiment_model

sentiment_model = load_model()

FEATURE_KEYWORDS = {
    "Camera": [
        "camera", "photo", "picture", "selfie"
    ],

    "Battery": [
        "battery", "charging", "backup", "drain"
    ],

    "Performance": [
        "performance", "processor", "speed",
        "lag", "smooth", "fast"
    ],

    "Display": [
        "display", "screen", "brightness"
    ],

    "Design": [
        "design", "build", "premium", "look"
    ],

    "Heating": [
        "heat", "heating", "hot"
    ],

    "Sound": [
        "speaker", "sound", "audio"
    ]
}

st.title("🛒 Product Review Intelligence System")
st.caption("AI Powered Product Feedback & Feature Analysis Dashboard")

st.sidebar.title("📌 Features")

st.sidebar.info("""
✔ High Accuracy Sentiment Analysis

✔ AI Generated Summary

✔ Feature-Based Sentiment Analysis

✔ Automatic Product Rating

✔ Business Insights

✔ Interactive Dashboard

✔ Downloadable Reports
""")

product_name = st.text_input(
    "📦 Enter Product Name"
)

reviews_input = st.text_area(
    "✍ Paste Reviews (One Review Per Line)",
    height=250,
    placeholder="""
Amazing camera quality and smooth performance
Battery backup is excellent for daily use
The mobile heats while gaming
Very poor charging speed
Excellent display and premium design
Camera quality is bad in low light
Worth buying for this price
Speaker sound is too low
Fast processor and smooth multitasking
Battery drains quickly after update
"""
)

analyze = st.button("🚀 Analyze Reviews")

if analyze:

    if product_name == "":
        st.warning("Please enter product name")
        st.stop()

    if reviews_input == "":
        st.warning("Please enter reviews")
        st.stop()

    reviews = [
        review.strip()
        for review in reviews_input.split("\n")
        if review.strip()
    ]

    sentiments = []
    confidence_scores = []
    ratings = []

    progress = st.progress(0)

    for i, review in enumerate(reviews):

        result = sentiment_model(review[:512])[0]

        label = result["label"]
        score = result["score"]

        if label in ["LABEL_2", "positive"]:

            sentiment = "Positive"
            rating = 5

        elif label in ["LABEL_1", "neutral"]:

            sentiment = "Neutral"
            rating = 3

        else:

            sentiment = "Negative"
            rating = 1

        sentiments.append(sentiment)
        confidence_scores.append(round(score * 100, 2))
        ratings.append(rating)

        progress.progress((i + 1) / len(reviews))

    df = pd.DataFrame({
        "Review": reviews,
        "Sentiment": sentiments,
        "Confidence": confidence_scores,
        "Rating": ratings
    })

    average_rating = round(df["Rating"].mean(), 2)

    positive_reviews = (
        df["Sentiment"] == "Positive"
    ).sum()

    negative_reviews = (
        df["Sentiment"] == "Negative"
    ).sum()

    st.divider()

    st.subheader(f"📊 Analysis Results for {product_name}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "⭐ Average Rating",
        average_rating
    )

    col2.metric(
        "📝 Total Reviews",
        len(reviews)
    )

    col3.metric(
        "😊 Positive Reviews",
        positive_reviews
    )

    col4.metric(
        "😡 Negative Reviews",
        negative_reviews
    )

    st.divider()

    st.subheader("🧠 AI Generated Summary")

    positive_points = []
    negative_points = []

    for review, sentiment in zip(reviews, sentiments):

        if sentiment == "Positive":
            positive_points.append(review)

        elif sentiment == "Negative":
            negative_points.append(review)

    positive_summary = ", ".join(positive_points[:3])
    negative_summary = ", ".join(negative_points[:3])

    summary = f"""
Customers appreciated features such as {positive_summary}.

However, some users reported issues including {negative_summary}.
"""

    st.success(summary)

    st.divider()

    st.subheader("📈 Sentiment Distribution")

    sentiment_counts = (
        df["Sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    fig1 = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        hole=0.5
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.divider()

    st.subheader("⭐ Feature-Based Sentiment Analysis")

    feature_results = defaultdict(list)

    for review, sentiment in zip(reviews, sentiments):

        review_lower = review.lower()

        for feature, keywords in FEATURE_KEYWORDS.items():

            for keyword in keywords:

                if keyword in review_lower:

                    feature_results[feature].append(sentiment)

    feature_summary = []

    for feature, sentiment_list in feature_results.items():

        positive_count = sentiment_list.count("Positive")
        negative_count = sentiment_list.count("Negative")

        if positive_count > negative_count:
            feature_sentiment = "Positive"

        elif negative_count > positive_count:
            feature_sentiment = "Negative"

        else:
            feature_sentiment = "Neutral"

        feature_summary.append({
            "Feature": feature,
            "Positive Reviews": positive_count,
            "Negative Reviews": negative_count,
            "Overall Sentiment": feature_sentiment
        })

    feature_df = pd.DataFrame(feature_summary)

    st.dataframe(
        feature_df,
        use_container_width=True
    )

    fig2 = px.bar(
        feature_df,
        x="Feature",
        y=["Positive Reviews", "Negative Reviews"],
        barmode="group"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    st.subheader("🤖 Automatic Product Rating")

    if average_rating >= 4.5:

        st.success("★★★★★ Excellent Product")

    elif average_rating >= 4:

        st.success("★★★★☆ Very Good Product")

    elif average_rating >= 3:

        st.warning("★★★☆☆ Average Product")

    elif average_rating >= 2:

        st.error("★★☆☆☆ Poor Product")

    else:

        st.error("★☆☆☆☆ Very Bad Product")

    st.divider()

    st.subheader("💡 Business Insights")

    negative_features = feature_df[
        feature_df["Overall Sentiment"] == "Negative"
    ]["Feature"].tolist()

    positive_features = feature_df[
        feature_df["Overall Sentiment"] == "Positive"
    ]["Feature"].tolist()

    positive_text = ", ".join(positive_features)
    negative_text = ", ".join(negative_features)

    insights = f"""
Customers are highly satisfied with features such as {positive_text}.

However, improvements are recommended in areas like {negative_text}.

Recommended Actions:
• Improve negatively rated features
• Focus marketing on strong features
• Monitor customer feedback regularly
"""

    st.success(insights)

    st.divider()

    st.subheader("📋 Complete Review Analysis")

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Report",
        csv,
        "review_analysis.csv",
        "text/csv"
    )

st.sidebar.divider()

st.sidebar.subheader("⚙ Requirements")

st.sidebar.code("""
pip install streamlit
pip install pandas
pip install plotly
pip install transformers
pip install torch
""")

st.sidebar.subheader("▶ Run Application")

st.sidebar.code("""
streamlit run product.py
""")
