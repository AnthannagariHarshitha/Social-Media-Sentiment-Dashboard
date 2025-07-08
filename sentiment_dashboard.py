import streamlit as st
import pandas as pd
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.set_page_config(page_title="Social Media Sentiment Dashboard", layout="wide")

# Title
st.title("🌍 Social Media Sentiment Dashboard - Healthcare Focus")
st.markdown("Upload a dataset to simulate and analyze sentiment trends.")

# Step 1: Upload CSV File
uploaded_file = st.file_uploader("📤 Upload your CSV file (e.g., Countries.CSV)", type=['csv'])

if uploaded_file:
    # Load dataset
    try:
        countries_df = pd.read_csv(uploaded_file)
        required_columns = {'country', 'country_code'}
        if not required_columns.issubset(set(countries_df.columns)):
            st.error(f"❌ Uploaded file must contain columns: {required_columns}")
        else:
            countries_df.dropna(inplace=True)

            # Step 2: Simulate healthcare-related text
            healthcare_statements = [
                "The new vaccine rollout has been amazing!",
                "Poor healthcare service in recent months.",
                "Hospital staff have shown great resilience.",
                "Government healthcare policy needs revision.",
                "Mental health support is getting better.",
                "Still facing issues getting medical appointments.",
                "Excellent care at the city clinic.",
                "Vaccination drives are very slow lately.",
                "More awareness needed for health checkups.",
                "Public hospitals are doing a great job!",
                "Medical expenses are still very high.",
                "Ambulance response time has improved a lot.",
                "Health insurance policy needs to be simplified.",
                "Doctors are doing their best with limited resources.",
                "Vaccination centers are overcrowded.",
                "Thankful for free healthcare services."
            ]

            # Assign random text and date
            random.seed(42)
            n = len(countries_df)
            countries_df['text'] = [random.choice(healthcare_statements) for _ in range(n)]
            countries_df['date'] = pd.date_range(start='2023-01-01', periods=n, freq='D')
            countries_df['text'] = countries_df['text'].astype(str).str.lower()

            # Step 3: Sentiment Analysis using VADER
            analyzer = SentimentIntensityAnalyzer()

            def get_sentiment(text):
                score = analyzer.polarity_scores(text)['compound']
                if score >= 0.05:
                    return 'Positive'
                elif score <= -0.05:
                    return 'Negative'
                else:
                    return 'Neutral'

            countries_df['sentiment'] = countries_df['text'].apply(get_sentiment)

            # Sidebar Filters
            with st.sidebar:
                st.header("🔍 Filter Options")
                sentiment_filter = st.multiselect("Select Sentiment(s)", ["Positive", "Negative", "Neutral"],
                                                  default=["Positive", "Negative", "Neutral"])
                country_filter = st.multiselect("Select Country(s)",
                                                countries_df['country'].unique(),
                                                default=countries_df['country'].unique())

            # Apply filters
            filtered_df = countries_df[
                (countries_df['sentiment'].isin(sentiment_filter)) &
                (countries_df['country'].isin(country_filter))
            ]

            # Bar Chart – Sentiment by Country
            st.subheader("📊 Sentiment Distribution by Country")
            fig_bar = px.bar(filtered_df, x='country', color='sentiment', title="Sentiment by Country",
                             labels={'country': 'Country'}, height=600)
            fig_bar.update_layout(xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig_bar, use_container_width=True)

            # Line Chart – Sentiment Trend Over Time
            st.subheader("📈 Sentiment Trend Over Time")
            trend = filtered_df.groupby(['date', 'sentiment']).size().reset_index(name='count')
            fig_line = px.line(trend, x='date', y='count', color='sentiment', title="Sentiment Trend Over Time")
            st.plotly_chart(fig_line, use_container_width=True)

            # Show DataFrame
            st.subheader("🧾 Full Dataset Preview")
            st.dataframe(filtered_df[['country', 'country_code', 'text', 'sentiment', 'date']])

            # Export
            st.download_button("📥 Download Results as CSV", filtered_df.to_csv(index=False), file_name="Sentiment_Results.csv")
    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
