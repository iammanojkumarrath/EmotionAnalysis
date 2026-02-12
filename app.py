import streamlit as st
import assemblyai as aai
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURATION & SECURITY ---
st.set_page_config(page_title="Voice Sentiment Dashboard", layout="wide")

# When deploying to GitHub/Streamlit, use Secrets. 
# For local testing, you can replace st.secrets["AAI_API_KEY"] with "your_actual_key"
if "AAI_API_KEY" in st.secrets:
    aai.settings.api_key = st.secrets["AAI_API_KEY"]
else:
    # Fallback for local testing if you haven't set up secrets.toml
    # aai.settings.api_key = "PASTE_YOUR_KEY_HERE_FOR_LOCAL_ONLY"
    st.warning("API Key not found in Streamlit Secrets. Please configure it to run.")

def get_sentiment_data(audio_file):
    """Handles the transcription and sentiment analysis logic."""
    try:
        config = aai.TranscriptionConfig(
            sentiment_analysis=True,
            speech_models=["universal-3-pro"]
        )
        transcriber = aai.Transcriber()
        
        # Upload and process
        transcript = transcriber.transcribe(audio_file, config)

        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"AI Error: {transcript.error}")
            return None

        sentiment_results = []
        if transcript.sentiment_analysis:
            for result in transcript.sentiment_analysis:
                sentiment_results.append({
                    "Time (Sec)": round(result.start / 1000, 2),
                    "Text": result.text,
                    "Sentiment": result.sentiment,
                    "Confidence": round(result.confidence, 2)
                })
        return sentiment_results
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# --- 2. USER INTERFACE ---
st.title("🎙️ Voice Sentiment Dashboard")
st.markdown("Developed for real-time emotional analysis of speech audio.")

# Sidebar info
with st.sidebar:
    st.header("How it works")
    st.write("1. Upload a clear audio file.")
    st.write("2. The AI pinpoints emotions by the second.")
    st.write("3. View the distribution and timeline charts.")

uploaded_file = st.file_uploader("Upload Audio (MP3 or WAV)", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file)
    
    if st.button("Generate Dashboard"):
        with st.spinner('Analyzing speech patterns...'):
            data = get_sentiment_data(uploaded_file)
            
            if data:
                df = pd.DataFrame(data)

                # --- 3. VISUALIZATIONS ---
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("Emotion Timeline")
                    # Professional Color Palette
                    color_map = {"POSITIVE": "#00CC96", "NEUTRAL": "#636EFA", "NEGATIVE": "#EF553B"}
                    
                    fig = px.scatter(
                        df, 
                        x="Time (Sec)", 
                        y="Sentiment", 
                        color="Sentiment",
                        color_discrete_map=color_map,
                        hover_data=["Text"],
                        size_max=12
                    )
                    # Line to connect the flow
                    fig.update_traces(mode='lines+markers', marker=dict(size=10))
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Overall Mood")
                    counts = df["Sentiment"].value_counts()
                    fig_pie = px.pie(
                        values=counts.values, 
                        names=counts.index,
                        color=counts.index,
                        color_discrete_map=color_map,
                        hole=0.5
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # --- 4. DATA LOG ---
                st.divider()
                st.subheader("Timestamped Emotion Log")
                st.dataframe(df, use_container_width=True)
                
                # Download Feature
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Export Report (CSV)", csv, "sentiment_analysis.csv", "text/csv")
            
            else:
                st.warning("Analysis complete, but no specific emotional markers were detected. Try a longer audio clip.")

else:
    st.info("Awaiting audio upload...")