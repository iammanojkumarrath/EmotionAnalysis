import streamlit as st
import assemblyai as aai
import pandas as pd
import plotly.express as px

# 1. Configuration
st.set_page_config(page_title="Voice Sentiment Dashboard", layout="wide")
aai.settings.api_key = "6d40fb0ac0c84bbc84a2a41e839ff394" # <--- Paste your key here

def get_sentiment_data(audio_file):
    """Transcribes audio and extracts sentiment with timestamps."""
    config = aai.TranscriptionConfig(
        sentiment_analysis=True,
        speech_models=["universal-3-pro"]
    )
    transcriber = aai.Transcriber()
    
    # Upload the file to AssemblyAI
    transcript = transcriber.transcribe(audio_file, config)

    if transcript.status == aai.TranscriptStatus.error:
        st.error(f"Transcription Error: {transcript.error}")
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

# 2. Sidebar / Header
st.title("🎙️ Voice Sentiment Dashboard")
st.markdown("Analyze the emotional journey of your audio files with second-by-second precision.")

# 3. File Upload
uploaded_file = st.file_uploader("Upload Audio (MP3 or WAV)", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file)
    
    if st.button("Analyze Sentiment"):
        with st.spinner('AI is analyzing the voice emotions...'):
            data = get_sentiment_data(uploaded_file)
            
            if data:
                df = pd.DataFrame(data)

                # --- DASHBOARD LAYOUT ---
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("Emotion Timeline")
                    # Mapping colors for better visuals
                    color_map = {"POSITIVE": "#00CC96", "NEUTRAL": "#636EFA", "NEGATIVE": "#EF553B"}
                    
                    fig = px.scatter(
                        df, 
                        x="Time (Sec)", 
                        y="Sentiment", 
                        color="Sentiment",
                        color_discrete_map=color_map,
                        hover_data=["Text", "Confidence"],
                        title="Sentiment Captured at Specific Moments"
                    )
                    fig.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')))
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Summary")
                    sentiment_counts = df["Sentiment"].value_counts()
                    fig_pie = px.pie(
                        values=sentiment_counts.values, 
                        names=sentiment_counts.index,
                        color=sentiment_counts.index,
                        color_discrete_map=color_map,
                        hole=0.4
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # --- DETAILED TABLE ---
                st.divider()
                st.subheader("Detailed Sentiment Log")
                st.dataframe(df, use_container_width=True)
                
                # Download Option
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Report as CSV", csv, "sentiment_report.csv", "text/csv")
            
            else:
                st.warning("The analysis is complete, but no specific sentiments were detected. Try a longer recording with more expressive speech.")

else:
    st.info("Please upload an audio file to begin the analysis.")