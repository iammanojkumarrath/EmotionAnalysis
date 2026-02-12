import assemblyai as aai

aai.settings.api_key = "6d40fb0ac0c84bbc84a2a41e839ff394"

def get_sentiment_data(audio_url):
    # Use "universal-3-pro" as required by the error message
    config = aai.TranscriptionConfig(
        sentiment_analysis=True,
        speech_models=["universal-3-pro"] 
    )
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_url, config)
    
    # The rest of your code remains the same
    sentiment_results = []
    if transcript.sentiment_analysis:
        for sentiment_result in transcript.sentiment_analysis:
            sentiment_results.append({
                "start": sentiment_result.start / 1000, 
                "text": sentiment_result.text,
                "sentiment": sentiment_result.sentiment, 
                "confidence": sentiment_result.confidence
            })
    return sentiment_results