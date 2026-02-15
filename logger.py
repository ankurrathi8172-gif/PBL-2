import pandas as pd
from datetime import datetime
import os

LOG_FILE = "emotion_log.csv"

def log_emotion(emotion, confidence):
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["Time", "Emotion", "Confidence"])
        df.to_csv(LOG_FILE, index=False)

    df = pd.read_csv(LOG_FILE)

    new_row = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Emotion": emotion,
        "Confidence": confidence
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)
