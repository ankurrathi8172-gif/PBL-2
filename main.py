import cv2
import time
from deepface import DeepFace
from logger import log_emotion

emoji_map = {
    "happy": ":)",
    "sad": ":(",
    "angry": "!!",
    "surprise": ":O",
    "fear": "!!",
    "disgust": "X",
    "neutral": ":|"
}

song_map = {
    "happy": ["Love You Zindagi", "Uptown Funk", "Can't Stop The Feeling"],
    "sad": ["Channa Mereya", "Let Her Go", "Someone Like You"],
    "angry": ["Believer", "Till I Collapse", "Stronger"],
    "neutral": ["Perfect", "Memories", "Counting Stars"],
    "surprise": ["On Top of the World", "Thunder", "Shape of You"],
    "fear": ["Fix You", "Demons", "Lovely"],
    "disgust": ["Lose Yourself", "Numb", "In The End"]
}

alert_emotions = ["sad", "angry", "fear"]
alert_threshold_seconds = 5

cap = cv2.VideoCapture(0)

start_alert_time = None
last_emotion = None

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

        emotion_data = result[0]["emotion"]
        emotion = result[0]["dominant_emotion"]
        confidence = emotion_data[emotion]

        emoji = emoji_map.get(emotion, "")
        text = f"Emotion: {emotion.upper()} {emoji} ({confidence:.2f}%)"

        cv2.putText(frame, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        log_emotion(emotion, round(confidence, 2))

        if emotion in alert_emotions:
            if last_emotion == emotion:
                if start_alert_time is None:
                    start_alert_time = time.time()
                else:
                    elapsed = time.time() - start_alert_time
                    if elapsed >= alert_threshold_seconds:
                        cv2.putText(frame, "ALERT: STRESS DETECTED! TAKE A BREAK!",
                                    (20, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            else:
                start_alert_time = time.time()
        else:
            start_alert_time = None

        last_emotion = emotion

        songs = song_map.get(emotion, [])
        y_pos = 160

        cv2.putText(frame, "Recommended Songs:", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        for s in songs[:3]:
            cv2.putText(frame, f"- {s}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_pos += 30

    except:
        cv2.putText(frame, "Face not detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Facial Emotion Recognition (Alert + Songs)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
