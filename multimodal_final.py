import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import torch
import whisper
from tkinter import filedialog, Tk
from moviepy.editor import VideoFileClip
from deepface import DeepFace
from transformers import pipeline
from typing import Dict, Tuple, Optional, List

# Constants
MAX_FRAMES = 5
MIN_CONFIDENCE = 0.5

# --- File Selector ---
def select_video_file() -> Optional[str]:
    try:
        root = Tk()
        root.withdraw()
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        return path if path and os.path.exists(path) else None
    except:
        return None

# --- AUDIO MODULE ---
def extract_audio(video_path: str, output_audio: str = "audio.wav") -> Optional[str]:
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(output_audio, codec='pcm_s16le')
        return output_audio if os.path.exists(output_audio) else None
    except:
        return None

def whisper_transcription(audio_path: str) -> str:
    try:
        model = whisper.load_model("base")  # You can use "tiny", "small", "medium", or "large"
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"[✘] Whisper Error: {e}")
        return "[TRANSCRIPTION FAILED]"

def text_sentiment(text: str) -> Dict[str, float]:
    try:
        if text.startswith("["):
            return {"label": "neutral", "score": MIN_CONFIDENCE}
        sentiment = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
        result = sentiment(text)[0]
        return {"label": result["label"], "score": result["score"]}
    except:
        return {"label": "neutral", "score": MIN_CONFIDENCE}

def extract_pitch(audio_path: str) -> float:
    try:
        y, sr_lib = librosa.load(audio_path)
        pitches, _ = librosa.piptrack(y=y, sr=sr_lib)
        valid_pitches = pitches[pitches > 0]
        return np.mean(valid_pitches) if len(valid_pitches) else 150.0
    except:
        return 150.0

def classify_audio_emotion(text: str, pitch: float) -> Tuple[str, float]:
    try:
        emotion_data = text_sentiment(text)
        label, score = emotion_data["label"].lower(), emotion_data["score"]
        if label == "joy" and pitch > 250: return "Happy", max(score, 0.85)
        if label == "sadness" and pitch < 180: return "Sad", max(score, 0.82)
        if label == "anger" and pitch > 300: return "Angry", max(score, 0.89)
        if label == "fear" and pitch > 260: return "Fear", max(score, 0.80)
        if label == "disgust" and pitch < 150: return "Disgust", max(score, 0.78)
        if label == "surprise" and 220 < pitch < 270: return "Surprise", max(score, 0.83)
        return "Neutral", max(score, MIN_CONFIDENCE)
    except:
        return "Neutral", MIN_CONFIDENCE

# --- VIDEO MODULE ---
def extract_frames(video_path: str, frame_dir: str = "frames", num_frames: int = MAX_FRAMES) -> Optional[List[str]]:
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    paths = []

    for i in range(num_frames):
        frame_index = i * (total_frames // num_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(frame_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            paths.append(frame_path)
    cap.release()
    return paths if paths else None

def detect_faces(frame_paths: List[str]) -> Tuple[str, float]:
    emotions, confidences = [], []
    for frame in frame_paths:
        try:
            result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False, silent=True)
            emotion = result[0]['dominant_emotion'].capitalize()
            confidence = result[0]['emotion'][emotion.lower()] / 100
            emotions.append(emotion)
            confidences.append(confidence)
        except:
            emotions.append("Neutral")
            confidences.append(MIN_CONFIDENCE)

    if not emotions:
        return "Neutral", MIN_CONFIDENCE

    dominant = max(set(emotions), key=emotions.count)
    avg_conf = np.mean([c for e, c in zip(emotions, confidences) if e == dominant])
    return dominant, avg_conf

# --- FUSION ---
def multimodal_fusion(video_path: str) -> Dict:
    try:
        audio_path = extract_audio(video_path)
        if not audio_path:
            raise Exception("Audio extraction failed.")
        
        text = whisper_transcription(audio_path)
        pitch = extract_pitch(audio_path)
        audio_emotion, audio_conf = classify_audio_emotion(text, pitch)

        frames = extract_frames(video_path)
        if not frames:
            raise Exception("Frame extraction failed.")

        video_emotion, video_conf = detect_faces(frames)

        dominant_emotion = audio_emotion if audio_conf > video_conf else video_emotion
        dominant_conf = max(audio_conf, video_conf)

        return {
            "status": "success",
            "text": text,
            "audio_emotion": audio_emotion,
            "audio_confidence": audio_conf,
            "video_emotion": video_emotion,
            "video_confidence": video_conf,
            "dominant_emotion": dominant_emotion,
            "dominant_confidence": dominant_conf
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# --- PLOT ---
def plot_emotion_analysis(results: Dict):
    if results.get("status") != "success":
        return

    plt.figure(figsize=(10, 4))
    plt.suptitle("Multimodal Emotion Confidence", fontsize=14)

    plt.subplot(1, 2, 1)
    plt.bar(["Audio", "Video"], [results["audio_confidence"], results["video_confidence"]], color=["blue", "orange"])
    plt.ylim(0, 1)
    plt.title("Confidence Scores")
    for i, val in enumerate([results["audio_confidence"], results["video_confidence"]]):
        plt.text(i, val + 0.01, f"{val:.2f}", ha='center')

    plt.subplot(1, 2, 2)
    sns.histplot([results["audio_confidence"], results["video_confidence"]], bins=5, kde=True, color='green')
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.tight_layout()
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    print("\n🎥 Select a Video File")
    path = select_video_file()
    if not path:
        print("[✘] No file selected or file does not exist.")
        exit()

    result = multimodal_fusion(path)

    if result["status"] == "success":
        print("\n📝 Emotion Report")
        print(f"🎧 Audio Emotion: {result['audio_emotion']} ({result['audio_confidence']:.2f})")
        print(f"🎥 Video Emotion: {result['video_emotion']} ({result['video_confidence']:.2f})")
        print(f"🏆 Dominant Emotion: {result['dominant_emotion']} ({result['dominant_confidence']:.2f})")
        print(f"🗣 Transcription:\n{result['text']}")
        plot_emotion_analysis(result)
    else:
        print(f"[✘] Error: {result['error']}")
