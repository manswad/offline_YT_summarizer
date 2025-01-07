import os

# Set environment variable to avoid KMP duplicate library error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore")

import yt_dlp
import ffmpeg
import re
import gc
import math
import whisper
from pydub import AudioSegment

# Set environment variable to avoid KMP duplicate library error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Function to validate YouTube URL
def is_valid_youtube_url(url):
    youtube_regex = r"^(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|shorts/)[A-Za-z0-9_-]{11}$"
    return re.match(youtube_regex, url) is not None

# Function to download the video and extract audio
def download_youtube_video_as_mp3(url):
    if not is_valid_youtube_url(url):
        print("The URL provided is not a valid YouTube video URL.")
        return

    try:
        # Download video using yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',  # Best quality audio
            'outtmpl': 'downloaded_video.%(ext)s',  # Save the file with a default name
        }

        # Using yt-dlp to download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            video_filename = ydl.prepare_filename(result)

            # Check if the downloaded file is in audio format, if not, proceed with conversion
            if video_filename.endswith(".webm") or video_filename.endswith(".m4a"):
                mp3_filename = video_filename.rsplit('.', 1)[0] + '.mp3'

                # Use ffmpeg to convert the downloaded video to MP3
                ffmpeg.input(video_filename).output(mp3_filename).run()

                # Remove the original downloaded file (optional)
                os.remove(video_filename)

                print(f"Downloaded and converted to MP3: {mp3_filename}")
                return mp3_filename
            else:
                print(f"Audio already in MP3 format: {video_filename}")
                return video_filename
    except Exception as e:
        print(f"Error: {e}")

# Load the Whisper model
def load_whisper_model():
    # Use 'base' or 'small' for VRAM/RAM constraints
    return whisper.load_model("medium")  # Use 'medium' or 'large' for better accuracy

# MP3 to WAV conversion
def mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = mp3_file.replace(".mp3", ".wav")
    audio.export(wav_file, format="wav")
    return wav_file

# Split the audio into smaller chunks for better performance
def split_audio(audio, chunk_length_ms=300000):  # 5-minute chunks (300000 ms)
    chunks = []
    num_chunks = math.ceil(len(audio) / chunk_length_ms)
    for i in range(num_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, len(audio))
        chunk = audio[start_ms:end_ms]
        chunks.append(chunk)
    return chunks

# Transcribe using Whisper
def transcribe_audio(model, audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

# Transcribe function
def transcribe(mp3_file):
    wav_file = mp3_to_wav(mp3_file)
    audio = AudioSegment.from_wav(wav_file)
    
    # Split the audio into smaller chunks if needed
    chunks = split_audio(audio)
    
    model = load_whisper_model()
    full_transcription = ""
    
    for i, chunk in enumerate(chunks):
        chunk_file = wav_file.replace(".wav", f"_chunk_{i}.wav")
        chunk.export(chunk_file, format="wav")
        transcription = transcribe_audio(model, chunk_file)
        full_transcription += transcription + "\n"  # Combine transcriptions from each chunk
        
    with open('text.txt', 'w') as f:
        f.write(full_transcription)

def remove_audio_files():
    for file in os.listdir():
        if file.endswith(".wav") or file.endswith(".mp3"):
            os.remove(file)

url = input("Enter YouTube video URL: ")
download_youtube_video_as_mp3(url)
mp3_file_path = f"downloaded_video.mp3"
transcribe(mp3_file_path)
remove_audio_files()
gc.collect()