from dotenv import load_dotenv
import os

load_dotenv()

HOST = os.getenv("HOST", "")
PORT = int(os.getenv("PORT", ""))

# Whisper parameters
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "")
LANGUAGE = os.getenv("LANGUAGE", "")
WHISPER_COMPRESS_RATIO_THRESHOLD = float(
    os.getenv("WHISPER_COMPRESS_RATIO_THRESHOLD", ""))
WHISPER_NO_SPEECH_THRESHOLD = float(
    os.getenv("WHISPER_NO_SPEECH_THRESHOLD", ""))
PIPELINE_MAX_SPEAKERS = int(os.getenv("PIPELINE_MAX_SPEAKERS", ""))
PIPELINE_DURATION = float(os.getenv("PIPELINE_DURATION", ""))
PIPELINE_STEP = float(os.getenv("PIPELINE_STEP", ""))
PIPELINE_SAMPLE_RATE = int(os.getenv("PIPELINE_SAMPLE_RATE", ""))
PIPELINE_TAU = float(os.getenv("PIPELINE_TAU", ""))
PIPELINE_RHO = float(os.getenv("PIPELINE_RHO", ""))
PIPELINE_DELTA = float(os.getenv("PIPELINE_DELTA", ""))
PIPELINE_CHUNK_DURATION = float(os.getenv("PIPELINE_CHUNK_DURATION", ""))
POWER_GPU_OR_CPU = os.getenv("POWER_GPU_OR_CPU", "")
LATENCY = os.getenv("LATENCY", "")
BEAM_SIZE = int(os.getenv("BEAM_SIZE", ""))
ASR_DURATION = int(os.getenv("ASR_DURATION", ""))

# Output audio parameters
AUDIO_FOLDER_PATH = os.getenv("AUDIO_FOLDER_PATH", "")
