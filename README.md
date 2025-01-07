# YouTube Video Transcriber and Summarizer

This project provides a Python-based solution for transcribing YouTube videos and generating concise summaries. It consists of two main components: a video transcription script and a text summarization script that work together to process YouTube content.

## Features

- YouTube video download and audio extraction
- Audio transcription using OpenAI's Whisper model
- Text summarization using LLaMA-based model
- Support for processing long videos through chunking
- 4-bit quantization for efficient model loading
- Automatic cleanup of temporary files

## Prerequisites

Before running this project, make sure you have the following dependencies installed:

Prefer installing torch from [here](https://pytorch.org/get-started/locally/).

```bash
pip install yt-dlp ffmpeg-python whisper pydub transformers
```

You'll also need FFmpeg installed on your system:
- **Linux**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from the official FFmpeg website

## Project Structure

```
.
├── transcribe.py    # Video downloading and transcription script
├── summarize.py     # Text summarization script
├── text.txt         # Generated transcription output
└── summary.txt      # Generated summary output
```

## Usage

### 1. Transcribing a YouTube Video

Run the transcription script:

```bash
python transcribe.py
```

When prompted, enter the YouTube video URL. The script will:
1. Download the video and extract audio
2. Convert the audio to WAV format
3. Transcribe the audio using Whisper
4. Save the transcription to `text.txt`
5. Clean up temporary audio files

### 2. Generating a Summary

After transcription is complete, run the summarization script:

```bash
python summarize.py
```

The script will:
1. Read the transcription from `text.txt`
2. Generate a summary using the LLaMA model
3. Save the summary to `summary.txt`

## Technical Details

### Transcription Component

- Uses `yt-dlp` for reliable YouTube video downloading
- Implements URL validation using regex
- Splits long audio files into 5-minute chunks for better performance
- Utilizes Whisper's "medium" model for transcription
- Includes automatic memory management and file cleanup

### Summarization Component

- Uses a fine-tuned LLaMA model optimized for summarization
- Implements 4-bit quantization for reduced memory usage
- Supports both CPU and CUDA execution
- Handles long texts through intelligent chunking
- Configurable parameters for summary length and quality

## Configuration Options

### Transcription Settings

```python
# Change Whisper model size
model = whisper.load_model("medium")  # Options: base, small, medium, large

# Adjust chunk size for audio processing (in milliseconds)
chunk_length_ms = 300000  # Default: 5 minutes
```

### Summarization Settings

```python
# Model configuration
model_name = "prithivMLmods/Llama-Chat-Summary-3.2-3B"
max_input_length = 1024  # Token limit per chunk
max_new_tokens = 150    # Summary length

# Generation parameters
temperature = 0.7       # Creativity vs. consistency
top_p = 0.9            # Nucleus sampling threshold
```

## Error Handling

The project includes robust error handling for common issues:
- Invalid YouTube URLs
- Download failures
- Audio conversion errors
- Memory constraints
- File system errors

## Memory Management

Both scripts implement memory optimization techniques:
- Garbage collection after processing
- File cleanup routines
- Efficient chunk processing
- 4-bit model quantization

## Limitations

- Requires sufficient disk space for temporary audio files
- Processing time depends on video length and hardware capabilities
- GPU recommended for faster processing
- Internet connection required for video download and model loading

## License

This project is available under the MIT License. Feel free to modify and distribute as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.