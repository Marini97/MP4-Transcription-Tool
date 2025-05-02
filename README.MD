# 🎬 MP4 to Text Transcription Tool
## 📝 Description
This Python script provides a powerful transcription tool that converts MP4 video files to text transcriptions. The tool uses OpenAI's Whisper for high-quality speech recognition and PyAnnote for speaker diarization, allowing you to identify different speakers in your videos. Specialized for Italian content but supporting multiple languages.

## ✨ Features
- 🔊 Transcribe MP4 files to text with high accuracy
- 👥 Speaker diarization to identify different speakers in your video
- ⏰ Timestamps for each transcription segment
- 🧠 GPU acceleration for faster processing (if available)
- 🔧 Command-line interface for batch processing
- 📂 Automatic output file organization
- 🧹 Temporary file cleanup

## 🛠 Prerequisites
### Software Requirements
- Python 3.10+
- FFmpeg
- CUDA (optional, for GPU acceleration)
- HuggingFace API token (required for speaker diarization)

## 🚀 Installation
### 1. Clone the Repository
```bash
git clone https://github.com/Marini97/MP4-Transcription-Tool.git
cd MP4-Transcription-Tool
```

### 2. Install FFmpeg
#### Windows
- Download from [FFmpeg Official Site](https://www.gyan.dev/ffmpeg/builds/)
- Add to system PATH

#### macOS
```bash
brew install ffmpeg
```

#### Linux
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### 3. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set up HuggingFace Token
Create a `.env` file in the project directory and add your token:
```
HUGGINGFACE_TOKEN=your_huggingface_token_here
```
You can get your token by:
1. Creating an account at HuggingFace
2. Going to your profile → Settings → Access Tokens
3. Creating a new token with at least read access

## 🖥 Usage
### Basic Usage
```bash
python mp4-transcription.py
```

1. Run the script
2. When prompted, enter the full path to your MP4 file
3. Wait for transcription
4. Find the transcription in the `output` folder

### Command-line Usage
```bash
python mp4-transcription.py path/to/video.mp4 --speakers 3 --language en
```

### Available Command-line Options
```
--output-dir DIR    Directory for output files (default: output)
--model MODEL       Whisper model to use (default: turbo)
--language          LANG Primary language in the video (default: it)
--speakers NUM      Number of speakers for diarization (default: 2)
--cpu               Force CPU usage even if GPU is available
--keep-temp         Keep temporary files after processing
```

### Example Input Paths
- Relative: `video.mp4`
- Full Path: `C:\Users\YourName\Videos\video.mp4`
- You can also drag and drop files into the terminal window

## 📦 Output
The tool generates two types of transcription files:
### 1. Basic Transcription
- Filename: `[video_name]_transcription.txt`
- Contains timestamps and text without speaker identification

### 2. Transcription with Speaker Diarization
- Filename: `[video_name]_transcription_with_speakers.txt`
- Includes timestamps, speaker identification, and text
- Format: `[00:01:23] Speaker 1: Text of what was said`

## 🔍 Advanced Configuration
### Whisper Models
You can choose from various Whisper models with the `--model` flag:
- `tiny`: Fastest, least accurate
- `base`: Fast with reasonable accuracy
- `small`: Good balance of speed and accuracy
- `medium`: High accuracy, slower processing
- `large`: Highest accuracy, slowest processing
- `turbo`: OpenAI's optimized model (default)

### Languages
Set the primary language with the `--language` flag:
- `it`: Italian (default)
- `en`: English
- `fr`: French
- `de`: German
- And many others supported by Whisper

## 🛠 Customization
The tool can be customized by modifying the config dictionary in the code:
```python
self.config = {
    'output_dir': 'output',
    'temp_dir': 'temp',
    'whisper_model': 'turbo',
    'language': 'it',
    'num_speakers': 2,
    'sample_rate': '44100',
    'channels': '2',
    'use_gpu': torch.cuda.is_available(),
    'cleanup_temp': True
}
```
## 🔧 Troubleshooting
### Speaker Diarization Not Working
- Check that your `.env` file contains a valid `HUGGINGFACE_TOKEN`
- Ensure you have internet access for API calls

### Poor Transcription Quality
- Try using a larger Whisper model with `--model medium` or `--model large`
- Ensure your audio has minimal background noise
- Try adjusting the number of speakers with `--speakers` option

### Error: "Input file not found"
- Check that the file path is correct
- If you're dragging and dropping files, the path may include quotes
- Use absolute paths if relative paths aren't working

### Slow Processing
- Enable GPU acceleration if available
- For large files, use smaller Whisper models like `--model small`

## 📝 Notes
- Speaker diarization works best for clear audio with distinct speakers
- For multilingual content, set --language to the primary language
- Using GPU acceleration can significantly improve processing speed

## 🙏 Acknowledgments
- [OpenAI Whisper](https://github.com/openai/whisper)
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio)
- [HuggingFace](https://huggingface.co/)
- [FFmpeg](https://ffmpeg.org/)
---

**Happy Transcribing!** 🎧📝