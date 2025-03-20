# 🎬 MP4 to Text Transcription Tool

## 📝 Description

This Python script provides a transcription tool that converts MP4 video files to text transcriptions, with special support for Italian and multilingual content. The tool uses OpenAI's Whisper for high-quality speech recognition and supports various audio extraction and transcription features.

## ✨ Features

- 🔊 Transcribe MP4 files to text
- 🇮🇹 Specialized support for Italian and mixed-language content
- ⏰ Timestamps for each transcription segment
- 📂 Automatic output file generation
- 🧹 Temporary file cleanup

## 🛠 Prerequisites

### Software Requirements
- Python 3.10+
- FFmpeg

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

## 🖥 Usage

### Basic Usage
```bash
python mp4-transcription.py
```

1. Run the script
2. When prompted, enter the full path to your MP4 file
3. Wait for transcription
4. Find the transcription in the `output` folder

### Example Input Paths
- Relative: `video.mp4`
- Full Path: `C:\Users\YourName\Videos\video.mp4`

## 🔍 Transcription Details
- Supports mixed Italian and English content
- Includes timestamps
- Capitalizes first letter of segments
- Removes excessive whitespace

## 📦 Output
- Generates a text file in the `output` directory
- Filename format: `[original_filename]_transcription.txt`
- Includes timestamps and cleaned text

## 🛠 Customization

### Changing Transcription Language
Modify the `transcribe_audio()` function:
- `language="it"`: Italian
- `task="transcribe"`: Keep original language
- `task="translate"`: Convert to English

## 📝 Notes
- Larger Whisper models provide better accuracy
- Mixed-language content may require manual verification

## 🐛 Troubleshooting
- Ensure FFmpeg is correctly installed and in PATH
- Check file permissions
- Verify MP4 file integrity

## 🙏 Acknowledgments
- [OpenAI Whisper](https://github.com/openai/whisper)
- [FFmpeg](https://ffmpeg.org/)

---

**Happy Transcribing!** 🎧📝