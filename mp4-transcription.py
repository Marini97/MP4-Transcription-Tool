# whisper
import whisper
import textwrap
import os
import glob
import re
import subprocess
import sys
from shutil import which

# create temp directory
if not os.path.exists("temp"):
    os.makedirs("temp")

# create output directory
if not os.path.exists("output"):
    os.makedirs("output")

output_directory = "output"
temp_directory = "temp"

# find current directory and specify the removal of temp files
main_path = os.getcwd()
wav_files = glob.glob(os.path.join(main_path, temp_directory, "*.wav"))

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible"""
    if which('ffmpeg') is None:
        print("Error: ffmpeg is not installed or not found in system PATH")
        print("\nTo install ffmpeg:")
        if sys.platform == 'win32':
            print("1. Download ffmpeg from https://www.gyan.dev/ffmpeg/builds/")
            print("2. Extract the archive")
            print("3. Add the bin folder to your system PATH")
            print("\nOr install using chocolatey:")
            print("choco install ffmpeg")
        elif sys.platform == 'darwin':
            print("Install using homebrew:")
            print("brew install ffmpeg")
        else:
            print("Install using your package manager:")
            print("sudo apt install ffmpeg  # for Ubuntu/Debian")
            print("sudo yum install ffmpeg  # for CentOS/RHEL")
        return False
    return True

def normalize_path(path):
    """Normalize file path for Windows"""
    # Remove quotes and clean path
    path = path.strip().strip('"').strip("'")
    
    # Convert to absolute path
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    # Normalize path separators
    path = os.path.normpath(path)
    
    return path

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def extract_audio_from_mp4(input_file):
    """Extract audio from MP4 file and save as WAV"""
    # Normalize input path
    input_file = normalize_path(input_file)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is required but not found")
    
    output_file = os.path.join(temp_directory, "temp.wav")
    
    try:
        # Use ffmpeg to extract audio
        command = [
            'ffmpeg',
            '-i', input_file,  # Input file
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Number of audio channels
            '-y',  # Overwrite output file if it exists
            output_file
        ]
        
        subprocess.run(command, check=True, stderr=subprocess.PIPE)
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")

def transcribe_audio(file_path_mp3, video_name):
    # Load the Whisper model with better language detection
    model = whisper.load_model("medium")
    
    # Transcribe with language detection and translation
    result = model.transcribe(
        file_path_mp3, 
        language="it",  # Specify Italian as primary language
        task="transcribe",  # or "translate" if you want English output
        fp16=False  # Disable FP16 for compatibility
    )
    del model

    # Process the transcription with timestamps
    transcription_with_timestamps = []
    for segment in result['segments']:
        start_time = format_timestamp(segment['start'])
        text = segment['text'].strip()
        
        # Basic language and formatting improvements
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Capitalize first letter of each segment
        text = text.capitalize()
        
        transcription_with_timestamps.append(f"\n\n[{start_time}] {text}")

    # Combine into paragraphs
    paragraphs = []
    paragraph = []
    for line in transcription_with_timestamps:
        paragraph.append(line)
        if len(paragraph) == 4:
            paragraphs.append("\n\n".join(paragraph))
            paragraph = []

    if paragraph:
        paragraphs.append("".join(paragraph))

    # Formatting
    formatted = "".join(paragraphs)

    # Word-wrapping
    wrapped = " ".join(
        [" \n" + textwrap.fill(p, width=100) for p in formatted.split("\n\n")]
    )

    # Add an extra newline between paragraphs
    wrapped_with_extra_newline = "".join([wrapped, ""])

    # Sanitizing filename
    video_name = re.sub(r"[\W_]+", " ", video_name)

    # Establishing output directory and file name
    output_dir = r"output"
    file_name = f"{video_name}_transcription.txt"
    file_path = os.path.join(output_dir, file_name)

    # Writing output to file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(wrapped_with_extra_newline)
    
    return transcription_with_timestamps

if __name__ == "__main__":
    print("MP4 to Text Transcription Tool")
    print("-" * 50)
    print("\nTips for entering file path:")
    print("1. You can drag and drop the file into this window")
    print("2. For relative paths, make sure the file is in the same folder as the script")
    print("3. For full paths, you can copy the path from File Explorer")
    print("\nExample paths:")
    print("Relative: video.mp4")
    print(r"Full: C:\Users\YourName\Videos\video.mp4")
    print("-" * 50)
    
    # Get input MP4 file path
    mp4_path = input("\nEnter the path to your MP4 file: ")
    
    try:
        # Extract audio from MP4
        wav_path = extract_audio_from_mp4(mp4_path)
        
        # Get base filename without extension for output
        video_name = os.path.splitext(os.path.basename(mp4_path))[0]
        
        # Transcribe the audio
        print("\nExtracting audio completed. Starting transcription...")
        transcription = transcribe_audio(wav_path, video_name)
        
        print(f"\nTranscription completed! Check the 'output' folder for {video_name}_transcription.txt")
        
        # Optional: Print a preview of the transcription
        print("\nTranscription Preview:")
        for line in transcription[:10]:  # Show first 10 lines
            print(line)
        
    except FileNotFoundError as e:
        print(f"\nError: File not found")
        print("Please check that:")
        print("1. The file path is correct")
        print("2. The file exists")
        print("3. You have permission to access the file")
        print(f"\nPath tried: {mp4_path}")
    except RuntimeError as e:
        print(f"\nError: {str(e)}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("Please make sure the file is a valid MP4 video file.")
    
    finally:
        # Cleanup
        if wav_files:
            for wav_file in wav_files:
                try:
                    os.remove(wav_file)
                except:
                    pass

    input("\nPress Enter to exit...")