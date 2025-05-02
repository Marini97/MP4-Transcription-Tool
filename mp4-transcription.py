import whisper
from pyannote.audio import Pipeline
import torch
from tqdm import tqdm
import os
import glob
import re
import subprocess
import sys
import argparse
import shutil
from pathlib import Path
from shutil import which
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TranscriptionTool:
    def __init__(self, config=None):
        """Initialize the transcription tool with configuration options."""
        # Load environment variables
        load_dotenv()
        self.huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
        if not self.huggingface_token:
            logger.warning("HUGGINGFACE_TOKEN not found in .env file. Speaker diarization may fail.")
        
        # Set up default configuration
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
        
        # Update with user config if provided
        if config:
            self.config.update(config)
        
        # Create directories
        for directory in [self.config['output_dir'], self.config['temp_dir']]:
            os.makedirs(directory, exist_ok=True)
        
        # Check dependencies
        self._check_dependencies()
        
        # Track resources for cleanup
        self.resources_to_cleanup = []

    def _check_dependencies(self):
        """Check if all required dependencies are available."""
        # Check ffmpeg
        if which('ffmpeg') is None:
            self._show_ffmpeg_installation_guide()
            raise RuntimeError("ffmpeg is required but not found")
            
        # Check if CUDA is available when requested
        if self.config['use_gpu'] and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Falling back to CPU.")
            self.config['use_gpu'] = False

    def _show_ffmpeg_installation_guide(self):
        """Show instructions for installing ffmpeg based on the platform."""
        logger.error("ffmpeg is not installed or not found in system PATH")
        logger.info("\nTo install ffmpeg:")
        
        if sys.platform == 'win32':
            logger.info("1. Download ffmpeg from https://www.gyan.dev/ffmpeg/builds/")
            logger.info("2. Extract the archive")
            logger.info("3. Add the bin folder to your system PATH")
            logger.info("\nOr install using chocolatey: choco install ffmpeg")
        elif sys.platform == 'darwin':
            logger.info("Install using homebrew: brew install ffmpeg")
        else:
            logger.info("Install using your package manager:")
            logger.info("sudo apt install ffmpeg  # for Ubuntu/Debian")
            logger.info("sudo yum install ffmpeg  # for CentOS/RHEL")

    def extract_audio(self, input_file):
        """Extract audio from video file and save as WAV."""
        # Normalize and validate path
        input_path = Path(input_file).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Set up output path
        output_file = Path(self.config['temp_dir']) / "temp.wav"
        self.resources_to_cleanup.append(output_file)
        
        logger.info(f"Extracting audio from {input_path.name}...")
        
        try:
            # Use ffmpeg to extract audio
            command = [
                'ffmpeg',
                '-i', str(input_path),
                '-vn',  # Disable video
                '-acodec', 'pcm_s16le',
                '-ar', self.config['sample_rate'],
                '-ac', self.config['channels'],
                '-y',
                str(output_file)
            ]
            
            # Run with progress information
            process = subprocess.run(
                command, 
                check=True, 
                stderr=subprocess.PIPE, 
                stdout=subprocess.PIPE
            )
            
            return str(output_file)
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Failed to extract audio: {error_msg}")

    def format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def transcribe(self, audio_path, video_name):
        """Transcribe audio with Whisper and perform speaker diarization."""
        device = "cuda" if self.config['use_gpu'] else "cpu"
        logger.info(f"Using device: {device} for transcription")
        
        # Load Whisper model
        try:
            model = whisper.load_model(self.config['whisper_model'], device=device)
            
            # Transcribe audio
            logger.info("Starting transcription...")
            result = model.transcribe(
                audio_path,
                verbose=True,
                language=self.config['language'],
                task="transcribe",
                fp16=False
            )
            
            # Free up GPU memory
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
                
            # Process and save basic transcription
            safe_video_name = re.sub(r"[\W_]+", "_", video_name)
            self._save_basic_transcription(result, safe_video_name)
            
            # Perform speaker diarization if token available
            if self.huggingface_token:
                return self._process_with_speakers(result, audio_path, video_name)
            else:
                logger.warning("Speaker diarization skipped: No HuggingFace token available")
                return None
                
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise

    def _save_basic_transcription(self, result, safe_video_name):
        """Save the basic transcription with timestamps."""
        output_file = Path(self.config['output_dir']) / f"{safe_video_name}_transcription.txt"
        
        # Format transcription with timestamps
        formatted_lines = []
        for segment in result['segments']:
            start_time = self.format_timestamp(segment['start'])
            formatted_lines.append(f"[{start_time}] {segment['text'].strip()}")
            
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("--- Transcription with Timestamps ---\n\n")
                f.write("\n".join(formatted_lines))
            logger.info(f"Basic transcription saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving transcription: {e}")

    def _process_with_speakers(self, result, audio_path, video_name):
        """Process transcription with speaker diarization."""
        logger.info("Starting speaker diarization...")
        
        try:
            # Initialize diarization pipeline
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1", 
                use_auth_token=self.huggingface_token
            )
            
            # Run diarization
            diarization = diarization_pipeline(
                audio_path, 
                num_speakers=self.config['num_speakers']
            )
            
            # Free up resources
            del diarization_pipeline
            if self.config['use_gpu']:
                torch.cuda.empty_cache()
                
            # Build a more accurate speaker mapping
            speaker_turns = []
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                speaker_turns.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker_label
                })
                
            # Sort turns by start time
            speaker_turns.sort(key=lambda x: x['start'])
            
            # Filter Whisper segments by confidence
            min_confidence = 0.5  # Minimum confidence threshold
            filtered_segments = []
            for segment in result['segments']:
                # Skip segments with very low confidence (likely noise)
                if hasattr(segment, 'confidence') and segment['confidence'] < min_confidence:
                    continue
                    
                # Skip very short segments (likely noise)
                if segment['end'] - segment['start'] < 0.3:
                    continue
                    
                filtered_segments.append(segment)
                
            # Map segments to speakers with better algorithm
            speaker_segments = []
            prev_text = ""
            
            for segment in tqdm(filtered_segments, desc="Mapping speakers"):
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()
                
                # Skip empty or very short text
                if len(text) <= 1:
                    continue
                    
                # Skip exact duplicates of previous line
                if text == prev_text:
                    continue
                    
                # Find the most likely speaker for this segment
                best_speaker = None
                most_overlap = 0
                
                for turn in speaker_turns:
                    # Calculate overlap with speaker turn
                    overlap_start = max(turn['start'], start_time)
                    overlap_end = min(turn['end'], end_time)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    
                    if overlap_duration > most_overlap:
                        most_overlap = overlap_duration
                        best_speaker = turn['speaker']
                
                # If no significant overlap found, use the speaker active at the midpoint
                if best_speaker is None:
                    midpoint = (start_time + end_time) / 2
                    for turn in speaker_turns:
                        if turn['start'] <= midpoint <= turn['end']:
                            best_speaker = turn['speaker']
                            break
                
                # Default if still no speaker found
                if best_speaker is None:
                    best_speaker = "UNKNOWN"
                    
                # Format with speaker information
                start_timestamp = self.format_timestamp(start_time)
                formatted_text = f"[{start_timestamp}] Speaker {best_speaker}: {text}"
                speaker_segments.append(formatted_text)
                
                # Remember this line to avoid duplicates
                prev_text = text
                
            # Filter out "junk" entries that might be noise at the end of the file
            # These often contain random characters, repeated short segments, or non-words
            def is_likely_junk(text):
                # Check for patterns that suggest junk/noise
                suspicious_patterns = [
                    r'\d+\s*[A-Z]+\s*$',  # Patterns like "2 AH"
                    r'^\s*\d+\.\s*[A-Z]+\s*$',  # Patterns like "1. DACES"
                    r'^\s*[A-Z]\s*[A-Z]\s*$',  # Patterns like "E A"
                    r'^\s*\d+\s*$',  # Just numbers
                    r'[*]{3,}',  # Multiple asterisks
                    r'[!]{2,}'  # Multiple exclamation marks
                ]
                
                # Check if any of the patterns match
                for pattern in suspicious_patterns:
                    if re.search(pattern, text):
                        return True
                        
                # Check for very short entries
                speaker_part, _, content = text.rpartition(':')
                if len(content.strip()) < 3:
                    return True
                    
                return False
                
            # Apply the junk filter
            filtered_output = []
            for i, segment in enumerate(speaker_segments):
                # If we're in the last 20% of the file, apply more strict filtering
                if i > len(speaker_segments) * 0.8 and is_likely_junk(segment):
                    continue
                filtered_output.append(segment)
                
            # Save to file
            output_file = Path(self.config['output_dir']) / f"{video_name}_transcription_with_speakers.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("--- Transcription with Speaker Diarization ---\n\n")
                f.write("\n".join(filtered_output))
                
            logger.info(f"Speaker diarization saved to: {output_file}")
            return filtered_output
            
        except Exception as e:
            logger.error(f"Speaker diarization error: {str(e)}")
            return None

    def cleanup(self):
        """Clean up temporary files."""
        if not self.config['cleanup_temp']:
            return
            
        logger.info("Cleaning up temporary files...")
        for resource in self.resources_to_cleanup:
            try:
                if os.path.exists(resource):
                    os.remove(resource)
            except Exception as e:
                logger.warning(f"Failed to clean up {resource}: {e}")

    def process_file(self, input_file):
        """Process a single video file from input to transcription."""
        try:
            # Get base filename without extension for output
            video_path = Path(input_file)
            video_name = video_path.stem
            
            # Extract audio from video
            audio_path = self.extract_audio(input_file)
            
            # Transcribe the audio
            logger.info("\nAudio extraction completed. Starting transcription...")
            transcription = self.transcribe(audio_path, video_name)
            
            logger.info(f"\nTranscription completed! Check the '{self.config['output_dir']}' folder.")
            
            # Print preview if transcription is available
            if transcription:
                logger.info("\nTranscription Preview:")
                for line in transcription[:5]:  # Show first 5 lines
                    print(line)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
            return False
        finally:
            self.cleanup()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MP4 to Text Transcription Tool")
    
    parser.add_argument("input", nargs="?", help="Path to input video file (optional)")
    parser.add_argument("--output-dir", default="output", help="Directory for output files")
    parser.add_argument("--model", default="turbo", help="Whisper model to use")
    parser.add_argument("--language", default="it", help="Primary language in the video")
    parser.add_argument("--speakers", type=int, default=2, help="Number of speakers for diarization")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Configure based on arguments
    config = {
        'output_dir': args.output_dir,
        'whisper_model': args.model,
        'language': args.language,
        'num_speakers': args.speakers,
        'use_gpu': not args.cpu and torch.cuda.is_available(),
        'cleanup_temp': not args.keep_temp
    }
    
    # Initialize transcription tool
    tool = TranscriptionTool(config)
    
    # Get input file - either from command line or interactive prompt
    input_file = args.input
    if not input_file:
        print("\nMP4 to Text Transcription Tool")
        print("-" * 50)
        print("Tips for entering file path:")
        print("1. You can drag and drop the file into this window")
        print("2. For relative paths, make sure the file is in the same folder as the script")
        print("3. For full paths, copy the path from your file explorer")
        print("-" * 50)
        input_file = input("\nEnter the path to your video file: ")
        
        # Strip surrounding quotes that might be added when dragging files into terminal
        input_file = input_file.strip("'\"")
    
    # Process the file
    success = tool.process_file(input_file)
    
    if not args.input:  # Only wait for input in interactive mode
        input("\nPress Enter to exit...")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()