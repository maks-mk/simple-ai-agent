import asyncio
import os
import logging
from typing import Literal, Dict, Any, Optional
from langchain_core.tools import tool

# Attempt to import yt_dlp, handle if missing
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

logger = logging.getLogger("agent")

def _build_ydl_options(media_type: str, resolution: str) -> Dict[str, Any]:
    """
    Constructs the configuration dictionary for yt-dlp based on user preferences.
    """
    # Base options for all downloads
    opts = {
        'quiet': True,
        'no_warnings': True,
        'outtmpl': '%(title)s.%(ext)s',
        'restrictfilenames': True,  # ASCII chars only to be safe
        'noplaylist': True,         # Download only single video
    }

    if media_type == "audio":
        # Audio configuration: Prefer MP3, fallback to best audio
        opts.update({
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            # Continue if postprocessing fails (e.g. no ffmpeg)
            'ignoreerrors': True,
        })
    else:
        # Video configuration
        if resolution.isdigit():
            # "1080" -> best video <= 1080p + best audio
            fmt = f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]/best"
        elif resolution == "worst":
            fmt = "worst"
        else:
            # Default "best"
            fmt = "bestvideo+bestaudio/best"
        
        opts['format'] = fmt

    return opts

def _run_sync_download(url: str, opts: Dict[str, Any], media_type: str) -> str:
    """
    Executes the blocking yt-dlp download process.
    """
    try:
        logger.info(f"🎥 Starting media download: {url} (Type: {media_type})")
        
        with yt_dlp.YoutubeDL(opts) as ydl:
            # extract_info(download=True) performs the download and returns metadata
            info = ydl.extract_info(url, download=True)
            
            # Determine the final filename
            filename = ydl.prepare_filename(info)
            
            # Check for post-processed audio file (e.g., .mp3)
            if media_type == "audio":
                base, _ = os.path.splitext(filename)
                mp3_name = f"{base}.mp3"
                if os.path.exists(mp3_name):
                    filename = mp3_name
            
            title = info.get('title', 'Unknown')
            return f"Success: Downloaded '{filename}' (Title: {title})"
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Media download failed: {error_msg}")
        
        if "ffmpeg" in error_msg.lower():
            return (
                f"Error: Download failed due to missing FFMPEG. "
                f"Try setting resolution='best' (legacy) or check system requirements. Details: {error_msg}"
            )
        return f"Error downloading media: {error_msg}"

@tool
async def download_media(
    url: str, 
    media_type: Literal["video", "audio"] = "video", 
    resolution: str = "best"
) -> str:
    """
    Downloads video/audio via yt-dlp to CWD.
    Args: url, media_type ('video'/'audio'), resolution ('1080', 'best').
    """
    if not yt_dlp:
        return "Error: 'yt-dlp' library is not installed. Please install it via pip."

    # Build configuration options
    opts = _build_ydl_options(media_type, resolution)

    # Run in a separate thread to avoid blocking the async event loop
    return await asyncio.to_thread(_run_sync_download, url, opts, media_type)
