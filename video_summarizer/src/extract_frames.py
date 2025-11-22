import cv2
import os
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from video with progress tracking.

    Args:
        video_path: Path to input video
        output_folder: Directory to save extracted frames
        frame_rate: Frames per second to extract (default: 1)

    Returns:
        Number of frames extracted
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate) if fps > 0 else 1

    logger.info(f"Extracting frames from {video_path}")
    logger.info(f"Total frames: {total_frames}, FPS: {fps}, Extracting every {frame_interval} frames")

    count = 0
    extracted = 0

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted += 1

            count += 1
            pbar.update(1)

    cap.release()
    logger.info(f"Extracted {extracted} frames to {output_folder}")
    return extracted
