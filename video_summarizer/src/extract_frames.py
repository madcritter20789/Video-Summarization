import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{count}.jpg"), frame)
        count += 1

    cap.release()

# Example usage
# extract_frames('data/video/sample.mp4', 'data/frames/', frame_rate=1)
