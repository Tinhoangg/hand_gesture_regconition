import os
import cv2
import numpy as np

def save_30fps_video(video_path,output_video,num_frame=60):
    # === read video ===
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    total_frames = len(frames)
    print(f" Video có tổng {total_frames} frame.")

    # === choose n frame  ===
    if total_frames == 0:
        raise ValueError("Không đọc được video hoặc video rỗng!")

    idx = np.linspace(0, total_frames - 1, num_frame).astype(int)
    sampled_frames = [frames[i] for i in idx]

    # === save as 30 fps video ===
    frame_height, frame_width = sampled_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0  

    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        raise RuntimeError(f"Không mở được file video để ghi: {output_video}")

    for frame in sampled_frames:
        out.write(frame)

    out.release()

    print(f"✅ Video mới đã được lưu: {output_video}")

if __name__ == "__main__":
    path = 'own_label'
    video_files = []
    for root, dirs, files in os.walk(path):
        if os.path.basename(root):
            for file in files:
                if file.endswith(".mp4"):
                    full_path = os.path.join(root, file)            #get file name 
                    file_name = os.path.basename(full_path)
                    #get folder parent name
                    parent_folder = os.path.basename(os.path.dirname(full_path))

                    os.makedirs(f"30fps/{parent_folder}", exist_ok=True)
                    save_30fps_video(full_path, f"30fps/{parent_folder}/{file_name}.mp4")












    