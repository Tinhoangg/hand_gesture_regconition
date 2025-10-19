import cv2
import time
import os

# Mở camera laptop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không mở được camera.")
    exit()

# Lấy kích thước khung hình
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0
num_videos = 16  # số lần quay
duration = 3  # thời gian mỗi video (giây)

for i in range(1, num_videos + 1):
    filename = f"026_001_00{i}.mp4"
    out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

    print(f"\n🎬 Chuẩn bị quay video {i}/{num_videos}...")
    time.sleep(1)  # nghỉ 1s để bạn chuẩn bị

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Không đọc được khung hình.")
            break

        elapsed = time.time() - start_time
        remaining = duration - elapsed

        # Hiển thị đếm ngược trên video
        if remaining > 0:
            cv2.putText(frame, f"Recording {i}/{num_videos} - {remaining:.1f}s",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow('Recording Gesture Dataset', frame)

        # Dừng sau 3s
        if elapsed >= duration:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            exit()

    out.release()
    print(f"✅ Video {i} đã lưu: {filename}")

    # Nghỉ 1 giây giữa các lần quay
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()
print("\n🏁 Hoàn tất quay 16 video gesture!")

# import cv2
# import numpy as np
# import os

# # === 1. Đường dẫn video gốc ===
# video_path = "D:/Semester/Semester5/DPL302/Project/dataset/Milk/021_001_003.mp4"   # đổi theo file của bạn
# output_video = "hand_gesture_30frames.mp4"  # video sau khi giảm frame
# num_samples = 60                  # số frame muốn chọn đều

# # === 2. Đọc video gốc ===
# cap = cv2.VideoCapture(video_path)
# frames = []

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frames.append(frame)

# cap.release()

# total_frames = len(frames)
# print(f"📹 Video có tổng {total_frames} frame.")

# # === 3. Chọn 30 frame đều nhau ===
# if total_frames == 0:
#     raise ValueError("Không đọc được video hoặc video rỗng!")

# idx = np.linspace(0, total_frames - 1, num_samples).astype(int)
# sampled_frames = [frames[i] for i in idx]

# # === 4. Lưu lại thành video mới ===
# frame_height, frame_width = sampled_frames[0].shape[:2]
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 30.0  # bạn có thể để 30 hoặc 15 tuỳ ý

# out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# for frame in sampled_frames:
#     out.write(frame)

# out.release()

# print(f"✅ Video mới đã được lưu: {output_video}")




