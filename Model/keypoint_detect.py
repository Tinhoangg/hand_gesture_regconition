import mediapipe as mp
import cv2
import numpy as np

# ---------- MP SETUP ------------
class Keypoint_detect:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_pose = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def get_neck_point(self, input_video):
        cap = cv2.VideoCapture(input_video)  # đường dẫn video

        pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.5)

        all_keypoints = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Chuyển ảnh sang RGB cho MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                # Lấy toạ độ 33 keypoints
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                all_keypoints.append(keypoints)

                # Vẽ skeleton lên frame
                self.mp_drawing_pose.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose.close()

        # trung bình 2 vai của tất cả frame
        shoulder_rights = np.array([kp[11][:2] for kp in all_keypoints])
        shoulder_lefts = np.array([kp[12][:2] for kp in all_keypoints])
        neck = (shoulder_lefts.mean(axis=0) + shoulder_rights.mean(axis=0)) / 2
        shoulder_dist = np.linalg.norm(shoulder_lefts.mean(axis=0) - shoulder_rights.mean(axis=0))
        return neck, shoulder_dist


    def normalize_keypoints(self,keypoint, neck_point, shoulder_dist):
        normalized = np.zeros_like(keypoint)
        for i in range(2):
            hand = keypoint[i]
            if np.all(hand == 0):
                continue
            rel = hand - neck_point # dịch tất cả các diêm về gốc tọa độ ở cổ
            
            normalized[i] = rel / shoulder_dist
        
        return normalized.tolist()

    def get_keypoint(self,input_video):
        frame_count = 0
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print("Không mở được video.")
            return
        
        all_keypoints = []  # mỗi phần tử là (2, 21, 2)
        neck_point, shoulder_dist = self.get_neck_point(input_video)

        with self.mp_hands.Hands(static_image_mode=False,
                            max_num_hands=2,
                            min_detection_confidence=0.4,
                            min_tracking_confidence=0.6) as hands:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                # Khởi tạo 2 tay = 0
                keypoints_frame = np.zeros((2, 21, 2))

                if results.multi_hand_landmarks and results.multi_handedness:
                    num_hands = min(len(results.multi_hand_landmarks), 2)
                    print(f"Frame {frame_count}: detected {num_hands} hands")

                    for i in range(num_hands):
                        hand_label = results.multi_handedness[i].classification[0].label  # 'Left' or 'Right'
                        hand_idx = 0 if hand_label == 'Left' else 1

                        hand_landmarks = results.multi_hand_landmarks[i]
                        keypoints = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                        keypoints_frame[hand_idx] = keypoints

                # Chuẩn hóa và lưu lại
                normalized_kp = self.normalize_keypoints(keypoints_frame, neck_point, shoulder_dist)
                all_keypoints.append(normalized_kp)

        cap.release()
        all_keypoints = np.array(all_keypoints)
        print('-------------------------------------------------------------------------------------------')
        print(all_keypoints)
        return all_keypoints
    
if __name__ == "__main__":
    detect = Keypoint_detect()
    detect.get_keypoint("D:/Semester/Semester5/DPL302/Project/30fps/everyone/026_001_005.mp4.mp4")

