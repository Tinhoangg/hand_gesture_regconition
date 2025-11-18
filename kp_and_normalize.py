import mediapipe as mp
import cv2
import numpy as np
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
# config
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)


# Normalize
def get_neck_shoulder(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None, None
    lm = results.pose_landmarks.landmark
    left = np.array([lm[11].x, lm[11].y])
    right = np.array([lm[12].x, lm[12].y])
    neck = (left + right) / 2
    shoulder_dist = np.linalg.norm(left - right)
    return neck, shoulder_dist

def normalize_keypoints(keypoints, neck, shoulder_dist):
    if keypoints is None or neck is None or shoulder_dist is None or shoulder_dist == 0:
        return np.zeros((21,3), dtype=np.float32)
    normalized = np.copy(keypoints)
    normalized[:,0] = (keypoints[:,0] - neck[0]) / shoulder_dist
    normalized[:,1] = (keypoints[:,1] - neck[1]) / shoulder_dist
    normalized[:,2] = keypoints[:,2] / shoulder_dist
    return normalized.astype(np.float32)

def extract_hands_normalized(frame):
    neck, shoulder_dist = get_neck_shoulder(frame)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    all_hands = np.zeros((2,21,3), dtype=np.float32)
    if results.multi_hand_landmarks and neck is not None and shoulder_dist is not None:
        for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_label.classification[0].label
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            keypoints_norm = normalize_keypoints(keypoints, neck, shoulder_dist)
            if label == 'Left':
                all_hands[0] = keypoints_norm
            else:
                all_hands[1] = keypoints_norm
    return all_hands