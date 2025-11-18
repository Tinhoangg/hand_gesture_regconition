import numpy as np
import torch
import torch.nn as nn
def landmarks_to_array_21x3(landmarks_list):
    if not landmarks_list or len(landmarks_list) != 21:
        return np.zeros((21, 3), dtype=np.float32)
    return np.array(
        [[lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)] for lm in landmarks_list],
        dtype=np.float32
    )

def get_normalization_params(pose_landmarks):
    if not pose_landmarks or len(pose_landmarks) < 13:
        return None, None 
    try:
        lm_11 = pose_landmarks[11]
        lm_12 = pose_landmarks[12]
        left = np.array([lm_11.get('x', 0), lm_11.get('y', 0)])
        right = np.array([lm_12.get('x', 0), lm_12.get('y', 0)])
        neck = (left + right) / 2
        shoulder_dist = np.linalg.norm(left - right)
        if shoulder_dist < 0.01: 
            return None, None 
        return neck, shoulder_dist
    except Exception as e:
        return None, None

def normalize_keypoints(keypoints, neck, shoulder_dist):
    normalized = np.copy(keypoints)
    normalized[:,0] = (keypoints[:,0] - neck[0]) / shoulder_dist
    normalized[:,1] = (keypoints[:,1] - neck[1]) / shoulder_dist
    normalized[:,2] = keypoints[:,2] / shoulder_dist
    return normalized.astype(np.float32)

def decode_prediction(prediction_output):
    CLASS_NAMES = [
        'Accept', 'Buy', 'Call', 'Candy', 'Catch', 'Deaf', 'Everyone', 'Food', 'Give', 'Green', 
        'Help', 'Hungry', 'I', 'Learn', 'Light-blue', 'Like', 'Milk', 'Music', 'Name', 'Red',
        'Ship', 'Son', 'Thanks', 'Want', 'Water', 'Where', 'Women', 'Yellow', 'Yogurt', 'You'
    ]
    if isinstance(prediction_output, torch.Tensor):
        prediction_output = prediction_output.cpu().numpy()
    
    class_index = np.argmax(prediction_output)
    

    if class_index < len(CLASS_NAMES):
        return CLASS_NAMES[class_index]
    else:
        return "..." # None