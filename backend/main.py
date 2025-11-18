import socketio
from fastapi import FastAPI
import numpy as np
import torch
import torch.nn as nn
from collections import Counter 
import time
# --- Cài Đặt Chung ---
SEQUENCE_LENGTH = 60  # Khớp với MAX_FRAMES (60)
NUM_FEATURES = 126    # 63 (tay trái) + 63 (tay phải)
client_data = {}      # Bộ đệm cho mỗi client

# ### <<< THAY ĐỔI Ở ĐÂY (Làm chậm lại) >>> ###
# Giữ bộ đệm là 25 frame
VOTE_BUFFER_SIZE = 25 
# NGƯỠNG MỚI: Từ phải xuất hiện ít nhất 20/25 lần thì mới "chốt"
VOTE_THRESHOLD = 20

# --- 1. KHỞI TẠO SERVER ---
sio = socketio.AsyncServer(async_mode='asgi',cors_allowed_origins='*')
app = FastAPI()
app.mount('/', socketio.ASGIApp(sio))

# --- 2. TẢI MODEL PYTORCH ---
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load('hand_gesture.pt', map_location=device)
    model.to(device)
    model.eval()
    print(f" Đã tải model TorchScript 'hand_gesture.pt' thành công. Chạy trên {device}.")
except FileNotFoundError:
    print(" CẢNH BÁO: Không tìm thấy file 'hand_gesture.pt'.")
    model = None
except Exception as e:
    print(f" Lỗi nghiêm trọng khi tải model TorchScript: {e}")
    model = None

# --- 3. CÁC HÀM HỖ TRỢ (SAO CHÉP TỪ real_time.py) ---
# (Các hàm landmarks_to_array_21x3, get_normalization_params, 
#  normalize_keypoints, decode_prediction giữ nguyên)

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
    
    # print(f"--- DEBUG: Model predicted index: {class_index}") # (Tắt bớt debug)

    if class_index < len(CLASS_NAMES):
        return CLASS_NAMES[class_index]
    else:
        return "..." # Lớp "im lặng"

# --- 4. XỬ LÝ SỰ KIỆN WEBSOCKET ---
@sio.event
async def connect(sid, environ):
    print(f" Client đã kết nối: {sid}")
    client_data[sid] = { 
        'seq_buffer': [], 
        'pred_buffer': [], 
        'sentence': [],
        'last_word': '...' # Biến để theo dõi "khoảng nghỉ"
    }

@sio.event
async def disconnect(sid):
    print(f" Client đã ngắt kết nối: {sid}")
    if sid in client_data: del client_data[sid]
        
@sio.event
async def clear_sequence(sid):
    if sid in client_data: 
        client_data[sid]['seq_buffer'] = []
        client_data[sid]['pred_buffer'] = []
        client_data[sid]['sentence'] = []
        client_data[sid]['last_word'] = '...' # Reset khoảng nghỉ
        print(f" Đã xóa buffer cho client: {sid}")
        await sio.emit('prediction_result', {'text': '...'}, to=sid) 

@sio.event
async def process_keypoints(sid, data):
    try:
        if sid not in client_data:
            client_data[sid] = { 'seq_buffer': [], 'pred_buffer': [], 'sentence': [], 'last_word': '...' }
        
        state = client_data.get(sid)

        hands_data_list = data.get('hands_data') 
        pose_landmarks_data = data.get('pose')

        # Logic chuẩn hóa (Giống hệt file training)
        neck, shoulder_dist = get_normalization_params(pose_landmarks_data)
        hands_present = hands_data_list[0] is not None or hands_data_list[1] is not None
        
        if not hands_present or neck is None or shoulder_dist is None:
            keypoints_flat = np.zeros(NUM_FEATURES, dtype=np.float32)
        else:
            # (Logic chuẩn hóa 2 tay giữ nguyên...)
            hand_landmarks_left = hands_data_list[0]
            if hand_landmarks_left is None:
                norm_kps_left_flat = np.zeros(NUM_FEATURES // 2, dtype=np.float32)
            else:
                keypoints_left_21x3 = landmarks_to_array_21x3(hand_landmarks_left)
                normalized_kps_left = normalize_keypoints(keypoints_left_21x3, neck, shoulder_dist)
                norm_kps_left_flat = normalized_kps_left.flatten()

            hand_landmarks_right = hands_data_list[1]
            if hand_landmarks_right is None:
                norm_kps_right_flat = np.zeros(NUM_FEATURES // 2, dtype=np.float32)
            else:
                keypoints_right_21x3 = landmarks_to_array_21x3(hand_landmarks_right)
                normalized_kps_right = normalize_keypoints(keypoints_right_21x3, neck, shoulder_dist)
                norm_kps_right_flat = normalized_kps_right.flatten()
            
            keypoints_flat = np.concatenate((norm_kps_left_flat, norm_kps_right_flat))
        
        # (Tắt bớt debug)
        # print(f"--- DEBUG: Đã nhận frame. Tổng giá trị: {np.sum(keypoints_flat)}") 

        state['seq_buffer'].append(keypoints_flat)

        if len(state['seq_buffer']) > SEQUENCE_LENGTH:
            state['seq_buffer'].pop(0)

        prediction_text = "..."
        
        if len(state['seq_buffer']) == SEQUENCE_LENGTH:
            if model is not None:
                numpy_seq = np.array(state['seq_buffer'])
                seq_input = torch.tensor([numpy_seq], dtype=torch.float32).to(device)
                
                if np.sum(numpy_seq) != 0:
                    with torch.no_grad():
                        model_output = model(seq_input)
                    pred_class = decode_prediction(model_output[0])
                else:
                    pred_class = "..." 
                
                # --- LOGIC TẠO CÂU MỚI (CÓ NGƯỠNG BIỂU QUYẾT) ---
                state['pred_buffer'].append(pred_class)
                
                if len(state['pred_buffer']) > VOTE_BUFFER_SIZE:
                    state['pred_buffer'].pop(0)

                # 1. Đếm tất cả các dự đoán trong bộ đệm
                vote_counts = Counter(state['pred_buffer'])
                # 2. Tìm từ_ưu_thế (từ_thắng)
                most_common_word, count = vote_counts.most_common(1)[0]
                
                chosen_word = "..." # Mặc định là "im lặng"

                # 3. Chỉ "chốt" từ nếu nó vượt qua NGƯỠNG
                if count >= VOTE_THRESHOLD:
                    chosen_word = most_common_word

                # 4. Logic thêm câu (giống như trước, nhưng dùng chosen_word)
                if chosen_word == "...":
                    # Đã có "khoảng nghỉ", reset last_word
                    state['last_word'] = "..."
                elif chosen_word != state['last_word']:
                    # Đây là từ MỚI VÀ ỔN ĐỊNH
                    state['sentence'].append(chosen_word)
                    state['last_word'] = chosen_word
                
                prediction_text = " ".join(state['sentence'])
                # --- KẾT THÚC LOGIC MỚI ---

            else:
                prediction_text = "Đang ở chế độ GIẢ LẬP"
            
            await sio.emit('prediction_result', {'text': prediction_text}, to=sid)

    except Exception as e:
        print(f"Lỗi khi xử lý keypoints: {e}")


# (Code để chạy server)
if __name__ == '__main__':
    import uvicorn
    print("Khởi chạy server... Truy cập http://127.0.0.1:8000")
    uvicorn.run(app, host='127.0.0.1', port=8000)