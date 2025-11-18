import socketio
from fastapi import FastAPI
import numpy as np
import torch
import torch.nn as nn
from collections import Counter 
import time
from utils.preprocess import get_normalization_params, normalize_keypoints, decode_prediction, landmarks_to_array_21x3
# SETUP
SEQUENCE_LENGTH = 60  
NUM_FEATURES = 126    
client_data = {}      


VOTE_BUFFER_SIZE = 25 
VOTE_THRESHOLD = 20

# LOAD SERVER 
sio = socketio.AsyncServer(async_mode='asgi',cors_allowed_origins='*')
app = FastAPI()
app.mount('/', socketio.ASGIApp(sio))

# load pytorch model
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load('model/hand_gesture.pt', map_location=device)
    model.to(device)
    model.eval()
    print(f" Đã tải model TorchScript 'hand_gesture.pt' thành công. Chạy trên {device}.")
except FileNotFoundError:
    print(" CẢNH BÁO: Không tìm thấy file 'hand_gesture.pt'.")
    model = None
except Exception as e:
    print(f" Lỗi nghiêm trọng khi tải model TorchScript: {e}")
    model = None

# Process websocket event
@sio.event
async def connect(sid, environ):
    print(f" Client đã kết nối: {sid}")
    client_data[sid] = { 
        'seq_buffer': [], 
        'pred_buffer': [], 
        'sentence': [],
        'last_word': '...' 
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
        client_data[sid]['last_word'] = '...' ỉ
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

        neck, shoulder_dist = get_normalization_params(pose_landmarks_data)
        hands_present = hands_data_list[0] is not None or hands_data_list[1] is not None
        
        if not hands_present or neck is None or shoulder_dist is None:
            keypoints_flat = np.zeros(NUM_FEATURES, dtype=np.float32)
        else:
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

                vote_counts = Counter(state['pred_buffer'])
                most_common_word, count = vote_counts.most_common(1)[0]
                
                chosen_word = "..." 

                if count >= VOTE_THRESHOLD:
                    chosen_word = most_common_word

                if chosen_word == "...":
                    # reset last_word
                    state['last_word'] = "..."
                elif chosen_word != state['last_word']:
                    # add new word
                    state['sentence'].append(chosen_word)
                    state['last_word'] = chosen_word
                
                prediction_text = " ".join(state['sentence'])

            else:
                prediction_text = "Đang ở chế độ GIẢ LẬP"
            
            await sio.emit('prediction_result', {'text': prediction_text}, to=sid)

    except Exception as e:
        print(f"Lỗi khi xử lý keypoints: {e}")


if __name__ == '__main__':
    import uvicorn
    print("Khởi chạy server... Truy cập http://127.0.0.1:8000")
    uvicorn.run(app, host='127.0.0.1', port=8000)
