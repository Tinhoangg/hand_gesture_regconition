// --- 1. LẤY CÁC THÀNH PHẦN HTML ---
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const gestureResultElement = document.getElementById('gesture_result');
const pauseButton = document.getElementById('check-button');
const deleteButton = document.getElementById('delete-button');

// --- 2. BIẾN TRẠNG THÁI ---
let latestPoseResults = null;
let latestHandsResults = null; 
let isPaused = false;

// --- 3. KẾT NỐI WEBSOCKET ---
const socket = io('http://127.0.0.1:8000');

socket.on('connect', () => console.log(' Đã kết nối tới server:', socket.id));
socket.on('disconnect', () => gestureResultElement.innerText = "...Mất kết nối...");

socket.on('prediction_result', (data) => {
    if (!isPaused) {
        gestureResultElement.innerText = data.text;
    }
});

// --- 4. XỬ LÝ NÚT BẤM ---
pauseButton.addEventListener('click', () => {
    isPaused = !isPaused;
    pauseButton.innerText = isPaused ? 'Resume (Tiếp Tục)' : 'Pause (Tạm Dừng)';
    if (isPaused) gestureResultElement.innerText = "Tạm dừng";
});

deleteButton.addEventListener('click', () => {
    socket.emit('clear_sequence');
    console.log(' Đã gửi yêu cầu xóa chuỗi (clear_sequence).');
    gestureResultElement.innerText = "...";
});

// --- 5. HÀM CALLBACK CỦA MEDIAPIPE ---

function onPoseResults(results) {
    latestPoseResults = results;
    drawCombinedResults(); 
}

function onHandsResults(results) {
    latestHandsResults = results;
    drawCombinedResults(); 
}

/**
 * Hàm này chịu trách nhiệm VẼ và GỬI DỮ LIỆU
 */
function drawCombinedResults() {
    // --- Vẽ ---
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

    let pose_detected = false;
    let hands_detected = false;

    if (latestPoseResults && latestPoseResults.poseLandmarks) {
        pose_detected = true;
        if (latestPoseResults.poseLandmarks[11] && latestPoseResults.poseLandmarks[12]) {
             drawLandmarks(canvasCtx, [latestPoseResults.poseLandmarks[11], latestPoseResults.poseLandmarks[12]], { color: '#FF0000', lineWidth: 2 });
        }
    }
    if (latestHandsResults && latestHandsResults.multiHandLandmarks && latestHandsResults.multiHandLandmarks.length > 0) {
        hands_detected = true;
        for (const landmarks of latestHandsResults.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00CCFF', lineWidth: 5 });
            drawLandmarks(canvasCtx, landmarks, { color: '#FF00FF', lineWidth: 2 });
        }
    }
    canvasCtx.restore();

    // --- Gửi dữ liệu (Logic 2 tay) ---
    if (isPaused) return; 

    let leftHand_kps = null;
    let rightHand_kps = null;

    // Sửa lỗi TypeError
    if (hands_detected) {
        for (let i = 0; i < latestHandsResults.multiHandLandmarks.length; i++) {
            const hand = latestHandsResults.multiHandLandmarks[i];
            let handedness = null;

            if (latestHandsResults.multiHandedness &&
                latestHandsResults.multiHandedness[i] && 
                latestHandsResults.multiHandedness[i].classification &&
                latestHandsResults.multiHandedness[i].classification[0]) 
            {
                handedness = latestHandsResults.multiHandedness[i].classification[0].label;
            }

            if (handedness === 'Left') {
                leftHand_kps = hand;
            } else if (handedness === 'Right') {
                rightHand_kps = hand;
            } 
            // Xử lý dự phòng (nếu handedness bị null)
            else if (latestHandsResults.multiHandLandmarks.length === 1) {
                leftHand_kps = hand;
            }
        }
    }
    
    const pose_landmarks = pose_detected ? latestPoseResults.poseLandmarks : null;

    // LUÔN GỬI dữ liệu (kể cả null) lên server
    const data = {
        hands_data: [leftHand_kps, rightHand_kps],
        pose: pose_landmarks 
    };
    
    socket.emit('process_keypoints', data);
}

// --- 6. KHỞI TẠO MEDIAPIPE ---
const pose = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
});
pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});
pose.onResults(onPoseResults);

const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
    maxNumHands: 2, 
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});
hands.onResults(onHandsResults);

// --- 7. KHỞI ĐỘNG CAMERA VÀ VÒNG LẶP ---
const camera = new Camera(videoElement, {
    onFrame: async () => {
        if (!isPaused) {
            await pose.send({ image: videoElement });
            await hands.send({ image: videoElement });
        }
    },
    width: 640,
    height: 480
});
camera.start();

console.log("...Đang khởi tạo model và camera, vui lòng chờ...");

