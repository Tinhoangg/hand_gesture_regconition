// --- 1. LẤY CÁC THÀNH PHẦN HTML ---
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const gestureResult = document.getElementById('gesture_result');
const pauseBtn = document.getElementById('check-button');
const clearBtn = document.getElementById('delete-button');

let isPaused = false;
let latestPose = null;
let latestHands = null;
let frameCount = 0;

// --- 2. KẾT NỐI WEBSOCKET ---
const socket = io('http://127.0.0.1:8000');
socket.on('connect', () => console.log("✅ Connected to backend:", socket.id));
socket.on('disconnect', () => gestureResult.innerText = "⚠️ Disconnected");
socket.on('prediction_result', (data) => {
    if (!isPaused) gestureResult.innerText = data.text;
});

// --- 3. NÚT BẤM ---
pauseBtn.onclick = () => {
    isPaused = !isPaused;
    pauseBtn.innerText = isPaused ? "Resume" : "Pause";
    if (isPaused) gestureResult.innerText = "⏸️ Paused";
};
clearBtn.onclick = () => {
    socket.emit('clear_sequence');
    gestureResult.innerText = "...";
};

// --- 4. CALLBACKS CỦA MEDIAPIPE ---
function onPoseResults(results) {
    latestPose = results;
    drawFrame();
}
function onHandsResults(results) {
    latestHands = results;
    drawFrame();
}

// --- 5. VẼ & GỬI DỮ LIỆU ---
function drawFrame() {
    frameCount++;

    // Vẽ video lên canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

    let pose_detected = false;
    let hands_detected = false;

    // Vẽ pose (2 vai)
    if (latestPose?.poseLandmarks) {
        pose_detected = true;
        drawLandmarks(canvasCtx, [latestPose.poseLandmarks[11], latestPose.poseLandmarks[12]], { color: '#FF0000', lineWidth: 2 });
    }

    // Vẽ bàn tay
    if (latestHands?.multiHandLandmarks?.length > 0) {
        hands_detected = true;
        for (const landmarks of latestHands.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00CCFF', lineWidth: 3 });
            drawLandmarks(canvasCtx, landmarks, { color: '#FF00FF', lineWidth: 1 });
        }
    }

    canvasCtx.restore();
    if (isPaused || frameCount % 2 !== 0) return; // gửi mỗi 2 frame để nhẹ tải

    // --- GÁN CHÍNH XÁC TAY TRÁI (0) / TAY PHẢI (1) ---
    let leftHand = null;
    let rightHand = null;

    if (hands_detected) {
        const numHands = latestHands.multiHandLandmarks.length;

        for (let i = 0; i < numHands; i++) {
            const hand = latestHands.multiHandLandmarks[i];
            const handedness = latestHands.multiHandedness?.[i]?.classification?.[0]?.label;

            if (handedness === "Left") {
                leftHand = hand;
            } else if (handedness === "Right") {
                rightHand = hand;
            } else {
                // fallback nếu Mediapipe không xác định được
                const avgX = hand.reduce((sum, p) => sum + p.x, 0) / hand.length;
                if (avgX < 0.5) leftHand = hand;
                else rightHand = hand;
            }
        }

        // Nếu chỉ có 1 tay → gán đúng vị trí
        if (numHands === 1) {
            const onlyHand = latestHands.multiHandLandmarks[0];
            const label = latestHands.multiHandedness?.[0]?.classification?.[0]?.label;
            if (label === "Left") {
                leftHand = onlyHand;
                rightHand = null;
            } else if (label === "Right") {
                leftHand = null;
                rightHand = onlyHand;
            } else {
                const avgX = onlyHand.reduce((s, p) => s + p.x, 0) / onlyHand.length;
                if (avgX < 0.5) leftHand = onlyHand;
                else rightHand = onlyHand;
            }
        }
    }

    const pose = pose_detected ? latestPose.poseLandmarks : null;
    socket.emit('process_keypoints', { hands_data: [leftHand, rightHand], pose: pose });
}

// --- 6. KHỞI ĐỘNG CAMERA (CÓ KIỂM TRA QUYỀN) ---
async function initCamera() {
    console.log("📸 Kiểm tra quyền camera...");
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        await videoElement.play();
        console.log("✅ Camera sẵn sàng!");

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
        console.log("🚀 MediaPipe camera started.");
    } catch (err) {
        console.error("❌ Lỗi camera:", err);
        alert("⚠️ Không thể truy cập camera. Hãy cấp quyền trong Site Settings (🔒 > Camera > Allow).");
    }
}

// --- 7. KHỞI TẠO MEDIAPIPE ---
const pose = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
});
pose.setOptions({
    modelComplexity: 0, // nhẹ hơn, nhanh hơn
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
    modelComplexity: 0,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});
hands.onResults(onHandsResults);

// --- 8. KHỞI ĐỘNG ---
initCamera();
console.log("⚙️ Hệ thống khởi động. Đang load model và camera...");
