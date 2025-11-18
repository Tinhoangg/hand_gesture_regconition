// --- 1. LẤY HTML ELEMENTS ---
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const gestureResult = document.getElementById('gesture_result');
const pauseBtn = document.getElementById('check-button');
const clearBtn = document.getElementById('delete-button');

let isPaused = false;
let latestPose = null, latestHands = null;
let frameCount = 0;

// --- 2. SOCKET.IO ---
const socket = io('http://127.0.0.1:8000');
socket.on('connect', () => console.log("✅ Connected to server:", socket.id));
socket.on('disconnect', () => gestureResult.innerText = "Mất kết nối...");
socket.on('prediction_result', data => { if (!isPaused) gestureResult.innerText = data.text; });

// --- 3. BUTTON EVENTS ---
pauseBtn.onclick = () => {
  isPaused = !isPaused;
  pauseBtn.innerText = isPaused ? "Resume" : "Pause";
  if (isPaused) gestureResult.innerText = "Tạm dừng";
};
clearBtn.onclick = () => { socket.emit('clear_sequence'); gestureResult.innerText = "..."; };

// --- 4. CALLBACKS ---
function onPoseResults(r) { latestPose = r; drawFrame(); }
function onHandsResults(r) { latestHands = r; drawFrame(); }

// --- 5. VẼ & GỬI DỮ LIỆU ---
function drawFrame() {
  frameCount++;
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

  let pose_detected = false, hands_detected = false;

  if (latestPose?.poseLandmarks) {
    pose_detected = true;
    drawLandmarks(canvasCtx, [latestPose.poseLandmarks[11], latestPose.poseLandmarks[12]], { color: '#FF0000', lineWidth: 2 });
  }
  if (latestHands?.multiHandLandmarks?.length > 0) {
    hands_detected = true;
    for (const hand of latestHands.multiHandLandmarks) {
      drawConnectors(canvasCtx, hand, HAND_CONNECTIONS, { color: '#00CCFF', lineWidth: 4 });
      drawLandmarks(canvasCtx, hand, { color: '#FF00FF', lineWidth: 2 });
    }
  }

  if (isPaused || frameCount % 2 !== 0) return; // gửi mỗi 2 frame

  let left = null, right = null;
  if (hands_detected) {
    for (let i = 0; i < latestHands.multiHandLandmarks.length; i++) {
      const hand = latestHands.multiHandLandmarks[i];
      const handed = latestHands.multiHandedness?.[i]?.classification?.[0]?.label;
      if (handed === "Left") left = hand;
      else if (handed === "Right") right = hand;
      else if (latestHands.multiHandLandmarks.length === 1) left = hand;
    }
  }

  const pose = pose_detected ? latestPose.poseLandmarks : null;
  socket.emit('process_keypoints', { hands_data: [left, right], pose: pose });
}

// --- 6. CAMERA KHỞI ĐỘNG ---
async function initCamera() {
  console.log("📸 Đang kiểm tra camera...");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
    await videoElement.play();
    console.log("✅ Camera sẵn sàng!");

    const cam = new Camera(videoElement, {
      onFrame: async () => {
        if (!isPaused) {
          await pose.send({ image: videoElement });
          await hands.send({ image: videoElement });
        }
      },
      width: 640,
      height: 480
    });
    cam.start();
  } catch (err) {
    console.error("❌ Lỗi camera:", err);
    alert("⚠️ Không thể truy cập camera. Hãy bật quyền trong Site Settings.");
  }
}

// --- 7. MEDIAPIPE ---
const pose = new Pose({
  locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
});
pose.setOptions({
  modelComplexity: 0, // giảm độ phức tạp → nhanh hơn
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
pose.onResults(onPoseResults);

const hands = new Hands({
  locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 0, // nhẹ hơn
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
hands.onResults(onHandsResults);

// --- 8. START ---
initCamera();
console.log("🚀 Hệ thống khởi động...");
