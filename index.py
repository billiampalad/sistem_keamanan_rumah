import cv2
import streamlit as st
import numpy as np
import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
import pandas as pd
import time
import torch
import gc
import urllib.parse
import logging
import os

# Konfigurasi logging untuk mengurangi output
logging.getLogger('ultralytics').setLevel(logging.WARNING)
os.environ['YOLO_VERBOSE'] = 'False'

# Optimisasi torch untuk CPU
torch.set_num_threads(2)  # Batasi thread untuk CPU yang terbatas
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Inisialisasi state session yang lebih efisien
if 'stop_stream' not in st.session_state:
    st.session_state.stop_stream = False
if 'alarm_active' not in st.session_state:
    st.session_state.alarm_active = False
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0

# Fungsi alarm sederhana tanpa pygame (lebih ringan)
def trigger_alarm():
    """Trigger alarm dengan cooldown sederhana"""
    current_time = time.time()
    if current_time - st.session_state.last_detection_time > 10:  # Cooldown 10 detik
        st.session_state.alarm_active = True
        st.session_state.last_detection_time = current_time
        return True
    return False

def stop_alarm():
    """Stop alarm"""
    st.session_state.alarm_active = False

# Fungsi RTSP yang dioptimasi
def create_rtsp_url(base_url, username=None, password=None):
    """Buat URL RTSP dengan encoding yang aman"""
    if username and password and "@" not in base_url:
        encoded_username = urllib.parse.quote(username, safe='')
        encoded_password = urllib.parse.quote(password, safe='')
        protocol = base_url.split("://")[0]
        rest = base_url.split("://")[1]
        return f"{protocol}://{encoded_username}:{encoded_password}@{rest}"
    return base_url

# Fungsi deteksi yang dioptimasi
def detect_suspicious_activity(frame, model, conf_threshold, detection_history, frame_count):
    """
    Deteksi aktivitas mencurigakan dengan optimisasi performa
    """
    # Resize frame untuk deteksi yang lebih cepat
    small_frame = cv2.resize(frame, (320, 240))
    
    # Jalankan deteksi hanya setiap 5 frame
    if frame_count % 5 != 0:
        return frame, False
    
    try:
        # Deteksi dengan model yang sudah dioptimasi
        results = model(small_frame, verbose=False, conf=conf_threshold)
        
        person_detected = False
        detection_count = 0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Hanya deteksi orang (class 0 di COCO dataset)
                    if cls == 0 and conf > conf_threshold:
                        detection_count += 1
                        person_detected = True
                        
                        # Scale koordinat kembali ke ukuran asli
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1*2), int(y1*2), int(x2*2), int(y2*2)
                        
                        # Gambar bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Update history deteksi
        current_time = time.time()
        if person_detected:
            detection_history.append(current_time)
        
        # Bersihkan history lama (lebih dari 30 detik)
        detection_history = deque([t for t in detection_history if current_time - t < 30], maxlen=50)
        
        # Trigger alarm jika ada banyak deteksi dalam waktu singkat
        recent_detections = len([t for t in detection_history if current_time - t < 10])
        suspicious = recent_detections >= 3
        
        return frame, suspicious
        
    except Exception as e:
        st.error(f"Error dalam deteksi: {str(e)}")
        return frame, False

# Konfigurasi Streamlit yang dioptimasi
st.set_page_config(
    page_title="Smart Security System", 
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar collapsed by default
)

st.title("🔒 Smart Security System (Optimized)")

# Sidebar yang lebih sederhana
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Input kamera
    camera_type = st.selectbox("Camera Type", ["IP Camera (HTTP)", "RTSP Stream"])
    
    if camera_type == "IP Camera (HTTP)":
        camera_url = st.text_input("Camera URL", placeholder="http://192.168.1.10:8080/video")
    else:
        camera_url = st.text_input("RTSP URL", placeholder="rtsp://192.168.1.10:554/stream")
        rtsp_user = st.text_input("Username (optional)")
        rtsp_pass = st.text_input("Password (optional)", type="password")
    
    # Pengaturan deteksi
    conf_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.5, 0.1)
    
    # Tombol kontrol
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("🎥 Start", use_container_width=True)
    with col2:
        stop_btn = st.button("🛑 Stop", use_container_width=True)

# Layout utama
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 Status")
    status_placeholder = st.empty()
    
    st.subheader("📝 Detection Log")
    log_placeholder = st.empty()

# Status system
system_status = st.empty()

# Load model YOLO yang dioptimasi
@st.cache_resource
def load_optimized_model():
    """Load model YOLO nano yang sudah dioptimasi"""
    try:
        # Gunakan YOLOv8 nano - model terkecil dan tercepat
        model = YOLO("yolov8n.pt")
        
        # Optimisasi model untuk CPU
        model.model.eval()
        
        # Warm-up model dengan frame dummy
        dummy_frame = np.zeros((320, 240, 3), dtype=np.uint8)
        _ = model(dummy_frame, verbose=False)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
with st.spinner("Loading AI model..."):
    model = load_optimized_model()

if model is None:
    st.error("Failed to load YOLO model. Please check your installation.")
    st.stop()

# Main processing loop
if start_btn:
    st.session_state.stop_stream = False
    
    # Siapkan URL kamera
    if camera_type == "RTSP Stream" and rtsp_user and rtsp_pass:
        camera_url = create_rtsp_url(camera_url, rtsp_user, rtsp_pass)
    
    # Konfigurasi OpenCV untuk performa optimal
    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimal
    cap.set(cv2.CAP_PROP_FPS, 15)  # FPS lebih rendah
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Coba koneksi dengan timeout
    try:
        if not cap.open(camera_url):
            raise Exception("Cannot connect to camera")
        
        # Test read frame
        ret, test_frame = cap.read()
        if not ret:
            raise Exception("Cannot read from camera")
        
        system_status.success("✅ Connected to camera successfully!")
        
        # Inisialisasi variabel
        detection_history = deque(maxlen=50)
        activity_log = []
        frame_count = 0
        last_log_time = 0
        
        # Main loop
        while not st.session_state.stop_stream:
            try:
                ret, frame = cap.read()
                if not ret:
                    system_status.warning("⚠️ Frame read failed, retrying...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Resize frame untuk tampilan
                display_frame = cv2.resize(frame, (640, 480))
                
                # Deteksi aktivitas
                processed_frame, is_suspicious = detect_suspicious_activity(
                    display_frame, model, conf_threshold, detection_history, frame_count
                )
                
                # Update tampilan video
                video_placeholder.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )
                
                # Update status
                current_time = time.time()
                recent_detections = len([t for t in detection_history if current_time - t < 10])
                
                if is_suspicious:
                    if trigger_alarm():
                        status_placeholder.error("🚨 SUSPICIOUS ACTIVITY DETECTED!")
                        activity_log.append({
                            "time": datetime.datetime.now().strftime("%H:%M:%S"),
                            "event": "Suspicious Activity",
                            "confidence": "High"
                        })
                else:
                    if st.session_state.alarm_active:
                        stop_alarm()
                        status_placeholder.success("✅ Area Clear")
                    else:
                        status_placeholder.info(f"👀 Monitoring... ({recent_detections} recent detections)")
                
                # Update log setiap 5 detik
                if current_time - last_log_time > 5:
                    log_df = pd.DataFrame(activity_log[-10:])  # Show last 10 entries
                    if not log_df.empty:
                        log_placeholder.dataframe(log_df, use_container_width=True)
                    last_log_time = current_time
                
                # Garbage collection setiap 100 frame
                if frame_count % 100 == 0:
                    gc.collect()
                
                # Check stop button
                if stop_btn:
                    st.session_state.stop_stream = True
                    break
                
                # Small delay untuk mengurangi CPU usage
                time.sleep(0.03)  # ~30 FPS max
                
            except Exception as e:
                system_status.error(f"❌ Processing error: {str(e)}")
                break
        
        # Cleanup
        cap.release()
        system_status.info("🔴 Stream stopped")
        
        # Save log jika ada
        if activity_log:
            log_df = pd.DataFrame(activity_log)
            csv = log_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Activity Log",
                data=csv,
                file_name=f"security_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        system_status.error(f"❌ Connection failed: {str(e)}")

if stop_btn:
    st.session_state.stop_stream = True
    stop_alarm()

# Footer
st.markdown("---")
st.markdown("🔒 **Smart Security System** - Optimized for lightweight deployment")
st.markdown("💡 *Tip: Use lower resolution cameras for better performance on limited resources*")