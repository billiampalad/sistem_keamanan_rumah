import cv2
import streamlit as st
import numpy as np
import datetime
from collections import deque
from ultralytics import YOLO
import pandas as pd
import time
import torch
import gc
import urllib.parse
import logging
import os
import warnings

# Environment setup
os.environ['YOLO_VERBOSE'] = 'False'
logging.getLogger('ultralytics').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Torch optimization
try:
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
except Exception:
    pass

# Streamlit config
st.set_page_config(
    page_title="Smart Security System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified CSS with requested colors
st.markdown("""
<style>
    :root {
        --cream: #F3F3E0;
        --navy: #27548A;
        --dark-teal: #183B4E;
        --gold: #DDA853;
    }
    
    .stSpinner > div > div {
        color: var(--dark-teal) !important;
        font-weight: 500 !important;
    }
    
    .stApp {
        background: var(--cream);
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, var(--navy), var(--dark-teal));
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: white !important;
        border-right: 3px solid var(--gold);
    }
    
    .sidebar-section {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--navy);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sidebar-section h4 {
        color: var(--navy);
        margin: 0 0 1rem 0;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Camera container */
    .camera-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 2px solid var(--gold);
        margin-bottom: 1rem;
    }
    
    .camera-placeholder {
        background: var(--cream);
        border: 2px dashed var(--gold);
        border-radius: 8px;
        padding: 3rem 2rem;
        text-align: center;
        color: var(--dark-teal);
    }
    
    /* Status cards */
    .status-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--navy);
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-card.alert {
        border-left-color: #dc3545;
        background: #fff5f5;
    }
    
    .status-card h4 {
        color: var(--navy);
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }
    
    .status-card p {
        margin: 0.3rem 0;
        color: var(--dark-teal);
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--navy) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        width: 100% !important;
        margin: 0.2rem 0 !important;
    }
    
    .stButton > button:hover {
        background: var(--dark-teal) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: var(--gold) !important;
        color: var(--dark-teal) !important;
    }
    
    /* Form elements */
    .stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {
        border: 1px solid var(--gold) !important;
        border-radius: 6px !important;
    }
    
    .stSlider > div > div > div > div {
        background: var(--gold) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: var(--navy) !important;
    }
    
    /* Stats */
    .stat-card {
        background: var(--navy);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-card h3 {
        font-size: 1.5rem;
        margin: 0;
        color: var(--gold);
    }
    
    .stat-card p {
        margin: 0.3rem 0 0 0;
        font-size: 0.8rem;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid var(--gold);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ Smart Security System</h1>
    <p>Sistem Keamanan Cerdas dengan Deteksi AI</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        'cap': None,
        'streaming': False,
        'alarm_active': False,
        'last_detection_time': 0,
        'detection_history': deque(maxlen=30),
        'activity_data': deque(maxlen=20)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Alarm functions
def trigger_alarm():
    current_time = time.time()
    if current_time - st.session_state.last_detection_time > 5:  # 5 second cooldown
        st.session_state.alarm_active = True
        st.session_state.last_detection_time = current_time
        
        # Play alarm sound
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            audio_files = ["alarm system.mp3", "alarm.mp3", "beep.wav"]
            for audio_file in audio_files:
                if os.path.exists(audio_file):
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    break
        except Exception:
            pass
        return True
    return False

def stop_alarm():
    st.session_state.alarm_active = False
    try:
        import pygame
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception:
        pass

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")
        model.model.eval()
        # Warm-up
        dummy = np.zeros((240, 240, 3), dtype=np.uint8)
        model(dummy, verbose=False, device='cpu')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Enhanced camera initialization with multiple backends and retry mechanism
def init_camera(source_type, **kwargs):
    """Initialize camera with robust connection and multiple backend support"""
    
    if source_type == "HDMI":
        cam_idx = kwargs.get('cam_idx', 0)
        
        # Try multiple backends for better compatibility
        backends = [
            cv2.CAP_DSHOW,      # DirectShow (Windows)
            cv2.CAP_V4L2,       # Video4Linux (Linux)
            cv2.CAP_AVFOUNDATION, # AVFoundation (macOS)
            cv2.CAP_ANY         # Default backend
        ]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(cam_idx, backend)
                
                if cap.isOpened():
                    # Optimize for fast capture
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30) # Set FPS to 30
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                    
                    # Quick test - retry up to 3 times
                    for attempt in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            # Flush initial frames
                            for _ in range(5):
                                cap.read()
                            return cap
                        time.sleep(0.1)
                
                cap.release()
            except Exception:
                if 'cap' in locals():
                    cap.release()
                continue
        
        return None
    
    elif source_type == "RTSP":
        url = kwargs.get('rtsp_url')
        if not url:
            return None
        
        try:
            # Create capture with optimized settings
            cap = cv2.VideoCapture()
            
            # Set properties before opening
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 25)
            
            # Try to open with timeout simulation
            if cap.open(url):
                # Test connection with multiple attempts
                for attempt in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        # Flush buffer
                        for _ in range(3):
                            cap.read()
                        return cap
                    time.sleep(0.2)
            
            cap.release()
        except Exception:
            if 'cap' in locals():
                cap.release()
        
        return None
    
    return None

# Fast camera detection function
def detect_available_cameras():
    """Quickly detect available camera indices"""
    available_cameras = []
    
    # Test indices 0-4 quickly
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
            cap.release()
        except:
            continue
    
    return available_cameras if available_cameras else [0]

# Optimized detection
def detect_activity(frame, model, conf_threshold):
    if model is None:
        return frame, False
    
    try:
        # Resize for faster processing
        small_frame = cv2.resize(frame, (320, 240))
        results = model(small_frame, verbose=False, conf=conf_threshold, device='cpu')
        
        person_count = 0
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # Person class
                        person_count += 1
                        # Draw bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # Scale to original frame
                        scale_x, scale_y = frame.shape[1] / 320, frame.shape[0] / 240
                        x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update detection history
        if person_count > 0:
            st.session_state.detection_history.append(time.time())
        
        # Check for alert
        current_time = time.time()
        recent_detections = len([t for t in st.session_state.detection_history 
                               if current_time - t < 10])
        
        if recent_detections >= 20:
            trigger_alarm()
        elif person_count == 0:
            stop_alarm()
        
        return frame, st.session_state.alarm_active
        
    except Exception as e:
        st.error(f"Detection error: {e}")
        return frame, False

# Load model
model = load_model()

# Auto-detect available cameras on first load
if 'available_cameras' not in st.session_state:
    with st.spinner("🔍 Mencari kamera yang tersedia..."):
        st.session_state.available_cameras = detect_available_cameras()

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="sidebar-section"><h4>📹 Pengaturan Kamera</h4></div>', unsafe_allow_html=True)
    
    camera_type = st.selectbox(
        "Jenis Kamera",
        ["HDMI Capture Card", "IP Camera (RTSP)"]
    )
    
    if camera_type == "HDMI Capture Card":
        # Show available cameras
        if st.session_state.available_cameras:
            st.info(f"📸 Kamera terdeteksi: {', '.join(map(str, st.session_state.available_cameras))}")
            cam_options = [f"Kamera {i}" for i in st.session_state.available_cameras]
            selected_cam = st.selectbox("Pilih Kamera", cam_options)
            cam_index = st.session_state.available_cameras[cam_options.index(selected_cam)]
        else:
            st.warning("⚠️ Tidak ada kamera terdeteksi otomatis")
            cam_index = st.number_input("Index Kamera Manual", 0, 5, 0)
            
        # Quick test button
        if st.button("🔍 Test Kamera", help="Test koneksi kamera"):
            with st.spinner("Testing..."):
                test_cap = init_camera("HDMI", cam_idx=cam_index)
                if test_cap:
                    st.success("✅ Kamera terdeteksi!")
                    test_cap.release()
                else:
                    st.error("❌ Kamera tidak terdeteksi")
    else:
        camera_url = st.text_input("URL Kamera", placeholder="rtsp://192.168.1.100:554/stream")
        
        # Common RTSP examples
        with st.expander("📋 Contoh URL RTSP"):
            st.text("Generic IP Camera:")
            st.code("rtsp://192.168.1.100:554/stream")
            st.text("Dengan username/password:")
            st.code("rtsp://admin:password@192.168.1.100:554/stream")
            st.text("Hikvision:")
            st.code("rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101")
        
        # Quick test for RTSP
        if camera_url and st.button("🔍 Test Koneksi", help="Test koneksi RTSP"):
            with st.spinner("Testing RTSP..."):
                test_cap = init_camera("RTSP", rtsp_url=camera_url)
                if test_cap:
                    st.success("✅ RTSP terhubung!")
                    test_cap.release()
                else:
                    st.error("❌ Gagal terhubung ke RTSP")
    
    st.markdown('<div class="sidebar-section"><h4>🧠 Pengaturan AI</h4></div>', unsafe_allow_html=True)
    
    confidence = st.slider("Sensitivitas Deteksi", 0.1, 0.9, 0.5, 0.1)
    
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><h4>🔄 Pengaturan Tampilan</h4></div>', unsafe_allow_html=True)
        
        flip_horizontal = st.checkbox("Mirror Horizontal", value=False)
        flip_vertical = st.checkbox("Flip Vertical", value=False) 
    
    st.markdown('<div class="sidebar-section"><h4>🎮 Kontrol</h4></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Mulai", type="primary")
    with col2:
        stop_btn = st.button("⏹ Stop")

# Main layout
col_main, col_status = st.columns([2, 1])

with col_main:
    st.markdown("""<h3 style='color: #183B4E;'>📺 Live Camera Feed</h3> """, unsafe_allow_html=True)
    camera_placeholder = st.empty()
    
    st.markdown("""<h3 style='color: #183B4E;'>📊 Monitor Aktivitas</h3> """, unsafe_allow_html=True)
    chart_placeholder = st.empty()

with col_status:
    st.markdown("""<h3 style='color: #183B4E;'>🚨 Status Sistem</h3> """, unsafe_allow_html=True)
    status_placeholder = st.empty()
    
    st.markdown("""<h3 style='color: #183B4E;'>📈 Statistik</h3> """, unsafe_allow_html=True)
    stats_placeholder = st.empty()

# Enhanced button handlers with faster connection
if start_btn:
    # Stop any existing stream
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    
    st.session_state.streaming = False
    
    # Create progress container
    progress_container = st.empty()
    
    try:
        if camera_type == "HDMI Capture Card":
            with progress_container.container():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.markdown("""
                <p style='color: #183B4E; font-size: 16px; margin: 0;'>
                    🔌 Menghubungkan ke HDMI Capture...
                </p>
                """, unsafe_allow_html=True)
                progress_bar.progress(25)
                
                st.session_state.cap = init_camera("HDMI", cam_idx=cam_index)
                progress_bar.progress(75)
                
                if st.session_state.cap:
                    status_text.text("✅ Kamera HDMI terhubung!")
                    progress_bar.progress(100)
                    time.sleep(0.5)  # Brief pause to show success
                    st.session_state.streaming = True
                else:
                    status_text.text("❌ Gagal menghubungkan HDMI")
                    progress_bar.progress(100)
                    st.error("❌ Tidak dapat terhubung ke kamera HDMI. Coba index kamera lain.")
        
        elif camera_type == "IP Camera (RTSP)":
            if not camera_url:
                st.error("❌ Masukkan URL kamera RTSP!")
            else:
                with progress_container.container():
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🌐 Menghubungkan ke kamera IP...")
                    progress_bar.progress(25)
                    
                    st.session_state.cap = init_camera("RTSP", rtsp_url=camera_url)
                    progress_bar.progress(75)
                    
                    if st.session_state.cap:
                        status_text.text("✅ Kamera IP terhubung!")
                        progress_bar.progress(100)
                        time.sleep(0.5)
                        st.session_state.streaming = True
                    else:
                        status_text.text("❌ Gagal menghubungkan RTSP")
                        progress_bar.progress(100)
                        st.error("❌ Tidak dapat terhubung ke kamera IP. Periksa URL dan koneksi jaringan.")
        
        # Clear progress after connection attempt
        progress_container.empty()
        
        if st.session_state.streaming:
            st.success("🎉 Kamera berhasil terhubung! Monitoring dimulai...")
            # Reset detection history
            st.session_state.detection_history.clear()
            st.session_state.activity_data.clear()
        
    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Error koneksi: {str(e)}")

if stop_btn:
    st.session_state.streaming = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    stop_alarm()
    st.success("⏹️ Monitoring dihentikan")

# Enhanced main processing loop with better error handling
if st.session_state.streaming and st.session_state.cap:
    frame_count = 0
    consecutive_failures = 0
    max_failures = 10  # Allow some frame failures before stopping

    # Create status containers
    connection_status = st.empty()
    
    while st.session_state.streaming:
        try:
            # Flush buffer every 20 frames to reduce lag
            if frame_count % 20 == 0:
                for _ in range(2):
                    st.session_state.cap.grab()
            
            ret, frame = st.session_state.cap.read()
            if not ret or frame is None:
                st.error("❌ Gagal membaca frame dari kamera")
                break
            
            if flip_horizontal:
                frame = cv2.flip(frame, 1)
                
            if flip_vertical:
                frame = cv2.flip(frame, 0)
            
            if ret and frame is not None and frame.size > 0:
                frame_count += 1
                consecutive_failures = 0  # Reset failure counter
                
                # Process every 2nd frame for better performance
                if frame_count % 2 == 0:
                    # Resize for display
                    display_frame = cv2.resize(frame, (640, 480))
                    
                    # Detect activity
                    processed_frame, is_alarm = detect_activity(display_frame, model, confidence)
                    
                    # Display frame
                    with camera_placeholder.container():
                        st.markdown('<div class="camera-container">', unsafe_allow_html=True)
                        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                                use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Update status every 15 frames
                if frame_count % 15 == 0:
                    # Connection status
                    with connection_status.container():
                        st.success("🟢 Kamera terhubung - Live streaming aktif")
                    
                    # System status
                    status_icon = "🚨" if is_alarm else "✅"
                    status_text = "PERINGATAN AKTIF" if is_alarm else "Normal"
                    status_class = "alert" if is_alarm else ""
                    
                    with status_placeholder.container():
                        st.markdown(f"""
                        <div class="status-card {status_class}">
                            <h4>Status: {status_icon} {status_text}</h4>
                            <p>Deteksi: {len(st.session_state.detection_history)}</p>
                            <p>Waktu: {datetime.datetime.now().strftime('%H:%M:%S')}</p>
                            <p>Frame: {frame_count}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Stats
                    total_detections = len(st.session_state.detection_history)
                    current_time = time.time()
                    recent_detections = len([t for t in st.session_state.detection_history 
                                           if current_time - t < 60])
                    
                    with stats_placeholder.container():
                        st.markdown(f"""
                        <div class="stat-card">
                            <h3>{total_detections}</h3>
                            <p>Total Deteksi</p>
                        </div>
                        <div class="stat-card">
                            <h3>{recent_detections}</h3>
                            <p>Deteksi/Menit</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Update chart every 25 frames
                if frame_count % 25 == 0:
                    current_time = time.time()
                    recent_count = len([t for t in st.session_state.detection_history 
                                      if current_time - t < 10])
                    
                    st.session_state.activity_data.append({
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "activity": recent_count
                    })
                    
                    if len(st.session_state.activity_data) > 1:
                        df = pd.DataFrame(list(st.session_state.activity_data))
                        with chart_placeholder.container():
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.line_chart(df.set_index("time")["activity"])
                            st.markdown('</div>', unsafe_allow_html=True)
                
                # Optimized sleep for smooth performance
                time.sleep(0.03)  # ~30 FPS max
                
                # Memory cleanup
                if frame_count % 150 == 0:
                    gc.collect()
                    
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    connection_status.error("🔴 Koneksi kamera terputus")
                    st.error("❌ Koneksi kamera hilang. Coba restart monitoring.")
                    break
                time.sleep(0.1)  # Wait before retry
                
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                st.error(f"❌ Error streaming: {str(e)}")
                break
            time.sleep(0.1)
    
    # Cleanup after loop
    connection_status.empty()
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.streaming = False

else:
    # Placeholder when not streaming
    with camera_placeholder.container():
        st.markdown("""
        <div class="camera-placeholder">
            <h3>📹 Kamera Tidak Aktif</h3>
            <p>Klik tombol <strong>▶ Mulai</strong> untuk memulai monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    
    with status_placeholder.container():
        st.markdown("""
        <div class="status-card">
            <h4>Status: ⏸ Standby</h4>
            <p>Sistem siap digunakan</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_placeholder.container():
        st.markdown("""
        <div class="stat-card">
            <h3>0</h3>
            <p>Total Deteksi</p>
        </div>
        <div class="stat-card">
            <h3>0</h3>
            <p>Deteksi/Menit</p>
        </div>
        """, unsafe_allow_html=True)