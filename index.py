import cv2
import streamlit as st
import numpy as np
import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
import pygame
import threading
import pandas as pd
import time
import torch
from ultralytics.nn.tasks import DetectionModel
import urllib.parse

# Inisialisasi state session
if 'stop_stream' not in st.session_state:
    st.session_state.stop_stream = False
if 'alarm_cooldown' not in st.session_state:
    st.session_state.alarm_cooldown = False
if 'last_alarm_time' not in st.session_state:
    st.session_state.last_alarm_time = 0

torch.serialization.add_safe_globals([
    torch.nn.modules.container.Sequential,
    DetectionModel
])

# Fungsi alarm dengan cooldown
def play_alarm():
    try:
        if st.session_state.alarm_cooldown:
            if time.time() - st.session_state.last_alarm_time > 30:  # Cooldown 30 detik
                st.session_state.alarm_cooldown = False
            else:
                return
        
        pygame.mixer.init()
        pygame.mixer.music.load("alarm system.mp3")
        pygame.mixer.music.play(-1)
        st.session_state.last_alarm_time = time.time()
    except Exception as e:
        st.error(f"Gagal memainkan alarm: {e}")

def stop_alarm():
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            st.session_state.alarm_cooldown = True
    except Exception as e:
        st.error(f"Gagal menghentikan alarm: {e}")

# Fungsi RTSP dengan encoding URL yang aman
def create_rtsp_url(base_url, username=None, password=None):
    if username and password:
        if "@" in base_url:
            return base_url
        
        # Encode username dan password untuk karakter khusus
        encoded_username = urllib.parse.quote(username, safe='')
        encoded_password = urllib.parse.quote(password, safe='')
        
        protocol = base_url.split("://")[0]
        rest = base_url.split("://")[1]
        return f"{protocol}://{encoded_username}:{encoded_password}@{rest}"
    return base_url

# Fungsi deteksi dengan verifikasi ganda
def detect_suspicious_activity(frame, model, conf_threshold, heatmap, aois,
                             activity_logs, max_repeated_movements, alarm_triggered,
                             heatmap_history):
    results = model(frame, verbose=False)  # Nonaktifkan logging YOLO
    current_activities = defaultdict(int)
    suspicious = False
    detection_count = 0  # Untuk verifikasi deteksi ganda

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = result.names[int(box.cls[0])]

            if label == "person" and conf > conf_threshold:
                detection_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Update heatmap dengan resolusi lebih rendah
                small_heatmap = cv2.resize(heatmap, (320, 240))
                small_heatmap[y1//6:y2//6, x1//6:x2//6] += 1
                heatmap = cv2.resize(small_heatmap, (frame.shape[1], frame.shape[0]))

                for idx, (ax1, ay1, ax2, ay2) in enumerate(aois):
                    if x1 >= ax1 and y1 >= ay1 and x2 <= ax2 and y2 <= ay2:
                        current_activities[idx] += 1

    # Verifikasi deteksi ganda (minimal 3 deteksi dalam 5 frame)
    if detection_count >= 3:
        for idx, count in current_activities.items():
            if count > 0:
                activity_logs[idx].append(datetime.datetime.now())

            cutoff = datetime.datetime.now() - datetime.timedelta(seconds=10)
            activity_logs[idx] = [t for t in activity_logs[idx] if t > cutoff]

            if len(activity_logs[idx]) > max_repeated_movements:
                suspicious = True
                if not alarm_triggered[0] and not st.session_state.alarm_cooldown:
                    alarm_triggered[0] = True
                    st.warning(f"\U0001F6A8 *ALARM*: Gerakan mencurigakan di Zona {idx+1}!")
                    threading.Thread(target=play_alarm, daemon=True).start()

    heatmap_max = int(np.max(heatmap))
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    heatmap_history.append({"time": timestamp, "activity": heatmap_max})

    if heatmap_max > 1 and detection_count >= 3:  # Verifikasi ganda
        suspicious = True
        if not alarm_triggered[0] and not st.session_state.alarm_cooldown:
            alarm_triggered[0] = True
            st.warning("\U0001F6A8 *ALARM*: Aktivitas mencurigakan terdeteksi!")
            threading.Thread(target=play_alarm, daemon=True).start()

    elif heatmap_max < 1 or detection_count < 1:
        if alarm_triggered[0]:
            alarm_triggered[0] = False
            stop_alarm()
            st.info("\u2705 Tidak ada aktivitas mencurigakan. Alarm dimatikan.")

    return frame

# Konfigurasi Streamlit
st.set_page_config(page_title="Smart Security System", layout="wide")
st.title("\U0001F512 Smart Security System (Kamera IP)")

with st.sidebar:
    st.header("\u2699\ufe0f Pengaturan Sistem")
    camera_type = st.radio("Tipe Kamera", ["IP Camera (HTTP)", "RTSP Stream"])
    
    if camera_type == "IP Camera (HTTP)":
        ip_camera_url = st.text_input("Masukkan URL Kamera IP", placeholder="http://192.168.1.10:8080/video")
    else:
        ip_camera_url = st.text_input("Masukkan URL RTSP", placeholder="rtsp://username:password@192.168.1.10:554/stream")
        rtsp_username = st.text_input("Username RTSP (jika tidak ada di URL)")
        rtsp_password = st.text_input("Password RTSP (jika tidak ada di URL)", type="password")
    
    conf_threshold = st.slider("*Threshold Kepercayaan Deteksi*", 0.0, 1.0, 0.5, 0.01)
    max_reps = st.number_input("*Threshold Gerakan untuk Alarm*", 1, 50, 5)
    start_stream = st.button("\U0001F3A5 Mulai Streaming")
    stop_stream = st.button("🛑 Hentikan Streaming")

    st.subheader("\U0001F4DC Zona Pengawasan (AOI)")
    num_aois = st.number_input("Jumlah Zona", 0, 5, 1)
    aois = []
    for i in range(num_aois):
        with st.expander(f"Pengaturan Zona {i+1}"):
            x1 = st.slider(f"X1 (Kiri)", 0, 1920, 200, key=f"x1_{i}")
            y1 = st.slider(f"Y1 (Atas)", 0, 1080, 200, key=f"y1_{i}")
            x2 = st.slider(f"X2 (Kanan)", 0, 1920, 800, key=f"x2_{i}")
            y2 = st.slider(f"Y2 (Bawah)", 0, 1080, 600, key=f"y2_{i}")
            aois.append((x1, y1, x2, y2))

col1, col2 = st.columns(2)
with col1:
    st.subheader("\U0001F3A5 Siaran Langsung Kamera")
    camera_placeholder = st.empty()
with col2:
    st.subheader("\U0001F4CA Grafik Aktivitas")
    heatmap_placeholder = st.empty()
    st.subheader("\U0001F4C8 Log Aktivitas")
    log_placeholder = st.empty()

status_text = st.empty()
status_text.info("\U0001F7E2 *Sistem aktif*. Menunggu deteksi...")

# Load model YOLO
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

if start_stream:
    st.session_state.stop_stream = False
    
    if camera_type == "RTSP Stream":
        ip_camera_url = create_rtsp_url(ip_camera_url, rtsp_username, rtsp_password)
    
    # Konfigurasi OpenCV dengan timeout
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    
    # Coba buka stream dengan retry
    max_retries = 3
    retry_delay = 2
    connected = False
    
    for i in range(max_retries):
        try:
            if not cap.open(ip_camera_url):
                raise Exception("Gagal membuka stream")
            
            ret, frame = cap.read()
            if not ret:
                raise Exception("Gagal membaca frame")
            
            connected = True
            break
        except Exception as e:
            if i < max_retries - 1:
                status_text.warning(f"⚠ Percobaan koneksi {i+1}/{max_retries} gagal. Mencoba lagi dalam {retry_delay} detik...")
                time.sleep(retry_delay)
                continue
            else:
                status_text.error(f"❌ Gagal terhubung ke kamera setelah {max_retries} percobaan. Error: {str(e)}")
                st.stop()
    
    if connected:
        status_text.success("✅ Berhasil terhubung ke stream kamera!")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        
        heatmap = np.zeros((frame_height, frame_width), dtype=np.uint8)
        activity_logs = defaultdict(list)
        alarm_triggered = [False]
        heatmap_history = deque(maxlen=100)
        activity_log = []
        
        frame_count = 0
        last_heatmap_update = 0
        
        while not st.session_state.stop_stream:
            try:
                ret, frame = cap.read()
                if not ret:
                    status_text.warning("⚠ Gagal membaca frame. Mencoba menghubungkan kembali...")
                    time.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(ip_camera_url)
                    continue
                
                # Skip frame untuk meningkatkan performa
                frame_count += 1
                if frame_count % 2 != 0:  # Hanya proses setiap frame kedua
                    continue
                
                frame = cv2.resize(frame, (640, 480))  # Resolusi lebih rendah
                
                # Update heatmap setiap 5 detik
                if time.time() - last_heatmap_update > 5:
                    heatmap = (heatmap * 0.9).astype(np.uint8)  # Decay lebih cepat
                    last_heatmap_update = time.time()

                frame = detect_suspicious_activity(
                    frame, model, conf_threshold, heatmap, aois,
                    activity_logs, max_reps, alarm_triggered, heatmap_history
                )
                
                # Tampilkan frame
                camera_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                     channels="RGB", 
                                     use_container_width=True)
                
                # Update grafik setiap 10 frame
                if frame_count % 10 == 0:
                    df_heat = pd.DataFrame(heatmap_history)
                    if not df_heat.empty:
                        heatmap_placeholder.line_chart(df_heat.set_index("time"))
                    
                    # Simpan log aktivitas
                    if len(activity_logs) > 0:
                        log_entry = {
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "activity": "Deteksi orang" if np.max(heatmap) > 0 else "Tidak ada aktivitas",
                            "zona": [k+1 for k,v in activity_logs.items() if len(v) > 0]
                        }
                        activity_log.append(log_entry)
                        log_df = pd.DataFrame(activity_log[-10:])  # Tampilkan 10 log terakhir
                        log_placeholder.dataframe(log_df)
                
                # Periksa tombol stop
                if stop_stream:
                    st.session_state.stop_stream = True
                    break
                    
            except Exception as e:
                status_text.error(f"❌ Error selama pemrosesan: {str(e)}")
                break
        
        cap.release()
        stop_alarm()
        
        # Simpan log ke CSV saat stream berhenti
        if len(activity_log) > 0:
            pd.DataFrame(activity_log).to_csv("activity_log.csv", index=False)
        
        status_text.info("🚫 Stream dihentikan")