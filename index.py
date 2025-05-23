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

# Load model YOLOv8
model = YOLO("yolov8n.pt")

# Function to play the alarm
def play_alarm():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("alarm system.mp3")
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"Failed to play alarm: {e}")

# Function to stop the alarm
def stop_alarm():
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception as e:
        print(f"Failed to stop alarm: {e}")

# Function to create RTSP URL with authentication
def create_rtsp_url(base_url, username=None, password=None):
    if username and password:
        if "@" in base_url:
            # URL already has authentication
            return base_url
        protocol = base_url.split("://")[0]
        rest = base_url.split("://")[1]
        return f"{protocol}://{username}:{password}@{rest}"
    return base_url

# Main detection function
def detect_suspicious_activity(frame, model, conf_threshold, heatmap, aois,
                               activity_logs, max_repeated_movements, alarm_triggered,
                               heatmap_history):
    results = model(frame)
    current_activities = defaultdict(int)
    suspicious = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = result.names[int(box.cls[0])]

            if label == "person" and conf > conf_threshold:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                heatmap[y1:y2, x1:x2] += 1

                for idx, (ax1, ay1, ax2, ay2) in enumerate(aois):
                    if x1 >= ax1 and y1 >= ay1 and x2 <= ax2 and y2 <= ay2:
                        current_activities[idx] += 1

    for idx, count in current_activities.items():
        if count > 0:
            activity_logs[idx].append(datetime.datetime.now())

        cutoff = datetime.datetime.now() - datetime.timedelta(seconds=10)
        activity_logs[idx] = [t for t in activity_logs[idx] if t > cutoff]

        if len(activity_logs[idx]) > max_repeated_movements:
            suspicious = True
            if not alarm_triggered[0]:
                alarm_triggered[0] = True
                st.warning(f"\U0001F6A8 *ALARM*: Suspicious movement in Zone {idx+1}!")
                threading.Thread(target=play_alarm, daemon=True).start()

    heatmap_max = int(np.max(heatmap))
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    heatmap_history.append({"time": timestamp, "activity": heatmap_max})

    if heatmap_max > 1:
        suspicious = True
        if not alarm_triggered[0]:
            alarm_triggered[0] = True
            st.warning("\U0001F6A8 *ALARM*: Suspicious activity detected!")
            threading.Thread(target=play_alarm, daemon=True).start()

    elif heatmap_max < 1:
        if alarm_triggered[0]:
            alarm_triggered[0] = False
            stop_alarm()
            st.info("\u2705 No suspicious activity. Alarm turned off.")

    return frame

# Streamlit configuration
st.set_page_config(page_title="Smart Security System", layout="wide")
st.title("\U0001F512 Smart Security System (IP Camera)")

with st.sidebar:
    st.header("\u2699\ufe0f System Settings")
    camera_type = st.radio("Camera Type", ["IP Camera (HTTP)", "RTSP Stream"])
    
    if camera_type == "IP Camera (HTTP)":
        ip_camera_url = st.text_input("Enter IP Camera URL", placeholder="http://192.168.1.10:8080/video")
    else:
        ip_camera_url = st.text_input("Enter RTSP URL", placeholder="rtsp://username:password@192.168.1.10:554/stream")
        rtsp_username = st.text_input("RTSP Username (if not in URL)")
        rtsp_password = st.text_input("RTSP Password (if not in URL)", type="password")
    
    conf_threshold = st.slider("*Detection Confidence Threshold*", 0.0, 1.0, 0.5, 0.01)
    max_reps = st.number_input("*Movement Threshold for Alarm*", 1, 50, 5)
    start_stream = st.button("\U0001F3A5 Start Streaming")

    st.subheader("\U0001F4DC Surveillance Zones (AOI)")
    num_aois = st.number_input("Number of Zones", 0, 5, 1)
    aois = []
    for i in range(num_aois):
        with st.expander(f"Zone {i+1} Settings"):
            x1 = st.slider(f"X1 (Left)", 0, 1920, 200, key=f"x1_{i}")
            y1 = st.slider(f"Y1 (Top)", 0, 1080, 200, key=f"y1_{i}")
            x2 = st.slider(f"X2 (Right)", 0, 1920, 800, key=f"x2_{i}")
            y2 = st.slider(f"Y2 (Bottom)", 0, 1080, 600, key=f"y2_{i}")
            aois.append((x1, y1, x2, y2))

col1, col2 = st.columns(2)
with col1:
    st.subheader("\U0001F3A5 Live Camera Feed")
    camera_placeholder = st.empty()
with col2:
    st.subheader("\U0001F4CA Activity Graph")
    heatmap_placeholder = st.empty()

status_text = st.empty()
status_text.info("\U0001F7E2 *System active*. Waiting for detection...")

if start_stream:
    if camera_type == "RTSP Stream":
        ip_camera_url = create_rtsp_url(ip_camera_url, rtsp_username, rtsp_password)
    
    # Set OpenCV to use FFMPEG with proper timeout
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture()
    
    # Try to open the stream with retries
    max_retries = 3
    retry_delay = 2
    connected = False
    
    for i in range(max_retries):
        try:
            if not cap.open(ip_camera_url):
                raise Exception("Failed to open stream")
            
            # Test if we can read a frame
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to read frame")
            
            connected = True
            break
        except Exception as e:
            if i < max_retries - 1:
                status_text.warning(f"⚠ Connection attempt {i+1}/{max_retries} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                status_text.error(f"❌ Failed to connect to camera after {max_retries} attempts. Error: {str(e)}")
                st.stop()
    
    if connected:
        status_text.success("✅ Successfully connected to camera stream!")
        
        # Get frame size from camera or use default
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        
        heatmap = np.zeros((frame_height, frame_width), dtype=np.uint8)
        activity_logs = defaultdict(list)
        alarm_triggered = [False]
        heatmap_history = deque(maxlen=100)
        
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    status_text.warning("⚠ Frame read error. Trying to reconnect...")
                    time.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(ip_camera_url)
                    continue
                
                # Resize frame if needed
                frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Update heatmap (decay old values)
                heatmap = (heatmap * 0.95).astype(np.uint8)
                
                frame = detect_suspicious_activity(
                    frame, model, conf_threshold, heatmap, aois,
                    activity_logs, max_reps, alarm_triggered, heatmap_history
                )
                
                # Display the frame
                camera_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                       channels="RGB", 
                                       use_container_width=True)
                
                # Update activity graph
                df_heat = pd.DataFrame(heatmap_history)
                if not df_heat.empty:
                    heatmap_placeholder.line_chart(df_heat.set_index("time"))
                
                # Check for stop condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                status_text.error(f"❌ Error during processing: {str(e)}")
                break
        
        cap.release()
        stop_alarm()
        status_text.info("🚫 Stream stopped")