<div align="center">

# 🛡️ Smart Security System
### Sistem Keamanan Rumah Cerdas Berbasis AI & Computer Vision

[![Python](https://img.shields.io/badge/Python-3.9%2B-27548A?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

<p align="center">
  <b>Solusi keamanan cerdas real-time untuk memantau rumah menggunakan deteksi manusia otomatis (Human Detection), notifikasi audio alarm, serta visualisasi analitik interaktif.</b>
</p>

[Fitur Utama](#-fitur-utama) •
[Arsitektur](#-arsitektur-sistem) •
[Instalasi](#-instalasi--penggunaan) •
[Struktur Direktori](#-struktur-proyek) •
[Konfigurasi Kamera](#-panduan-kamera) •
[Kontribusi](#-kontribusi)

---

</div>

## 📖 Tentang Proyek

**Smart Security System** adalah aplikasi pemantauan keamanan rumah yang menggabungkan kecanggihan **YOLOv8** dan **Streamlit**. Sistem ini dirancang untuk mendeteksi keberadaan orang secara akurat dari berbagai sumber video (HDMI Capture Card, Webcam USB, maupun IP Camera RTSP), kemudian memicu alarm suara saat terdeteksi aktivitas intrusi berkelanjutan.

Dengan tampilan modern beraksen *Navy & Gold*, sistem ini menyajikan metrik pemantauan *real-time*, grafik intensitas deteksi, serta kontrol penuh terhadap sensitivitas AI dan orientasi kamera.

---

## ✨ Fitur Utama

- 🧠 **Deteksi Orang Cerdas (YOLOv8 Nano)**
  - Menggunakan model *deep learning* YOLOv8 (`yolov8n.pt`) yang ringan dan berkecepatan tinggi.
  - Menggambar *bounding box* secara akurat khusus untuk kelas manusia (`person`).
- 📹 **Dukungan Multi-Sumber Kamera**
  - **HDMI Capture Card / Webcam USB**: Auto-detect index kamera aktif dengan dukungan multi-backend (DirectShow di Windows, V4L2 di Linux, AVFoundation di macOS).
  - **IP Camera (RTSP)**: Kompatibel dengan kamera CCTV seperti Hikvision, Dahua, mau pun aplikasi RTSP streamer di smartphone.
- 🚨 **Sistem Alarm Audio Cerdas**
  - Terintegrasi dengan audio mixer `pygame` yang memutar file audio peringatan (`alarm system.mp3`) secara otomatis saat terdeteksi aktivitas berulang.
  - Dilengkapi sistem *cooldown* agar alarm tidak mengalami *glitch* atau *spam*.
- 📊 **Dashboard Analitik Interaktif**
  - Menampilkan live streaming beresolusi optimal dengan latensi minim (buffer flushing otomatis).
  - Indikator status sistem (Normal vs Peringatan Aktif).
  - Statistik real-time: **Total Deteksi** dan **Deteksi per Menit**.
  - Grafik riwayat aktivitas tren deteksi waktu-nyata (*time-series chart*).
- ⚙️ **Kontrol Fleksibel di Tangan Anda**
  - Slider penyesuaian nilai kepercayaan AI (*confidence threshold*).
  - Opsi *Mirror Horizontal* dan *Flip Vertical* untuk orientasi kamera yang dipasang terbalik.
- 🔍 **Tool Diagnostik Mandiri (`test_camera.py`)**
  - Skrip pengujian otomatis untuk memindai port kamera yang tersedia, resolusi yang didukung, dan estimasi FPS sebelum memulai aplikasi utama.

---

## 🏗️ Arsitektur Sistem

```mermaid
flowchart LR
    A[📹 Input Kamera<br/>HDMI / USB / RTSP] --> B[⚙️ OpenCV Frame Grabber<br/>Buffer Flush & Resize]
    B --> C[🧠 YOLOv8 Inference<br/>Deteksi Objek 'Person']
    C --> D{Aktivitas Terus Menerus?}
    D -- Ya --> E[🚨 Pygame Audio Alarm<br/>'alarm system.mp3']
    D -- Tidak --> F[✅ Normal Mode]
    C --> G[📊 Streamlit Dashboard]
    E --> G
    F --> G
    G --> H[🖥️ Live Monitoring & Analytics]
```

---

## 🛠️ Teknologi yang Digunakan

| Komponen | Teknologi | Deskripsi |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Bahasa pemrograman utama |
| **Framework UI** | Streamlit | Dashboard interaktif & responsive |
| **AI / Object Detection** | Ultralytics YOLOv8 | Model deep learning deteksi manusia |
| **Computer Vision** | OpenCV (cv2) | Capture stream, manipulasi frame, buffer handling |
| **Deep Learning Engine** | PyTorch | Komputasi tensor inferensi AI |
| **Audio Engine** | Pygame Mixer | Pemutar efek suara sirine alarm |
| **Data Processing** | NumPy & Pandas | Analitik data dan pembuatan time-series chart |

---

## 🚀 Instalasi & Penggunaan

### 1. Kloning Repositori
```bash
git clone https://github.com/billiampalad/sistem_keamanan_rumah.git
cd sistem_keamanan_rumah
```

### 2. Buat & Aktifkan Virtual Environment (Opsional tapi Direkomendasikan)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

### 4. (Opsional) Uji Coba Kamera Terlebih Dahulu
Untuk memastikan kamera terdeteksi oleh sistem operasi dan mengecek resolusi serta FPS:
```bash
python test_camera.py
```

### 5. Jalankan Aplikasi Utama
```bash
streamlit run index.py
```
Aplikasi akan otomatis terbuka di peramban Anda di alamat: `http://localhost:8501`

---

## 📂 Struktur Proyek

```plaintext
sistem_keamanan_rumah/
│
├── index.py              # Aplikasi utama dashboard Streamlit & logika deteksi
├── test_camera.py        # Skrip utilitas pengujian kamera & deteksi port OpenCV
├── requirements.txt      # Daftar dependensi library Python
├── yolov8n.pt            # Bobot model YOLOv8 pre-trained (Nano version)
├── alarm system.mp3      # Audio sirine yang dibunyikan saat deteksi intrusi
└── README.md             # Dokumentasi proyek
```

---

## 📹 Panduan Kamera

### 1. HDMI Capture Card / USB Webcam
1. Hubungkan Capture Card atau Webcam ke port USB komputer.
2. Buka aplikasi, sistem akan mencari kamera yang aktif secara otomatis.
3. Pilih indeks kamera yang sesuai (misal: `Kamera 0` atau `Kamera 1`).
4. Klik tombol **"🔍 Test Kamera"** untuk memastikan koneksi lancar sebelum memulai monitoring.

### 2. IP Camera (RTSP Stream)
Pilih opsi **IP Camera (RTSP)** pada menu samping dan masukkan tautan RTSP:
- **Format Umum:**
  ```text
  rtsp://192.168.1.100:554/stream
  ```
- **Dengan Autentikasi Username & Password:**
  ```text
  rtsp://username:password@192.168.1.100:554/stream
  ```
- **Kamera Hikvision:**
  ```text
  rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
  ```

---

## 💡 Tips Optimasi Performa

- **Mengurangi Beban CPU**: Resolusi pemrosesan inferensi disesuaikan ke resolusi optimal (`320x240`) sehingga sistem tetap lancar (*smooth*) meskipun dijalankan pada CPU standar / laptop tanpa kartu grafis khusus.
- **Latency / Lagging**: Sistem telah dilengkapi pembersihan buffer otomatis (`cap.grab()`) untuk menjaga agar tampilan video selalu *real-time*.

---

## 🤝 Kontribusi

Kontribusi selalu diterima! Jika Anda memiliki ide untuk fitur baru atau perbaikan:
1. *Fork* repositori ini
2. Buat *branch* fitur baru (`git checkout -b fitur-baru`)
3. *Commit* perubahan Anda (`git commit -m 'Menambahkan fitur baru'`)
4. *Push* ke *branch* tersebut (`git push origin fitur-baru`)
5. Ajukan *Pull Request*

---

## 📄 Lisensi

Didistribusikan di bawah lisensi MIT. Silakan gunakan dan kembangkan sesuai kebutuhan Anda.

<div align="center">
  Dibuat dengan ❤️ oleh <a href="https://github.com/billiampalad"><b>Billiam Palad</b></a>
</div>
