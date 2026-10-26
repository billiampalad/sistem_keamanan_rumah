# 🛡️ Smart Home Security System Berbasis YOLOv8

> **Sistem Keamanan Rumah Cerdas Berbasis Machine Learning dengan Deteksi Objek Real-Time dan Antarmuka Web Interaktif Streamlit**

---

## 📌 Ringkasan Projek

**Smart Home Security System** adalah aplikasi pemantauan keamanan rumah berbasis *Computer Vision* dan *Machine Learning*. Sistem ini dirancang untuk mengatasi keterbatasan sistem keamanan konvensional melalui deteksi objek manusia (*person detection*) secara *real-time* menggunakan algoritma **YOLOv8** yang terintegrasi dengan antarmuka web **Streamlit**.

Sistem mampu mengintegrasikan berbagai sumber kamera, seperti **IP Camera RTSP** (berbasis protokol ONVIF/V380) dan **CCTV Analog/HD** (melalui HDMI Capture Card), serta memberikan peringatan alarm suara secara otomatis ketika terdeteksi aktivitas mencurigakan.

---

## 👥 Tim Penyusun

* **Program Studi:** D4 Teknik Informatika, Jurusan Teknik Elektro, Politeknik Negeri Manado (2025)
* **Dosen Pengampu:** Harson Kapoh, ST., MT
* **Kelompok 1 (Kelas 6TI5):**
  * Iqbal Jabar Marjan (NIM: 22024102)
  * Delon Gilbert Yeheskiel Temo (NIM: 22024095)
  * Clynten Billiam Palad (NIM: 22024108)
  * Israel Jody Mamusung (NIM: 22024110)
  * Wilsaylin Tahupia (NIM: 22024113)

---

## ✨ Fitur Utama

1. **Real-Time Human Detection (YOLOv8n)**  
   Mendeteksi keberadaan manusia (*class 0: person*) secara cepat dan akurat menggunakan model *lightweight* `yolov8n.pt`.
2. **Dukungan Multi-Kamera**  
   Mendukung kamera USB/WebCam, CCTV melalui HDMI Capture Card (USB 3.0 / DirectShow), dan IP Camera via URL RTSP.
3. **Dashboard Web Interaktif (Streamlit)**  
   Menampilkan *Live Camera Feed* dengan overlay *bounding box*, indikator status sistem (*Standby* / *PERINGATAN AKTIF*), serta grafik histori aktivitas.
4. **Sistem Peringatan Alarm Suara Otomatis**  
   Memicu suara sirine/alarm (`alarm system.mp3` via Pygame) jika frekuensi deteksi melebihi ambang batas (contoh: 20 deteksi dalam 10 detik).
5. **Pengaturan AI & Tampilan Fleksibel**  
   * Slider sensitivitas deteksi (*confidence threshold* 0.10–0.90).
   * Fitur *Mirror Horizontal* dan *Flip Vertical* untuk penyesuaian orientasi kamera.
6. **Optimasi Performa**  
   Penerapan *frame resizing* (320x240 untuk inferensi), *frame skipping*, *buffer flushing*, dan *garbage collection* periodik untuk menjaga *framerate* stabil (20–30 FPS) pada *hardware* terbatas.
7. **Kemudahan Deployment Remote**  
   Mendukung *tunneling* publik menggunakan Ngrok untuk akses *dashboard* dari luar jaringan lokal.

---

## 🏗️ Arsitektur Sistem

```text
[ Kamera / CCTV / RTSP ]
          │
          ▼
[ Video Ingestion & Preprocessing (OpenCV) ]
  • Frame Resize (320x240) & Normalisasi
          │
          ▼
[ Inference Engine (YOLOv8n / PyTorch) ]
  • Filter Kelas Person (Class ID: 0)
  • Coordinate Scaling ke Resolusi Asli (640x480)
          │
          ▼
[ Logic & Notification Engine ]
  • Evaluasi Detection History & Threshold
  • Trigger Audio Alarm (Pygame)
          │
          ▼
[ Streamlit Web Interface ]
  • Live Feed Display + Bounding Boxes
  • Real-time Stats & Activity Line Chart
```

---

## 📁 Struktur Direktori

```text
smart_home/
├── index.py              # File utama aplikasi Streamlit dan logika deteksi
├── security_config.json  # File konfigurasi eksternal (camera, alerts, detection settings)
├── yolov8n.pt            # Weight model pretrained YOLOv8-nano
├── alarm system.mp3      # Efek suara alarm peringatan
└── requirements.txt      # Daftar dependensi modul Python
```

---

## 💻 Kebutuhan Sistem

### Perangkat Keras (Hardware)
* **Prosesor:** Intel Core i5 / AMD Ryzen 5 atau setara.
* **RAM:** Minimum 8 GB.
* **Penyimpanan:** SSD 256 GB.
* **Kamera:** IP Camera ONVIF/V380 (RTSP) atau CCTV + HDMI Capture Card (USB 3.0).
* **Router/Modem:** Router TP-Link / Wi-Fi Dual-Band untuk koneksi IP Camera.

### Perangkat Lunak (Software)
* **Sistem Operasi:** Windows 10/11, Linux, atau macOS.
* **Bahasa Pemrograman:** Python 3.9 – 3.11.
* **Code Editor:** Visual Studio Code.
* **Utility Deployment:** Ngrok.

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Persiapan Environment
Pastikan Python telah terinstal dan terdaftar di PATH. Buka terminal/Command Prompt lalu jalankan:

```bash
# Masuk ke direktori projek
cd smart_home

# Instal dependensi yang diperlukan
pip install -r requirements.txt
```

Atau instal pustaka secara manual:
```bash
pip install opencv-python-headless streamlit numpy pandas torch torchvision ultralytics pygame
```

### 2. Jalankan Aplikasi Lokal
Untuk menjalankan dashboard Streamlit secara lokal:

```bash
streamlit run index.py
```

Setelah perintah dijalankan, antarmuka web akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

### 3. Petunjuk Penggunaan Dashboard
1. **Pilih Jenis Kamera** di Panel Kontrol Kiri (*Sidebar*):
   * **HDMI Capture Card:** Pilih indeks kamera yang terdeteksi lalu klik *Test Kamera*.
   * **IP Camera (RTSP):** Masukkan URL stream RTSP (contoh: `rtsp://admin:password@192.168.1.100:554/stream`).
2. Atur **Sensitivitas Deteksi** (*Confidence Threshold*) sesuai kondisi pencahayaan.
3. Klik tombol **▶ Mulai** untuk memulai *monitoring real-time*.
4. Jika objek manusia terdeteksi, *bounding box* hijau akan muncul, counter statistik diperbarui, dan alarm akan berbunyi bila aktivitas tinggi.
5. Klik **⏹ Stop** untuk menghentikan pemantauan.

---

## 🌐 Deployment Remote Menggunakan Ngrok

Agar *dashboard* dapat diakses secara *remote* dari jaringan eksternal:

1. Unduh dan ekstrak **Ngrok** dari situs resminya.
2. Buka **Dua Terminal**:
   * **Terminal 1 (Aplikasi):**
     ```bash
     streamlit run index.py --server.port 8501
     ```
   * **Terminal 2 (Ngrok Tunneling):**
     ```bash
     ngrok http 8501
     ```
3. Salin URL publik yang dihasilkan Ngrok pada baris `Forwarding` untuk mengakses aplikasi dari mana saja.

---

## 📜 Lisensi & Pengakuan

Projek ini dikembangkan sebagai Projek Akhir Mata Kuliah **Workshop Pembelajaran Mesin** di **Politeknik Negeri Manado**. Model YOLOv8 dikembangkan oleh *Ultralytics*.
