import cv2
import time

def test_cameras():
    print("🔍 Testing available cameras...")
    available_cameras = []
    
    for i in range(5):
        print(f"Testing camera {i}...")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow for Windows
        
        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"✅ Camera {i}: Working - Resolution: {width}x{height}")
                available_cameras.append(i)
            else:
                print(f"❌ Camera {i}: Can't read frame")
        else:
            print(f"❌ Camera {i}: Can't open")
        
        cap.release()
        time.sleep(0.5)
    
    if available_cameras:
        print(f"\n🎉 Found cameras: {available_cameras}")
        
        # Test first available camera
        print(f"\n📸 Testing camera {available_cameras[0]} for 5 seconds...")
        cap = cv2.VideoCapture(available_cameras[0], cv2.CAP_DSHOW)
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                cv2.imshow('Camera Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("❌ Frame read failed")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        fps = frame_count / 5
        print(f"📊 Average FPS: {fps:.2f}")
        
    else:
        print("❌ No cameras found!")
        print("\n🔧 Troubleshooting:")
        print("1. Check if camera is connected")
        print("2. Close other apps using camera (Zoom, Skype, etc.)")
        print("3. Check Windows Camera app")

if __name__ == "__main__":
    test_cameras()
    input("\nPress Enter to exit...")