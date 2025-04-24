import cv2
import requests
import pyttsx3
import time
from ultralytics import YOLO

def get_location_info():
    """Get geographical location information using IP address"""
    try:
        # Get public IP address
        ip_response = requests.get('https://api.ipify.org?format=json', timeout=5)
        public_ip = ip_response.json()['ip']

        # Get geographical data
        geo_response = requests.get(f'http://ip-api.com/json/{public_ip}', timeout=5)
        geo_data = geo_response.json()

        if geo_data['status'] == 'success':
            return {
                'city': geo_data.get('city', 'Unknown'),
                'region': geo_data.get('regionName', 'Unknown'),
                'country': geo_data.get('country', 'Unknown'),
                'coordinates': f"{geo_data.get('lat', 0)}, {geo_data.get('lon', 0)}"
            }
        return None
    except Exception as e:
        print(f"Location detection error: {str(e)}")
        return None


def detect_environment_objects():
    """Detect objects using YOLO model with camera"""
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    detected = set()

    try:
        if not cap.isOpened():
            raise RuntimeError("Camera access denied")

        start_time = time.time()
        print("Starting environment scan... (10 seconds)")

        while (time.time() - start_time) < 10:
            ret, frame = cap.read()
            if not ret:
                continue

            # Perform object detection
            results = model(frame)
            current_objects = {model.names[int(box.cls[0])] for box in results[0].boxes}
            detected.update(current_objects)

            # Display live feed with annotations
            annotated_frame = results[0].plot()
            cv2.imshow('Environment Scan', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return list(detected) if detected else ["no objects detected"]

    except Exception as e:
        print(f"Detection error: {str(e)}")
        return ["scan failed"]
    finally:
        cap.release()
        cv2.destroyAllWindows()

def locate_main():
    """Main function to coordinate detection and reporting"""
    tts = pyttsx3.init()
    tts.setProperty('rate', 150)
    tts.setProperty('volume', 1.0)

    # Get location information
    location_data = get_location_info()

    # Get environment objects
    object_list = detect_environment_objects()

    # Generate reports
    location_report = ("Current location: "
                       f"{location_data['city']}, "
                       f"{location_data['region']}, "
                       f"{location_data['country']}") if location_data else "Location detection failed"

    object_report = ("Detected objects: " +
                     ', '.join(object_list) if object_list else "No objects detected")

    # Combine and present results
    full_report = f"""
    Environment Analysis Report:
    - Location: {location_report}
    - Objects: {object_report}
    """
    print(full_report)

    # Voice output
    tts.say("Environment analysis complete. " +
            location_report.replace('-', '') +
            ". " + object_report)
    tts.runAndWait()
    tts.stop()

if __name__ == "__main__":
    locate_main()
