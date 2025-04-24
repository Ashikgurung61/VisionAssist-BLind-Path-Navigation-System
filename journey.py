# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from PIL import Image
# import time
# import os
# import requests
# import pyttsx3
# import speech_recognition as sr
# import threading

# # Constants
# DEPTH_THRESHOLD = 1.5  
# DANGER_THRESHOLD = 1.0  
# ACTION_SPACE = ["move forward", "adjust left", "adjust right", "slow down", "stop"]
# FRIENDS_PATH = "friends"

# # Check for face recognition availability
# try:
#     from cv2 import face
#     FACE_RECOGNITION_AVAILABLE = True
# except (ImportError, AttributeError):
#     FACE_RECOGNITION_AVAILABLE = False
#     print("Warning: OpenCV face recognition module not available - install opencv-contrib-python")

# # YOLO model URLs
# YOLO_CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
# YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
# COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# def download_file(url, filename):
#     if not os.path.exists(filename):
#         print(f"Downloading {filename}...")
#         try:
#             r = requests.get(url, stream=True)
#             with open(filename, 'wb') as f:
#                 for chunk in r.iter_content(chunk_size=1024):
#                     if chunk:
#                         f.write(chunk)
#         except Exception as e:
#             print(f"Error downloading {filename}: {e}")
#             return False
#     return os.path.exists(filename)

# class SafetyPolicy(nn.Module):
#     def __init__(self, input_size=3, hidden_size=128, output_size=5):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, output_size)
#         )

#     def forward(self, x):
#         return self.net(x)

# class FaceRecognizer:
#     def __init__(self):
#         if not FACE_RECOGNITION_AVAILABLE:
#             raise RuntimeError("Face recognition not available")
        
#         self.recognizer = cv2.face.LBPHFaceRecognizer_create()
#         self.face_cascade = cv2.CascadeClassifier(
#             cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.friend_labels = {}
#         self._load_friends()
        
#     def _load_friends(self):
#         print("⏳ Loading friend faces...")
#         faces = []
#         labels = []
#         label_ids = {}
#         current_id = 0

#         if not os.path.exists(FRIENDS_PATH):
#             os.makedirs(FRIENDS_PATH)
#             print(f"Created friends directory at {FRIENDS_PATH}")
#             return

#         for root, _, files in os.walk(FRIENDS_PATH):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     path = os.path.join(root, file)
#                     label = os.path.basename(root)

#                     if label not in label_ids:
#                         label_ids[label] = current_id
#                         current_id += 1

#                     try:
#                         pil_image = Image.open(path).convert('L')
#                         image_array = np.array(pil_image, 'uint8')
#                         faces.append(image_array)
#                         labels.append(label_ids[label])
#                     except Exception as e:
#                         print(f"Error loading {path}: {str(e)}")

#         if len(faces) > 0:
#             self.recognizer.train(faces, np.array(labels))
#             self.friend_labels = {v: k for k, v in label_ids.items()}
#             print(f"✅ Trained on {len(label_ids)} friends")
#         else:
#             print("⚠️ No valid friend images found")

#     def recognize_faces(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = self.face_cascade.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(30, 30)
#         )
#         recognized = []
#         for (x, y, w, h) in faces:
#             roi = gray[y:y+h, x:x+w]
#             label_id, confidence = self.recognizer.predict(roi)
            
#             if confidence < 70:  # Lower confidence threshold means more strict
#                 recognized.append({
#                     "name": self.friend_labels.get(label_id, "Unknown"),
#                     "box": (x, y, w, h),
#                     "confidence": confidence
#                 })
#         return recognized

# class PathAnalyzer:
#     def __init__(self):
#         self.classes = []
#         self._load_yolo()

#         # Initialize TTS engine
#         self.tts_engine = pyttsx3.init()
#         self.tts_engine.setProperty('rate', 150)
#         self.last_action = None

#         # Initialize face recognition (if available)
#         self.face_recognizer = None
#         if FACE_RECOGNITION_AVAILABLE:
#             try:
#                 self.face_recognizer = FaceRecognizer()
#             except Exception as e:
#                 print(f"Could not initialize face recognition: {e}")

#         # Initialize depth estimation
#         self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
#         self.midas.eval()
#         self.transform = transforms.Compose([
#             transforms.Resize(384),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

#         # Initialize policy network
#         self.policy_net = SafetyPolicy()
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

#         # Voice control
#         self.listening = True
#         self.shutdown_flag = False
#         self.start_voice_listener()

#     def _load_yolo(self):
#         if (download_file(YOLO_CFG_URL, "yolov3.cfg") and
#                 download_file(YOLO_WEIGHTS_URL, "yolov3.weights") and
#                 download_file(COCO_NAMES_URL, "coco.names")):
#             self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#             with open("coco.names", "r") as f:
#                 self.classes = [line.strip() for line in f.readlines()]
#             layer_names = self.net.getLayerNames()
#             layer_ids = self.net.getUnconnectedOutLayers()
#             if isinstance(layer_ids[0], int):  # OpenCV >= 4.5.5
#                 self.output_layers = [layer_names[i - 1] for i in layer_ids]
#             else:
#                 self.output_layers = [layer_names[i - 1] for i in layer_ids]
#             print("✅ YOLOv3 loaded successfully")
#         else:
#             raise RuntimeError("Failed to download YOLO files")

#     def start_voice_listener(self):
#         def listen_thread():
#             r = sr.Recognizer()
#             with sr.Microphone() as source:
#                 while self.listening:
#                     try:
#                         audio = r.listen(source, timeout=1)
#                         text = r.recognize_google(audio).lower()
#                         print(f"Voice command: {text}")
#                         if any(cmd in text for cmd in ["close", "exit", "shutdown"]):
#                             self.shutdown_flag = True
#                             self.listening = False
#                     except sr.WaitTimeoutError:
#                         continue
#                     except Exception as e:
#                         print(f"Voice recognition error: {e}")
                        
#         threading.Thread(target=listen_thread, daemon=True).start()

#     def announce_action(self, action):
#         if action != self.last_action:
#             self.tts_engine.say(action)
#             self.tts_engine.runAndWait()
#             self.last_action = action

#     def detect_objects(self, frame, depth_map=None):
#         frame = cv2.resize(frame, (640, 480))
#         height, width = frame.shape[:2]

#         blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#         self.net.setInput(blob)
#         outs = self.net.forward(self.output_layers)

#         objects = []
#         for out in outs:
#             for detection in out:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
#                 if confidence > 0.5:
#                     center_x = int(detection[0] * width)
#                     center_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)

#                     distance = None
#                     if depth_map is not None:
#                         region = depth_map[y:y + h, x:x + w]
#                         if region.size > 0:
#                             distance = float(np.mean(region))
#                     objects.append({
#                         "class_id": class_id,
#                         "class_name": self.classes[class_id],
#                         "confidence": float(confidence),
#                         "box": [x, y, w, h],
#                         "distance": distance
#                     })
#         return objects

#     def get_state(self, frame):
#         frame = cv2.resize(frame, (640, 480))
#         input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         input_tensor = self.transform(input_image).unsqueeze(0)

#         with torch.no_grad():
#             prediction = self.midas(input_tensor)
#             depth_map = prediction.squeeze().cpu().numpy()

#         h, w = depth_map.shape
#         center_region = depth_map[h // 3:h * 2 // 3, w // 3:w * 2 // 3]
#         avg_depth = np.mean(center_region)

#         objects = self.detect_objects(frame, depth_map)
#         num_obstacles = len(objects)

#         danger_zone = (depth_map < DANGER_THRESHOLD).sum()
#         danger_ratio = danger_zone / (h * w)

#         # Face recognition (if available)
#         faces = []
#         friend_present = 0.0
#         if self.face_recognizer is not None:
#             try:
#                 faces = self.face_recognizer.recognize_faces(frame)
#                 friend_present = 1.0 if len(faces) > 0 else 0.0
#             except Exception as e:
#                 print(f"Face recognition error: {e}")

#         enhanced_state = np.array([
#             avg_depth,
#             num_obstacles,
#             danger_ratio,
#             friend_present
#         ], dtype=np.float32)

#         return enhanced_state, depth_map, objects, faces

#     def calculate_reward(self, prev_state, action, new_state):
#         depth_reward = new_state[0] - prev_state[0]
#         obstacle_penalty = -new_state[1] * 0.1
#         danger_penalty = -new_state[2] * 5
#         friend_reward = new_state[3] * 2
#         return depth_reward + obstacle_penalty + danger_penalty + friend_reward

#     def get_action(self, state):
#         with torch.no_grad():
#             state_tensor = torch.tensor(state[:3], dtype=torch.float32).unsqueeze(0)
#             q_values = self.policy_net(state_tensor)
#             action_idx = torch.argmax(q_values).item()
            
#             # Special handling when friend is detected
#             if state[3] > 0.5:  # Friend present
#                 return "slow down"
            
#             return ACTION_SPACE[action_idx]

#     def visualize(self, frame, state, action, objects, faces):
#         frame = cv2.resize(frame, (640, 480))
        
#         # System status
#         cv2.putText(frame, f"Depth: {state[0]:.2f} m", (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#         cv2.putText(frame, f"Obstacles: {int(state[1])}", (10, 60), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#         cv2.putText(frame, f"Danger: {state[2]:.2f}", (10, 90), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#         cv2.putText(frame, f"Action: {action}", (10, 120), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
#         # Friend status
#         if state[3] > 0.5:
#             cv2.putText(frame, "Friend detected!", (10, 150), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

#         # Object boxes
#         for obj in objects:
#             x, y, w, h = obj["box"]
#             label = f"{obj['class_name']} ({obj['distance']:.2f}m)" if obj["distance"] else obj["class_name"]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 255), 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 2)

#         # Face boxes (if available)
#         for face in faces:
#             x, y, w, h = face["box"]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             label = f"{face['name']} ({100-face['confidence']:.1f}%)"
#             cv2.putText(frame, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         return frame

# def journey_main():
#     analyzer = PathAnalyzer()
#     cap = cv2.VideoCapture(0)

#     # Camera setup
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     cap.set(cv2.CAP_PROP_FPS, 30)

#     # Create window
#     cv2.namedWindow("Navigation System", cv2.WINDOW_NORMAL)

#     prev_state = None
#     last_action_time = 0
#     action_cooldown = 1.0
#     last_friend_time = 0
#     friend_cooldown = 5.0

#     try:
#         while not analyzer.shutdown_flag:
#             start_time = time.time()
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Process frame
#             state, depth_map, objects, faces = analyzer.get_state(frame)
#             action = analyzer.get_action(state)

#             # Announce action
#             current_time = time.time()
#             if action != analyzer.last_action and (current_time - last_action_time) > action_cooldown:
#                 analyzer.announce_action(action)
#                 last_action_time = current_time

#             # Announce friend
#             if len(faces) > 0 and (current_time - last_friend_time) > friend_cooldown:
#                 friend = max(faces, key=lambda f: f["box"][2]*f["box"][3])
#                 analyzer.announce_action(f"{friend['name']} detected")
#                 last_friend_time = current_time

#             # Visualize
#             frame = analyzer.visualize(frame, state, action, objects, faces)
#             cv2.imshow("Navigation System", frame)

#             # FPS counter
#             fps = 1.0 / (time.time() - start_time)
#             print(f"FPS: {fps:.1f} | Action: {action} | Friends: {len(faces)}")

#             # Exit conditions
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         analyzer.listening = False
#         cap.release()
#         cv2.destroyAllWindows()
#         analyzer.tts_engine.say("System shutting down")
#         analyzer.tts_engine.runAndWait()
#         print("System closed")

# if __name__ == "__main__":
#     journey_main()



import cv2
from ultralytics import YOLO
import pyttsx3
import time
import face_recognition
import numpy as np      
import os

# Initialize face recognition data
known_face_encodings = []
known_face_names = []
friends_folder = 'friends'

# Define obstacle classes (YOLO COCO classes)
OBSTACLE_CLASSES = {
    'person', 'chair', 'bench', 'backpack', 'handbag', 'suitcase',
    'bottle', 'cup', 'book', 'vase', 'car', 'truck', 'bicycle',
    'motorcycle', 'stop sign', 'parking meter', 'fire hydrant'
}

# Load known faces
for filename in os.listdir(friends_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(friends_folder, filename)
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])

def get_obstacle_direction(frame_width, boxes):
    # Divide frame into three vertical regions
    left_boundary = frame_width // 3
    right_boundary = 2 * frame_width // 3
    
    # Initialize obstacle counters
    left_count = 0
    center_count = 0
    right_count = 0

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        
        if center_x < left_boundary:
            left_count += 1
        elif center_x < right_boundary:
            center_count += 1
        else:
            right_count += 1

    # Determine safest direction
    if center_count > 0:
        if left_count < right_count:
            return "Move left"
        elif right_count < left_count:
            return "Move right"
        else:
            return "Stop: Obstacles ahead"
    return None

def start():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) 
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    last_spoken = {}
    direction_cooldown = 5  # seconds
    last_direction_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Process object detection
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Process face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_names = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    recognized_names.append(name)

            # Draw face annotations
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(annotated_frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        current_time = time.time()
        obstacle_boxes = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            obj_name = model.names[class_id]
            if obj_name in OBSTACLE_CLASSES:
                obstacle_boxes.append(box)

        # Get navigation suggestion
        direction = get_obstacle_direction(frame_width, obstacle_boxes)
        
        # Announce navigation suggestion
        if direction and (current_time - last_direction_time) > direction_cooldown:
            engine.say(direction)
            engine.runAndWait()
            last_direction_time = current_time
            # Draw direction text
            cv2.putText(annotated_frame, direction, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Original detection announcements
        current_objects = [model.names[int(box.cls)] for box in results[0].boxes]
        unique_objects = list(set(current_objects))
        
        # Prioritize face names over "person" detection
        final_names = []
        if "person" in unique_objects and recognized_names:
            unique_objects.remove("person")
            final_names = recognized_names
        else:
            final_names = unique_objects

        # Announce detections with cooldown
        for name in final_names:
            last_time = last_spoken.get(name, 0)
            if current_time - last_time > direction_cooldown:
                engine.say(f"{name} detected")
                engine.runAndWait()
                last_spoken[name] = current_time

        cv2.imshow("Navigation Assistant", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start()