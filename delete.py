import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import time
import os
import requests
import pyttsx3
import speech_recognition as sr
import threading

# Constants
DEPTH_THRESHOLD = 1.5  
DANGER_THRESHOLD = 1.0  
ACTION_SPACE = ["move forward", "adjust left", "adjust right", "slow down", "stop"]
FRIENDS_PATH = "friends"

# Check for face recognition
try:
    from cv2 import face
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

# URLs
YOLO_CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            r = requests.get(url, stream=True)
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False
    return True

class EnhancedPolicy(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        # Simple pre-training for demonstration
        self._pretrain()

    def _pretrain(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for _ in range(1000):
            dummy_input = torch.randn(1, 5)
            target = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])  # Favor adjust left
            output = self(dummy_input)
            loss = nn.MSELoss()(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def forward(self, x):
        return self.net(x)

class VisionSystem:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.last_action = None
        self.prev_gray = None
        
        # Initialize modules
        self._init_depth_estimation()
        self._init_object_detection()
        self._init_face_recognition()
        # self._init_policy_network()
        
        # Voice control
        self.listening = True
        self.shutdown_flag = False
        self._start_voice_listener()
        
        # Camera activation announcement
        self.tts_engine.say("Vision system activated")
        self.tts_engine.runAndWait()

    def _init_depth_estimation(self):
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.midas.eval()
        self.transform = transforms.Compose([
            transforms.Resize(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _init_object_detection(self):
        if not (download_file(YOLO_CFG_URL, "yolov3.cfg") and
                download_file(YOLO_WEIGHTS_URL, "yolov3.weights") and
                download_file(COCO_NAMES_URL, "coco.names")):
            raise RuntimeError("Failed to download YOLO files")
            
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]

    def _init_face_recognition(self):
        self.face_recognizer = None
        if FACE_RECOGNITION_AVAILABLE:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self._load_known_faces()
            except Exception as e:
                print(f"Face recognition init failed: {e}")

    def _load_known_faces(self):
        self.face_labels = {}
        label_id = 0
        faces = []
        labels = []

        if os.path.exists(FRIENDS_PATH):
            for root, _, files in os.walk(FRIENDS_PATH):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(root, file)
                        label = os.path.basename(root)
                        
                        if label not in self.face_labels:
                            self.face_labels[label_id] = label
                            label_id += 1
                            
                        img = Image.open(path).convert('L')
                        img_array = np.array(img, 'uint8')
                        faces.append(img_array)
                        labels.append(label_id-1)

            if faces:
                self.recognizer.train(faces, np.array(labels))

    def _start_voice_listener(self):
        def listener():
            r = sr.Recognizer()
            with sr.Microphone() as source:
                while self.listening:
                    try:
                        audio = r.listen(source, timeout=1)
                        text = r.recognize_google(audio).lower()
                        if any(cmd in text for cmd in ["shutdown", "stop", "exit"]):
                            self.shutdown_flag = True
                    except Exception as e:
                        pass
                        
        threading.Thread(target=listener, daemon=True).start()

    def _get_depth_analysis(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            depth_map = self.midas(input_tensor).squeeze().cpu().numpy()
        
        # Calculate depth regions
        h, w = depth_map.shape
        left_depth = depth_map[:, :w//3].mean()
        center_depth = depth_map[:, w//3:2*w//3].mean()
        right_depth = depth_map[:, 2*w//3:].mean()
        
        return depth_map, (left_depth, center_depth, right_depth)

    def _detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        objects = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], 
                                                   frame.shape[1], frame.shape[0]])
                    (x, y, w, h) = box.astype("int")
                    objects.append({
                        "class": self.classes[class_id],
                        "confidence": float(confidence),
                        "box": (x, y, w, h)
                    })
        return objects

    def _calculate_optical_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = None
        
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 
                                               0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_gray = gray
        return flow

    def analyze_frame(self, frame):
        # Depth analysis
        depth_map, depth_regions = self._get_depth_analysis(frame)
        
        # Object detection
        objects = self._detect_objects(frame)
        
        # Optical flow
        flow = self._calculate_optical_flow(frame)
        flow_mag = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)) if flow is not None else 0
        
        # Face recognition
        faces = []
        if self.face_recognizer:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_boxes = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x,y,w,h) in face_boxes:
                label_id, conf = self.recognizer.predict(gray[y:y+h,x:x+w])
                if conf < 70:
                    faces.append(self.face_labels.get(label_id, "Unknown"))

        # Create state vector
        state = np.array([
            depth_regions[0],    # Left depth
            depth_regions[1],    # Center depth
            depth_regions[2],    # Right depth
            len(objects),        # Object count
            flow_mag,            # Movement magnitude
            len(faces)          # Recognized faces
        ], dtype=np.float32)

        return {
            "state": state,
            "depth_map": depth_map,
            "objects": objects,
            "faces": faces,
            "optical_flow": flow
        }

class NavigationController:
    def __init__(self):
        self.policy = EnhancedPolicy()
        self.last_action = None
        self.action_history = []
        
    def decide_action(self, state):
        # Neural network decision
        state_tensor = torch.FloatTensor(state[:5])  # Use first 5 features
        with torch.no_grad():
            q_values = self.policy(state_tensor)
            action_idx = torch.argmax(q_values).item()
            
        # Rule-based overrides
        action = ACTION_SPACE[action_idx]
        
        # Depth-based corrections
        left_depth, center_depth, right_depth = state[0], state[1], state[2]
        if center_depth < DANGER_THRESHOLD:
            if left_depth > right_depth:
                action = "adjust left"
            else:
                action = "adjust right"
                
        # Friend recognition override
        if state[5] > 0:
            action = "slow down"
            
        # Emergency stop condition
        if center_depth < 0.5*DANGER_THRESHOLD:
            action = "stop"
            
        self.action_history.append(action)
        return action

def main():
    vision = VisionSystem()
    controller = NavigationController()
    cap = cv2.VideoCapture(0)
    
    # Camera setup
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while not vision.shutdown_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            analysis = vision.analyze_frame(frame)
            action = controller.decide_action(analysis["state"])
            
            # Announce action changes
            if action != vision.last_action:
                vision.tts_engine.say(action)
                vision.tts_engine.runAndWait()
                vision.last_action = action
                
            # Visualization
            vis_frame = frame.copy()
            # self._draw_overlays(vis_frame, analysis, action)
            cv2.imshow("Navigation System", vis_frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
                
    finally:
        vision.listening = False
        cap.release()
        cv2.destroyAllWindows()
        vision.tts_engine.say("System shutdown")
        vision.tts_engine.runAndWait()

if __name__ == "__main__":
    main()