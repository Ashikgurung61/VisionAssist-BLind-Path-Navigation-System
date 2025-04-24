import cv2
import os
import speech_recognition as sr
import pyttsx3

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
tts_engine = pyttsx3.init()

def get_voice_command(prompt=None):
    """Get voice input from the user with optional prompt."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        if prompt:
            tts_engine.say(prompt)
            tts_engine.runAndWait()
            print(prompt)
        
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=20)  
        
    try:
        name = r.recognize_google(audio)
        print(f"You said: {name}")
        return name.strip()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except (sr.RequestError, sr.WaitTimeoutError) as e:
        print(f"Error: {e}")
        return None

def capture_face_image():
    """Capture face image from webcam."""
    cap = cv2.VideoCapture(0)
    face_detected = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_detected = True
        
        cv2.imshow('Face Capture - Press Q to quit', frame)
        
        if face_detected:
            tts_engine.say("Face detected. Capturing in 3 seconds...")
            tts_engine.runAndWait()
            
            for i in range(3, 0, -1):
                print(f"Capturing in {i}...")
                cv2.waitKey(1000)  # Wait 1 second
                
            ret, frame = cap.read()
            if ret:
                cap.release()
                cv2.destroyAllWindows()
                return frame
            else:
                continue
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return None

def save_person_image():
    """Main function to save a person's image with voice command."""
    # Create directory if not exists
    save_dir = 'friends'
    os.makedirs(save_dir, exist_ok=True)
    
    # Step 1: Ask for friend's name
    name = None
    while not name:
        name = get_voice_command("Please say your friend's name clearly.")
        if not name:
            tts_engine.say("I didn't catch that. Please try again.")
            tts_engine.runAndWait()
    
    # Step 2: Capture face
    tts_engine.say(f"Now, please position {name}'s face in front of the camera.")
    tts_engine.runAndWait()
    
    face_image = capture_face_image()
    if face_image is None:
        tts_engine.say("Failed to capture face. Please try again.")
        tts_engine.runAndWait()
        return
    
    # Step 3: Save image
    filename = f"{name.lower().replace(' ', '_')}.jpg"
    save_path = os.path.join(save_dir, filename)
    
    # Avoid overwriting existing files
    counter = 1
    while os.path.exists(save_path):
        filename = f"{name.lower().replace(' ', '_')}_{counter}.jpg"
        save_path = os.path.join(save_dir, filename)
        counter += 1
    
    cv2.imwrite(save_path, face_image)
    tts_engine.say(f"Successfully saved {name}'s image.")
    tts_engine.runAndWait()
    print(f"Image saved at: {save_path}")

if __name__ == "__main__":
    tts_engine.say("Starting friend face capture program.")
    tts_engine.runAndWait()
    save_person_image()