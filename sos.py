# import smtplib
# from email.message import EmailMessage
# import cv2
# import socket
# import requests
# import time
# from datetime import datetime
# import os

# def get_location():
#     """Get detailed location information using multiple fallback services"""
#     location_data = {
#         'city': 'Unknown',
#         'region': 'Unknown',
#         'country': 'Unknown',
#         'coordinates': '0, 0',
#         'ip': 'Unknown'
#     }

#     try:
#         # First try ip-api.com (free tier)
#         try:
#             response = requests.get('http://ip-api.com/json/', timeout=5)
#             if response.status_code == 200:
#                 data = response.json()
#                 if data['status'] == 'success':
#                     location_data.update({
#                         'city': data.get('city', 'Unknown'),
#                         'region': data.get('regionName', 'Unknown'),
#                         'country': data.get('country', 'Unknown'),
#                         'coordinates': f"{data.get('lat', 0)}, {data.get('lon', 0)}",
#                         'ip': data.get('query', 'Unknown')
#                     })
#                     return location_data
#         except Exception as e:
#             print(f"ip-api.com service failed: {str(e)}")

#         # Fallback to ipapi.co
#         try:
#             ip_response = requests.get('https://api.ipify.org?format=json', timeout=5)
#             public_ip = ip_response.json()['ip']
#             geo_response = requests.get(f'https://ipapi.co/{public_ip}/json/', timeout=5)

#             if geo_response.status_code == 200:
#                 data = geo_response.json()
#                 location_data.update({
#                     'city': data.get('city', 'Unknown'),
#                     'region': data.get('region', 'Unknown'),
#                     'country': data.get('country_name', 'Unknown'),
#                     'coordinates': f"{data.get('latitude', 0)}, {data.get('longitude', 0)}",
#                     'ip': public_ip
#                 })
#                 return location_data
#         except Exception as e:
#             print(f"ipapi.co service failed: {str(e)}")

#         # Final fallback to freegeoip.app
#         try:
#             response = requests.get('https://freegeoip.app/json/', timeout=5)
#             if response.status_code == 200:
#                 data = response.json()
#                 location_data.update({
#                     'city': data.get('city', 'Unknown'),
#                     'region': data.get('region_name', 'Unknown'),
#                     'country': data.get('country_name', 'Unknown'),
#                     'coordinates': f"{data.get('latitude', 0)}, {data.get('longitude', 0)}",
#                     'ip': data.get('ip', 'Unknown')
#                 })
#                 return location_data
#         except Exception as e:
#             print(f"freegeoip.app service failed: {str(e)}")

#     except Exception as e:
#         print(f"All location services failed: {str(e)}")

#     # Return default values if all services fail
#     return location_data

# def capture_video(duration=5, output_file='emergency_video.mp4'):
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open video device")
#         return None

#     # Get video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps <= 0:
#         fps = 30  # default if cannot get fps

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

#     start_time = time.time()

#     print(f"Recording for {duration} seconds...")
#     while (time.time() - start_time) < duration:
#         ret, frame = cap.read()
#         if ret:
#             # Add timestamp to frame
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             cv2.putText(frame, timestamp, (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             out.write(frame)
#         else:
#             print("Error capturing frame")
#             break

#     # Release everything
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     return output_file


# def email_alert(subject, body, to, attachment_path=None):
#     msg = EmailMessage()
#     msg.set_content(body)
#     msg['subject'] = subject
#     msg['to'] = to

#     user = "ashikgurunge@gmail.com"
#     msg['from'] = user
#     password = "yuzj lqqa ffwj isqn"  # Consider using environment variables for security

#     if attachment_path:
#         with open(attachment_path, 'rb') as f:
#             file_data = f.read()
#             file_name = os.path.basename(attachment_path)
#         msg.add_attachment(file_data, maintype='video', subtype='mp4', filename=file_name)

#     try:
#         server = smtplib.SMTP("smtp.gmail.com", 587)
#         server.starttls()
#         server.login(user, password)
#         server.send_message(msg)
#         print("Email sent successfully")
#     except Exception as e:
#         print(f"Error sending email: {e}")
#     finally:
#         server.quit()

# def sos_main():
#     # Get location information
#     location = get_location()
#     print(f"Current location: {location}")

#     # Capture video
#     video_file = capture_video()

#     if video_file:
#         # Prepare email content
#         subject = "EMERGENCY: Immediate Attention Required"
#         body = f"""URGENT: I NEED HELP!

# My current location is: {location}
# Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# I'm in danger. Please check the attached video of my surroundings.

# This is an automated emergency message."""

#         recipient = "ashikgrg61@gmail.com"

#         # Send email with video attachment
#         email_alert(subject, body, recipient, video_file)

#         # Clean up - remove the video file after sending
#         try:
#             os.remove(video_file)
#             print("Temporary video file removed")
#         except Exception as e:
#             print(f"Error removing video file: {e}")
#     else:
#         print("Failed to capture video. Sending email without attachment.")
#         subject = "EMERGENCY: Immediate Attention Required (No Video)"
#         body = f"""URGENT: I NEED HELP!

# My current location is: {location}
# Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# I'm in danger but couldn't capture video.

# This is an automated emergency message."""

#         recipient = "ashikgrg61@gmail.com"
#         email_alert(subject, body, recipient)


# if __name__ == "__main__":
#     sos_main()




#-----------------------------

import smtplib
from email.message import EmailMessage
import cv2
import socket
import requests
import time
from datetime import datetime
import os
import pyttsx3

def get_location():
    """Get detailed location information using multiple fallback services"""
    location_data = {
        'city': 'Unknown',
        'region': 'Unknown',
        'country': 'Unknown',
        'coordinates': '0, 0',
        'ip': 'Unknown'
    }

    try:
        # First try ip-api.com (free tier)
        try:
            response = requests.get('http://ip-api.com/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    location_data.update({
                        'city': data.get('city', 'Unknown'),
                        'region': data.get('regionName', 'Unknown'),
                        'country': data.get('country', 'Unknown'),
                        'coordinates': f"{data.get('lat', 0)}, {data.get('lon', 0)}",
                        'ip': data.get('query', 'Unknown')
                    })
                    return location_data
        except Exception as e:
            print(f"ip-api.com service failed: {str(e)}")

        # Fallback to ipapi.co
        try:
            ip_response = requests.get('https://api.ipify.org?format=json', timeout=5)
            public_ip = ip_response.json()['ip']
            geo_response = requests.get(f'https://ipapi.co/{public_ip}/json/', timeout=5)

            if geo_response.status_code == 200:
                data = geo_response.json()
                location_data.update({
                    'city': data.get('city', 'Unknown'),
                    'region': data.get('region', 'Unknown'),
                    'country': data.get('country_name', 'Unknown'),
                    'coordinates': f"{data.get('latitude', 0)}, {data.get('longitude', 0)}",
                    'ip': public_ip
                })
                return location_data
        except Exception as e:
            print(f"ipapi.co service failed: {str(e)}")

        # Final fallback to freegeoip.app
        try:
            response = requests.get('https://freegeoip.app/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                location_data.update({
                    'city': data.get('city', 'Unknown'),
                    'region': data.get('region_name', 'Unknown'),
                    'country': data.get('country_name', 'Unknown'),
                    'coordinates': f"{data.get('latitude', 0)}, {data.get('longitude', 0)}",
                    'ip': data.get('ip', 'Unknown')
                })
                return location_data
        except Exception as e:
            print(f"freegeoip.app service failed: {str(e)}")

    except Exception as e:
        print(f"All location services failed: {str(e)}")

    # Return default values if all services fail
    return location_data

def capture_video(duration=5, output_file='emergency_video.mp4'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device")
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # default if cannot get fps

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()

    print(f"Recording for {duration} seconds...")
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            # Add timestamp to frame
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(frame)
        else:
            print("Error capturing frame")
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_file

def email_alert(subject, body, to, attachment_path=None):
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to

    user = "ashikgurunge@gmail.com"
    msg['from'] = user
    password = "yuzj lqqa ffwj isqn"  # Consider using environment variables for security

    if attachment_path:
        with open(attachment_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(attachment_path)
        msg.add_attachment(file_data, maintype='video', subtype='mp4', filename=file_name)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
        print("Email sent successfully")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
    finally:
        server.quit()

def sos_main():
    # Initialize TTS engine
    tts = pyttsx3.init()
    tts.setProperty('rate', 150)
    
    # Phase 1: Location Acquisition
    tts.say("Initializing emergency protocol. Getting your location.")
    tts.runAndWait()
    location = get_location()
    location_str = f"{location['city']}, {location['region']}, {location['country']}"
    tts.say(f"Location identified as {location_str}")
    tts.runAndWait()

    # Phase 2: Environment Recording
    tts.say("Recording surroundings for 5 seconds. Please stay still.")
    tts.runAndWait()
    video_file = capture_video()
    
    if video_file:
        tts.say("Environment recorded successfully. Preparing emergency alert.")
        tts.runAndWait()
    else:
        tts.say("Warning: Could not record video. Proceeding without visual data.")
        tts.runAndWait()

    # Phase 3: Emergency Dispatch
    tts.say("Sending emergency alert to contacts. Please wait.")
    tts.runAndWait()
    
    subject = "EMERGENCY ALERT: Immediate Assistance Required"
    body = f"""URGENT - DISTRESS SIGNAL ACTIVATED

User Location: {location_str}
Coordinates: {location['coordinates']}
IP Address: {location['ip']}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This is an automated emergency alert. The sender may be in danger."""

    recipient = "ashikgrg61@gmail.com"
    success = email_alert(subject, body, recipient, video_file if video_file else None)

    # Phase 4: Status Report
    if success:
        tts.say("Emergency alert successfully transmitted. Help is being notified.")
        tts.say("Recommended actions: Move to a safe location if possible, remain visible to responders.")
    else:
        tts.say("Alert transmission failed. Attempting to retry.")
        # Simple retry logic
        time.sleep(2)
        success = email_alert(subject, body, recipient, video_file if video_file else None)
        if success:
            tts.say("Retry successful. Help has been notified.")
        else:
            tts.say("Critical failure: Could not send alert. Please try alternative methods.")
    
    tts.runAndWait()

    # Cleanup
    if video_file and os.path.exists(video_file):
        os.remove(video_file)

    # Final instructions
    tts.say("Emergency protocol complete. Stay calm and await assistance.")
    tts.runAndWait()

if __name__ == "__main__":
    print("=== EMERGENCY ALERT SYSTEM ACTIVATED ===")
    sos_main()
    print("=== SYSTEM SHUTDOWN ===")