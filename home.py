from tkinter import *
import speech_recognition as sr
import threading
from locateme import *
from AddFriend import *
import pyttsx3
from journey import *
from sos import *
from main import *
import time

engine = pyttsx3.init()
flag = True
class VoiceNavigatedUI:
    def __init__(self):
        self.win1 = None
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening = False
        self.create_home_page()

    def map_command(self):
        messagebox.showinfo("Please Wait","Opening Map interface...")
        locate_main()

    def journey_command(self):
        # messagebox.showinfo( "Please Wait","Opening Journey interface...")
        start()

    def emergency_command(self):
        # messagebox.showinfo( "Please Wait","Opening Emergency interface...")
        sos_main()

    def go_to_login(self):
        # engine.say("Registration")
        self.win1.destroy()
        main_main()

    def add_friend_command(self):
        save_person_image()

    def start_listening(self):
        self.listening = True
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=3)
                    text = self.recognizer.recognize_google(audio).lower()
                    print("You said:", text)
                    time.sleep(0.2)

                    if "where am i" in text:
                        self.win1.after(0, self.map_command)
                    elif "journey" in text:
                        self.win1.after(0, self.journey_command)
                    elif "emergency" in text:
                        self.win1.after(0, self.emergency_command)
                    elif "add friend" in text or "friend" in text:
                        self.win1.after(0, self.add_friend_command)
                    elif "register" in text or "login" in text:
                        self.win1.after(0, self.go_to_login)
                    elif "exit" in text:
                        self.win1.destroy()
                        self.listening = False

                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    print("API unavailable")
                except sr.WaitTimeoutError:
                    continue

    def create_home_page(self):
        self.win1 = Tk()
        self.win1.title("Voice-Controlled Home Page")
        self.win1.geometry("1200x689+50+20")

        try:
            self.win1.attributes('-toolwindow', True)
        except TclError:
            print('Toolwindow not supported on this platform')

        main_back = PhotoImage(file="Images/landing.png")
        bl = PhotoImage(file="Images/register_home.png")

        Label(self.win1, image=main_back).place(x=0, y=0)

        Button(self.win1, text="Where am I?", foreground="white", font=("Times New Roman", 15),
            background="red", width=10, command=self.map_command).place(x=100, y=523)

        Button(self.win1, text="Journey", foreground="white", font=("Times New Roman", 15),
            background="red", width=10, command=self.journey_command).place(x=392, y=523)

        Button(self.win1, text="Emergency", foreground="white", font=("Times New Roman", 15),
            background="red", width=10, command=self.emergency_command).place(x=680, y=523)

        Button(self.win1, text="Add Friend", foreground="white", font=("Times New Roman", 15),
            background="red", width=10, command=self.add_friend_command).place(x=962, y=523)

        Button(self.win1, image=bl, borderwidth=0, highlightthickness=0,
            command=self.go_to_login).place(x=474, y=162)

        engine.say("How can I help you?")
        engine.runAndWait()

        self.listening = True
        threading.Thread(target=self.start_listening, daemon=True).start()

        self.win1.protocol("WM_DELETE_WINDOW", self.on_close)
        self.win1.mainloop()


    def on_close(self):
        self.listening = False
        self.win1.destroy()

# Start the application
if __name__ == "__main__":
    VoiceNavigatedUI()