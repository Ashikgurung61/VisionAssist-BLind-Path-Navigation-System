from tkinter import *
import mysql.connector
from tkinter import messagebox
import speech_recognition as sr
import threading
from home import *
from main import *

con = mysql.connector.connect(host="localhost", user="root", passwd="2020Bca01", database="blindpath")
cursor = con.cursor()

def home(current_window=None):
    current_window.destroy()
    VoiceNavigatedUI()

def main_main():
    mroot = Tk()
    mroot.title("Personalize Learning")

    try:
        mroot.attributes('-toolwindow', True)
    except TclError:
        print('Not supported on your platform')

    # Voice recognition setup
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    listening = True

    def voice_listener():
        nonlocal listening
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            while listening:
                try:
                    audio = recognizer.listen(source, timeout=3)
                    text = recognizer.recognize_google(audio).lower()
                    print("Heard:", text)
                    if "home" in text:
                        mroot.after(0, lambda: home(mroot))
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    print("API unavailable")
                except sr.WaitTimeoutError:
                    continue

    # Start voice listener thread
    threading.Thread(target=voice_listener, daemon=True).start()

    # Configuration of BackGround and inserting images
    mroot.geometry("1050x650+120+45")
    main_bg = PhotoImage(file="Images/login.png")
    hm = PhotoImage(file="Images/home.png")
    label_back = Label(mroot, image=main_bg)
    label_back.place(x=0, y=0)

    # Create entries
    name = Entry(mroot, bd=0, width=18, font=("Helvetica", 22), background="white")
    name.place(x=646, y=268)

    email = Entry(mroot, bd=0, width=18, font=("Helvetica", 22), background="white")
    email.place(x=646, y=358)

    password = Entry(mroot, bd=0, width=18, font=("Helvetica", 22), background="white")
    password.place(x=646, y=448)

    # Home button with command
    home_btn = Button(mroot, image=hm, borderwidth=0, highlightthickness=0, command=lambda: home(mroot))
    home_btn.place(x=37, y=25)

    def login():
        n = name.get()
        c = email.get()
        o = password.get()

        val1 = (n, c, o)
        try:
            if not all([n, c, o]):
                messagebox.showerror("Empty", "Please Enter Details")
            else:
                cursor.execute("INSERT INTO login (username, email, password) VALUES (%s, %s, %s)", val1)
                con.commit()
                messagebox.showinfo("Success", "Successful")
                mroot.destroy()
                # GameStart(last_id)
        except mysql.connector.Error as err:
            messagebox.showerror("Error", "Please Insert the correct Value")
        finally:
            con.commit()

    submit = Button(mroot, text="Register", font=("Helvetica", 18), width=10,
                    command=login, borderwidth=0, highlightthickness=0)
    submit.place(x=730, y=550)

    # Handle window close
    def on_close():
        nonlocal listening
        listening = False
        mroot.destroy()

    mroot.protocol("WM_DELETE_WINDOW", on_close)
    mroot.mainloop()

if __name__ == "__main__":
    main_main()