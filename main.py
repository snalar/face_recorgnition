import tkinter as tk
from app import FaceRecognitionApp

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()