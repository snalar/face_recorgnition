import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import subprocess
import threading
import os
import numpy as np
import pickle
import face_recognition
import time






class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x700")
        self.root.resizable(False, False)

        self.video_capture = None
        self.running = False

        self.build_ui()
        self.data = None
        self.load_model()

        self.mode = "idle"   # idle | collect | realtime
        self.last_save_time = 0
        self.save_interval = 1  # секунд
        self.face_counter = 0

        # Haar Cascade
        cascPath = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        



    def build_ui(self):
        # ===== ЛЕВАЯ ПАНЕЛЬ =====
        control_frame = tk.Frame(self.root, width=350, bg="#f0f0f0")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # === Сбор датасета ===
        tk.Label(control_frame, text="Сбор датасета", font=("Arial", 12, "bold")).pack(pady=5)

        tk.Label(control_frame, text="Имя пользователя:").pack()
        self.username_entry = tk.Entry(control_frame)
        self.username_entry.insert(0, "User1")
        self.username_entry.pack(pady=5)

        tk.Button(control_frame, text="▶ Начать сбор данных", command=self.collect_dataset).pack(pady=5)

        # === Обучение модели ===
        tk.Label(control_frame, text="Обучение модели", font=("Arial", 12, "bold")).pack(pady=10)
        tk.Button(control_frame, text="✏ Обучить на датасете", command=self.train_model).pack(pady=5)

        # === Фото ===
        tk.Label(control_frame, text="Распознавание по фото", font=("Arial", 12, "bold")).pack(pady=10)
        tk.Button(control_frame, text="📁 Выбрать и распознать фото", command=self.recognize_photo).pack(pady=5)

        # === Real-time ===
        tk.Label(control_frame, text="Распознавание в реальном времени", font=("Arial", 12, "bold")).pack(pady=10)
        tk.Button(control_frame, text="▶ Старт распознавания", command=self.start_realtime).pack(pady=5)
        tk.Button(control_frame, text="■ Остановить", command=self.stop_realtime).pack(pady=5)

        # === Статус ===
        tk.Label(control_frame, text="Статус", font=("Arial", 12, "bold")).pack(pady=10)
        self.status_label = tk.Label(control_frame, text="Режим: Ожидание")
        self.status_label.pack(pady=5)

        # ===== ПРАВАЯ ПАНЕЛЬ =====
        display_frame = tk.Frame(self.root, bg="black")
        display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.video_label = tk.Label(display_frame, bg="black")
        self.video_label.pack(expand=True)

        self.show_text("Выберите режим работы для отображения видео")

    # ===== ФУНКЦИИ =====

    def show_text(self, text):
        img = Image.new("RGB", (800, 600), "black")
        self.video_img = ImageTk.PhotoImage(img)
        self.video_label.config(image=self.video_img)
        self.video_label.config(text=text, fg="white", font=("Arial", 16))

    def collect_dataset(self):
        username = self.username_entry.get().strip()
        if not username:
            messagebox.showerror("Ошибка", "Введите имя пользователя")
            return

        self.dataset_path = os.path.join("Images", username)
        os.makedirs(self.dataset_path, exist_ok=True)

        self.face_counter = 0
        self.last_save_time = 0
        self.mode = "collect"

        self.cap = cv2.VideoCapture(0)
        self.status_label.config(text=f"Сбор датасета: {username}")

        self.update_camera_frame()

    def load_model(self):
        if os.path.exists("face_enc"):
            with open("face_enc", "rb") as f:
                self.data = pickle.loads(f.read())
            self.status_label.config(text="Модель загружена")
        else:
            self.data = None
            self.status_label.config(text="Модель не обучена")


    def update_camera_frame(self):
        # если ничего не запущено — ничего не делаем
        if self.mode == "idle":
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # ===============================
        # РЕЖИМ СБОРА ДАТАСЕТА (HAAR)
        # ===============================
        if self.mode == "collect":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                # рамка лица
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # сохранение лиц с интервалом
                if time.time() - self.last_save_time >= self.save_interval:
                    face_img = frame[y:y + h, x:x + w]
                    img_path = os.path.join(
                        self.dataset_path,
                        f"{self.face_counter}.jpg"
                    )
                    cv2.imwrite(img_path, face_img)
                    self.face_counter += 1
                    self.last_save_time = time.time()

                # отображение счётчика
                cv2.putText(
                    frame,
                    f"Saved: {self.face_counter}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        # ===================================
        # REAL-TIME РАСПОЗНАВАНИЕ (DLIB)
        # ===================================
        elif self.mode == "realtime":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding, (top, right, bottom, left) in zip(encodings, boxes):
                name = "Unknown"

                matches = face_recognition.compare_faces(
                    self.data["encodings"],
                    encoding,
                    tolerance=0.5
                )

                if True in matches:
                    matched_idxs = [i for i, v in enumerate(matches) if v]
                    counts = {}

                    for i in matched_idxs:
                        person = self.data["names"][i]
                        counts[person] = counts.get(person, 0) + 1

                    name = max(counts, key=counts.get)

                # рамка лица
                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2
                )

                # имя
                cv2.putText(
                    frame,
                    name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        # ===============================
        # ОТОБРАЖЕНИЕ В TKINTER
        # ===============================
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (800, 600))

        imgtk = ImageTk.PhotoImage(Image.fromarray(frame))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk

        # обновление кадра
        self.root.after(10, self.update_camera_frame)



    def train_model(self):
        self.status_label.config(text="Обучение модели...")

        process = subprocess.Popen(
            ["python", "face_recognition_from_dataset.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        def wait_and_load():
            process.wait()
            self.load_model()
            messagebox.showinfo("Готово", "Модель успешно обучена")

        threading.Thread(target=wait_and_load, daemon=True).start()


    def recognize_photo(self):
        if self.data is None:
            messagebox.showerror("Ошибка", "Модель не обучена")
            return
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png")]
        )
        if not file_path:
            return

        self.status_label.config(text="Распознавание по фото")

        # === безопасная загрузка (работает с кириллицей) ===
        img_array = np.fromfile(file_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            messagebox.showerror("Ошибка", "Не удалось загрузить изображение")
            return

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # === поиск лиц ===
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        # === распознавание ===
        for encoding, (top, right, bottom, left) in zip(encodings, boxes):
            matches = face_recognition.compare_faces(self.data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for i, b in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    counts[self.data["names"][i]] = counts.get(self.data["names"][i], 0) + 1

                name = max(counts, key=counts.get)

            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # === отображение в блоке "Отображение" ===
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (800, 600))

        imgtk = ImageTk.PhotoImage(Image.fromarray(image))
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk



    def start_realtime(self):
        if self.data is None:
            messagebox.showerror("Ошибка", "Сначала обучите модель")
            return

        self.mode = "realtime"
        self.cap = cv2.VideoCapture(0)
        self.status_label.config(text="Real-time распознавание")
        self.update_camera_frame()


    


    def update_realtime_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding, (top, right, bottom, left) in zip(encodings, boxes):
            matches = face_recognition.compare_faces(self.data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for i, b in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    counts[self.data["names"][i]] = counts.get(self.data["names"][i], 0) + 1

                name = max(counts, key=counts.get)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (800, 600))
        img = ImageTk.PhotoImage(Image.fromarray(frame))

        self.video_label.config(image=img)
        self.video_label.image = img

        self.root.after(10, self.update_realtime_frame)


    def stop_realtime(self):
        self.mode = "idle"
        if hasattr(self, "cap"):
            self.cap.release()
        self.status_label.config(text="Режим: Ожидание")