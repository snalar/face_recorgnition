import cv2
import os
import time

# === Имя пользователя, которого собираем ===
person_name = "User1"  # ← замени на нужное имя
dataset_path = f"Images/{person_name}"

# Создаем папку если ее нет
os.makedirs(dataset_path, exist_ok=True)

# Инициализируем веб-камеру
cap = cv2.VideoCapture(0)

# Загружаем Haar Cascade для детекции лиц
cascPath = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

print("Начинаю сбор датасета. Нажми 'q' чтобы выйти.")
counter = 0
last_save_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Рисуем рамку
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Автосохранение лица каждые 0.5 секунды
        if time.time() - last_save_time > 0.5:
            face_img = frame[y:y + h, x:x + w]
            img_path = f"{dataset_path}/{counter}.jpg"
            cv2.imwrite(img_path, face_img)
            counter += 1
            last_save_time = time.time()
            print(f"Сохранен кадр: {img_path}")

    cv2.imshow("Dataset Collector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Сбор датасета завершён.")