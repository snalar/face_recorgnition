import face_recognition
import pickle
import cv2

# Загрузка известного энкодинга
data = pickle.loads(open('face_enc', "rb").read())

# Загрузка изображения
image = cv2.imread('test1.jpg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Поиск лиц и их энкодингов
face_locations = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, face_locations)

# Распознавание
for (encoding, (top, right, bottom, left)) in zip(encodings, face_locations):
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        name = max(counts, key=counts.get)

    # Рисуем рамку и имя
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# Показываем результат
cv2.imshow("Frame", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
