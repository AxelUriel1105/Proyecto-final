#Aplicación del modelo de entrenamiento
import cv2
from yolov5 import YOLOv5

# Cargar el modelo entrenado
model = YOLOv5('model/platos_detector.pt')

# Abrir el video
cap = cv2.VideoCapture('video_input.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar detección en el frame
    results = model.predict(frame)

    # Dibujar cajas delimitadoras en el frame
    for result in results:
        x1, y1, x2, y2, conf, cls = result
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Mostrar el frame con detecciones
    cv2.imshow('Platos Detectados', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
