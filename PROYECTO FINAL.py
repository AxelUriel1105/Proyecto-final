#PROYECTO FINAL
# Detector de platas desechables 

# ENTRENAMIENTO DEL MODELO
# Importar librerías necesarias
import tensorflow as tf
from yolov5 import YOLOv5

# Cargar datos anotados y configuraciones
dataset = ... # Cargar tu dataset anotado
model = YOLOv5('yolov5s.yaml')  # Configurar YOLOv5 con un archivo de configuración

# Entrenar el modelo
model.train(data=dataset, epochs=50, batch_size=16)

# Guardar el modelo entrenado
model.save('model/platos_detector.pt')






