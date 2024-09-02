import cv2
import cv2.face
import os
import numpy as np
import time


def obtenerModelo(method, facesData, labels):
    if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
    if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Entrenando el reconocedor de rostros
    print(f"Entrenando ({method})...")
    inicio = time.time()
    emotion_recognizer.train(facesData, np.array(labels))
    tiempoEntrenamiento = time.time() - inicio
    print(f"Tiempo de entrenamiento ({method}): {tiempoEntrenamiento}")

    # Almacenando el modelo obtenido
    emotion_recognizer.write(f"modelo_{method}.xml")


dataPath = r'C:\Users\Cesar Benites\OneDrive - Academia Naval Almirante Illingworth (1)\JUEGO FERIA\Data'  # Cambia a la ruta donde hayas almacenado Data
emotionsList = os.listdir(dataPath)
print('Lista de emociones: ', emotionsList)

labels = []
facesData = []
label = 0

for nameDir in emotionsList:
    emotionsPath = os.path.join(dataPath, nameDir)

    for fileName in os.listdir(emotionsPath):
        labels.append(label)
        facesData.append(cv2.imread(os.path.join(emotionsPath, fileName), 0))
    label += 1

# Entrenar modelos
obtenerModelo('EigenFaces', facesData, labels)
obtenerModelo('FisherFaces', facesData, labels)
obtenerModelo('LBPH', facesData, labels)
