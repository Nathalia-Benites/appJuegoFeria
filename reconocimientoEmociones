import cv2
import os
import numpy as np

def obtener_mensaje(emotion):
    mensajes = {
        'Felicidad': '¡Qué bueno verte feliz! Sigue sonriendo :)',
        'Enojo': 'Parece que estás molesto. Intenta relajarte y respirar hondo.',
        'Sorpresa': '¡Vaya! Algo te sorprendió. ¡Espero que sea algo positivo!',
        'Tristeza': 'Lo siento que te sientas así. Recuerda que siempre hay esperanza.'
    }
    return mensajes.get(emotion, 'Emoción no reconocida.')

def emotion_image(emotion):
    emojis = {
        'Felicidad': 'Emojis/felicidad.jpeg',
        'Enojo': 'Emojis/enojo.jpeg',
        'Sorpresa': 'Emojis/sorpresa.jpeg',
        'Tristeza': 'Emojis/tristeza.jpeg'
    }
    image_path = emojis.get(emotion, 'Emojis/default.jpeg')
    if os.path.exists(image_path):
        return cv2.imread(image_path)
    else:
        print(f"Imagen de emoción no encontrada: {image_path}")
        return None

# Método usado para el entrenamiento y lectura del modelo
method = 'LBPH'  # Cambiar a 'EigenFaces', 'FisherFaces', o 'LBPH' según el modelo entrenado

try:
    if method == 'LBPH':
        from cv2.face import LBPHFaceRecognizer_create

        emotion_recognizer = LBPHFaceRecognizer_create()
    elif method == 'EigenFaces':
        from cv2.face import EigenFaceRecognizer_create

        emotion_recognizer = EigenFaceRecognizer_create()
    elif method == 'FisherFaces':
        from cv2.face import FisherFaceRecognizer_create

        emotion_recognizer = FisherFaceRecognizer_create()
    else:
        raise ValueError("Método no reconocido.")
except ImportError:
    print("No se pudo importar el módulo cv2.face. Verifica tu instalación de opencv-contrib-python.")
    exit()

try:
    emotion_recognizer.read(f'modelo_{method}.xml')
except Exception as e:
    print(f"Error al leer el modelo: {e}")
    exit()

dataPath = r'C:\Users\Cesar Benites\OneDrive - Academia Naval Almirante Illingworth (1)\JUEGO FERIA\Data'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede capturar el video.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        rostro = gray[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)

        emotion = imagePaths[result[0]]
        if (method == 'EigenFaces' and result[1] < 5700) or \
                (method == 'FisherFaces' and result[1] < 500) or \
                (method == 'LBPH' and result[1] < 60):
            cv2.putText(frame, emotion, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Mostrar imagen de la emoción
            emotion_img = emotion_image(emotion)
            if emotion_img is not None:
                emotion_img = cv2.resize(emotion_img, (100, 100))  # Redimensionar imagen
                frame[10:110, 10:110] = emotion_img  # Mostrar imagen en la esquina superior izquierda

            # Mostrar mensaje
            mensaje = obtener_mensaje(emotion)
            cv2.putText(frame, mensaje, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
        else:
            cv2.putText(frame, 'No identificado', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1,
                        cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27:  # Presiona 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()
