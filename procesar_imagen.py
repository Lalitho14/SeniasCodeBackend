import cv2
import mediapipe as mp
import numpy as np

def ProcesarImagen(image_reciv):
  mp_hands = mp.solutions.hands
  hands = mp_hands.Hands(static_image_mode=True, max_num_hands = 1, min_detection_confidence=0.5)
  
  image = cv2.imread(image_reciv)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  results = hands.process(image_rgb)

  if results.multi_hand_landmarks:
    height, width, _ = image.shape
    for hand_landmarks in results.multi_hand_landmarks:
      x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * width
      x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * width
      y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * height
      y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * height

      x_min, x_max = int(x_min) - 100, int(x_max) + 100
      y_min, y_max = int(y_min) - 100, int(y_max) + 100

      hand_image = image[y_min:y_max, x_min:x_max]

      cv2.imwrite("./cacheimg/Manorecortada.jpg", hand_image)
  else:
    print("No se detectaron manos en la imagen")

  hands.close()