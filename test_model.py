import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic, FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


def ExtraerKeypoints(results):
  pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
                  ).flatten() if results.pose_landmarks else np.zeros(33 * 4)
  face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
                  ).flatten() if results.face_landmarks else np.zeros(468 * 3)
  lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
                ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
  rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
                ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
  return np.concatenate([pose, face, lh, rh])


def GetWordIds(path):
  with open(path, 'r') as json_file:
    data = json.load(json_file)
    return data.get('word_ids')


def MediapipeDetection(image, model):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  results = model.process(image)
  return results


def InterpolateKeypoints(keypoints, target_length=15):
  current_length = len(keypoints)
  if current_length == target_length:
    return keypoints

  indices = np.linspace(0, current_length - 1, target_length)
  interpolated_keypoints = []
  for i in indices:
    lower_idx = int(np.floor(i))
    upper_idx = int(np.ceil(i))
    weight = i - lower_idx
    if lower_idx == upper_idx:
      interpolated_keypoints.append(keypoints[lower_idx])
    else:
      interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(keypoints[upper_idx])
      interpolated_keypoints.append(interpolated_point.tolist())

  return interpolated_keypoints


def NormalizarKeypoints(keypoints, target_length=15):
  current_length = len(keypoints)
  if current_length < target_length:
    return InterpolateKeypoints(keypoints, target_length)
  elif current_length > target_length:
    step = current_length / target_length
    indices = np.arange(0, current_length, step).astype(int)[:target_length]
    return [keypoints[i] for i in indices]
  else:
    return keypoints
  
def DibujarPuntos(image, results):
  # Cara
  draw_landmarks(
    image,
    results.face_landmarks,
    FACEMESH_CONTOURS,
    DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
    DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
  )
  # Cuerpo
  draw_landmarks(
    image,
    results.pose_landmarks,
    POSE_CONNECTIONS,
    DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
    DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
  )
  # Mano Izq
  draw_landmarks(
    image,
    results.left_hand_landmarks,
    HAND_CONNECTIONS,
    DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
    DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
  )
  # Mano Derecha
  draw_landmarks(
    image,
    results.right_hand_landmarks,
    HAND_CONNECTIONS,
    DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
    DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
  )


def EvaluarModelo(src=None, threshold=0.1, margin_frame=1, delay_frames=3):
  ROOT_PATH = os.getcwd()
  MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
  WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words.json")
  MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{15}.keras")
  kp_seq, sentence = [], []
  word_ids = GetWordIds(WORDS_JSON_PATH)
  model = load_model(MODEL_PATH)
  count_frame = 0
  fix_frames = 0
  capturando = False

  with Holistic() as holistic_model:
    video = cv2.VideoCapture(src or 0)

    while video.isOpened():
      ret, frame = video.read()
      if not ret:
        break

      results = MediapipeDetection(frame, holistic_model)

      if (results.left_hand_landmarks or results.right_hand_landmarks) or capturando:
        capturando = False
        count_frame += 1
        if count_frame > margin_frame:
          kp_frame = ExtraerKeypoints(results)
          kp_seq.append(kp_frame)

      else:
        if count_frame >= 5 + margin_frame:
          fix_frames += 1
          if fix_frames < delay_frames:
            capturando = True
            continue
          kp_seq = kp_seq[:-(margin_frame +delay_frames)]
          kp_normalzed = NormalizarKeypoints(kp_seq, 15)
          res = model.predict(np.expand_dims(kp_normalzed, axis=0))[0]
          
          print(np.argmax(res), f"({res[np.argmax(res)] * 100}%)")
          
          if(res[np.argmax(res)] > threshold):
            word_id = word_ids[np.argmax(res)].split('-')[0]
            
            print(f"Palabra detectada : {word_id}")
            return word_id
          
          else:
            return "No se deteccto frase"
            
        capturando = False
        fix_frames = 0
        count_frame = 0
        kp_seq = []
        
      if not src:
        cv2.rectangle(frame, (0,0), (640,35), (245,117,16), -1)
        cv2.putText(frame, ' | '.join(sentence), (5,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255))
        
        DibujarPuntos(frame, results)
        cv2.imshow('TEST MODEL', frame)
        if cv2. waitKey(10) & 0xFF == ord('q'):
          break
        
    video.release()
    cv2.destroyAllWindows()
    
# if __name__ == "__main__":
# #   EvaluarModelo(src="./uploaded_video/video.webm")
#   EvaluarModelo(src="./uploaded_video/video.webm")
