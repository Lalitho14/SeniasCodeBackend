import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import h5py

KEYPOINTS_LEN = 1659


class ExtractorDatos:
  def __init__(self, _holstic_model, _mp_holistic, _in_folder=None, _out_folder=None):
    self.holistic = _holstic_model
    self.mp_holistic = _mp_holistic
    self.in_folder = _in_folder
    self.out_folder = _out_folder

  def GuardarKeypointsH5(self, _sequences, _labels, _out_folder):
    with h5py.File(_out_folder, 'w') as f:
      for i, seq in enumerate(tqdm(_sequences, desc="Guardando secuencias...")):
        f.create_dataset(f'sequences/{i}', data=seq, compression='gzip')

      dt = h5py.string_dtype(encoding='utf-8')
      labels_array = np.array(_labels, dtype=dt)
      f.create_dataset('labels', data=labels_array, compression='gzip')

  def CargarKeypointsH5(self, _in_folder):
    sequences = []
    labels = []

    with h5py.File(_in_folder, 'r') as f:
      for i in tqdm(range(len(f['sequences'])), desc="Cargando secuencias..."):
        sequences.append(np.array(f[f'sequences/{i}']))

      # labels = [label.decode('utf-8') for label in f['sequences'][:]]
      if 'labels' in f:
        labels = [label.decode('utf-8') for label in f['labels'][:]]

    return sequences, labels

  def NormalizarKeypoints(self, keypoints_array):
    # Suponiendo que los primeros 33*3 valores son pose (índice 0 es nariz)
    reference_x, reference_y, reference_z = keypoints_array[
      0], keypoints_array[1], keypoints_array[2]
    normalized = keypoints_array.copy()
    for i in range(0, len(keypoints_array), 3):
      normalized[i] -= reference_x    # x
      normalized[i + 1] -= reference_y  # y
      normalized[i + 2] -= reference_z  # z

    return normalized

  def ProcesarFrame(self, frame):
    # holistic = InicializarHolistic()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.holistic.process(frame_rgb)

    # Extraer landmarks (si no hay detección, devuelve array de ceros)
    # Brazos
    pose = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
                    ).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    # Cara
    face = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.face_landmarks.landmark]
                    ).flatten() if results.face_landmarks else np.zeros(468 * 3)
    # Mano izquierda
    lh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.left_hand_landmarks.landmark]).flatten(
      ) if results.left_hand_landmarks else np.zeros(21 * 3)
    # Mano derecha
    rh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark]).flatten(
      ) if results.right_hand_landmarks else np.zeros(21 * 3)

    keypoints = np.concatenate([pose, face, lh, rh])

    # print(f"Forma del array de keypoints: {keypoints.shape}")

    # GraficarKeypoints(keypoints, "NO Normalizados")

    keypoints_normalizados = self.NormalizarKeypoints(keypoints)
    #self.VerificarFrame(frame, results)

    # GraficarKeypoints(keypoints_normalizados, "Normalizados")

    if len(keypoints_normalizados) != KEYPOINTS_LEN:
      keypoints_normalizados = np.pad(
        keypoints_normalizados, (0, KEYPOINTS_LEN - len(keypoints_normalizados)), 'constant')
    elif len(keypoints_normalizados) > KEYPOINTS_LEN:
      keypoints_normalizados = keypoints_normalizados[:KEYPOINTS_LEN]

    # Puntos recolectados
    return keypoints_normalizados

  def Extraer_Keypoints(self, in_ruta, out_ruta):
    # Procesar puntos recolectados a carpeta
    sequences = []
    labels = []

    for expresion in os.listdir(in_ruta):
      ruta_expresion = os.path.join(in_ruta, expresion)

      if not os.path.isdir(ruta_expresion):
        continue

      for muestra in tqdm(os.listdir(ruta_expresion), desc=f"Procesando {ruta_expresion}"):
        ruta_muestra = os.path.join(ruta_expresion, muestra)
        sequence_frames = []

        frame_files = [f for f in os.listdir(
          ruta_muestra) if f.endswith('.jpg') or f.endswith('.png')]

        for frame_file in frame_files:
          frame = cv2.imread(os.path.join(ruta_muestra, frame_file))
          if frame is not None:
            keypoints = self.ProcesarFrame(frame)
            sequence_frames.append(keypoints)

        sequences.append(np.array(sequence_frames))
        labels.append(expresion)

    self.GuardarKeypointsH5(sequences, labels, out_folder)
    print(
      f"Extraccion para {expresion} completada. Datos guardados en {out_ruta}")

  def VerificarFrame(self, _frame, results):
      # Dibujar landmarks en el frame
    mp_drawing = mp.solutions.drawing_utils
    annotated_frame = _frame.copy()
    mp_drawing.draw_landmarks(
      annotated_frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(
      annotated_frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
      annotated_frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
      annotated_frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

    cv2.imshow("Frame", annotated_frame)
    cv2.waitKey(2500)  # Pausa 1ms

  def GraficarKeypoints(self, array_keypoints, title):
    plt.scatter(array_keypoints[::3], array_keypoints[1::3], s=1)
    plt.title(title)
    plt.show()

  def InicializarHolistic(self, ):
    # InicializarMediapipe
    self.mp_holistic = mp.solutions.holistic
    self.holistic = self.mp_holistic.Holistic(
      static_image_mode=True,
      model_complexity=2,            # 0 (ligero), 1 (medio), 2 (completo)
      smooth_landmarks=True,
      refine_face_landmarks=True     # Más precisión en rostro
    )

    return holistic

def CargarKeypointsH5(_in_folder):
  sequences = []
  labels = []

  with h5py.File(_in_folder, 'r') as f:
    for i in tqdm(range(len(f['sequences'])), desc="Cargando secuencias..."):
      sequences.append(np.array(f[f'sequences/{i}']))

    # labels = [label.decode('utf-8') for label in f['sequences'][:]]
    if 'labels' in f:
      labels = [label.decode('utf-8') for label in f['labels'][:]]

  return sequences, labels

if __name__ == "__main__":
  # InicializarMediapipe
  mp_holistic = mp.solutions.holistic
  holistic = mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,            # 0 (ligero), 1 (medio), 2 (completo)
    smooth_landmarks=True,
    refine_face_landmarks=True     # Más precisión en rostro
  )

  in_folder = "./frame_actions"
  out_folder = "./data/sign_language_keypoints.h5"

  extractor_data = ExtractorDatos(holistic, mp_holistic, in_folder, out_folder)

  extractor_data.Extraer_Keypoints(in_folder, out_folder)
