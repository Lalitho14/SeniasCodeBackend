import requests
from io import BytesIO
import cv2
from PIL import Image
import json
import numpy as np

def categorizar():
  # response = requests.get(url)
  # img = Image.open(BytesIO(response.content))
  img = Image.open("./cacheimg/Manorecortada.jpg")
  img = np.array(img).astype(float) / 255
  img = cv2.resize(img, (128, 128))
#   prediccion = model.predict(img.reshape(-1,128,128,3))
  prediccion = img.reshape(-1, 128, 128, 3)
  print("Forma de prediccion_data después de reshape:",
        prediccion.shape)  # Verifica la forma
  return prediccion

def PrediccionLetraServer():
  prediccion_data = categorizar()

  headers = {
      "content-type": "application/json"
  }

  request = {
      "signature_name": "serving_default",
      "instances": prediccion_data.tolist()
  }

  data = json.dumps(request)
  # print(data)
  # # print("JSON de la solicitud:", data)  # Verifica el JSON final antes de enviarlo

  respone = requests.post(
    'http://192.168.100.79:8501/v1/models/saved_model/versions/1:predict', data=data, headers=headers)

  predictions = json.loads(respone.text)['predictions']

  print(f"Prediccion de letra: {predictions}")

  suma = np.argmax(predictions[0], axis=-1)

  print(f"Letra : {np.argmax(predictions[0], axis=-1)}")

  letra_salida = 65 + suma

  print(f"Letra Caracter : {chr(letra_salida)} confianza: {predictions[0][suma]}")
  
  if(predictions[0][suma] > 0.87):
    return chr(letra_salida)
  else:
    return "No se detecto ninguna senial"