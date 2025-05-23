from flask import Flask, request
import base64
from flask_cors import CORS
from procesar_imagen import ProcesarImagen
from inference_rest import PrediccionLetraServer
#from test_model import EvaluarModelo
from test_model_ import EvaluarModelo
import os
import time

VIDEO_CACHE = 'uploaded_video'
os.makedirs(VIDEO_CACHE, exist_ok=True)

app = Flask(__name__)
CORS(app)

@app.route('/uploadFrame', methods=['POST'])
def upload():
  data = request.json
  image_data = data['image']
  with open('captured_fram.jpg', 'wb') as f:
    f.write(base64.b64decode(image_data))
    
  ProcesarImagen('captured_fram.jpg')
  
  time.sleep(0.5)
  letra = PrediccionLetraServer()
  time.sleep(0.5)
  
  return letra, 200

@app.route('/upload_video', methods=['POST'])
def upload_video():
  
  if 'video' not in request.files:
    return "No se ha enviado el archivo de video", 400
  
  video = request.files['video']
  
  if video.filename == '':
    return "No se sha sleccionado ningun archivo", 400
  
  filepath = os.path.join(VIDEO_CACHE, video.filename)
  video.save(filepath)
  
  frase = EvaluarModelo(src=filepath)
  
  return frase, 200

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)