import tensorflow as tf
from flask import Flask, request
import numpy 
import cv2
app = Flask(__name__)

class Inferrer:
    def __init__(self):
        self.saved_path = './graymodel.h5'
        self.model = tf.keras.models.load_model(self.saved_path)
    
    def preprocess(self, image):
        return image
   
    def infer(self, image):
        image = image.read()
        image_bytes = numpy.fromstring(image, numpy.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (96, 96))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = numpy.reshape(image, (96, 96, 1))
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor_image = self.preprocess(tensor_image)
        shape = tensor_image.shape
        tensor_image = tf.reshape(tensor_image, [1, shape[0], shape[1], shape[2]])
        return self.model.predict(tensor_image)

inferrer = Inferrer()

@app.route('/infer', methods=["POST"])
def infer():
    image = request.files.get('image', '')
    res = inferrer.infer(image) 
    print(res)
    return "0" if res < 0.5 else "1"

@app.route('/infer', methods=['GET'])
def test():
    return "test"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 8080, debug=True)
