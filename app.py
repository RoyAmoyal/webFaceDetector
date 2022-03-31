from engineio.payload import Payload
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import io
from PIL import Image
import base64, cv2
import numpy as np
import time
from facerec import Facerec

app = Flask(__name__, template_folder='Templates')
socketio = SocketIO(app, cors_allowed_origins='*')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_recog = Facerec()
face_recog.load_encoding_images("images/")
Payload.max_decode_packets = 2048


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('main.html')


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def face_recogintion(frame):
    face_locations, face_names = face_recog.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
    return frame


def face_detector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame


@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)


@socketio.on('image')
def image(data_image):
    # start_time = time.time()
    frame = (readb64(data_image))
    h, w, c = frame.shape
    print('width:  ', w)
    print('height: ', h)
    # Convert into grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    # Draw rectangle around the faces
    start_time = time.time()
    #frame = face_detector(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = face_recogintion(frame)
    print("--- %s seconds ---" % (time.time() - start_time))

    # frame = cv2.flip(frame,1)
    imgencode = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData
    # print("--- %s seconds ---" % (time.time() - start_time))

    # emit the frame back
    emit('response_back', stringData)


if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)
