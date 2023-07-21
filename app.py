from flask import Flask, render_template, Response
import cv2
import numpy as np
from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    json_file = open("model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("model.h5")

    actions=['A','B','C','D','E','F', 'G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

    sequence = []
    sentence = []
    accuracy = []
    predictions = []
    threshold = 0.8
    max_actions_displayed = 3  # Maximum number of actions to display
    displayed_actions = []

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            cropframe = frame[40:400, 0:300]
            frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
            image, results = mediapipe_detection(cropframe, hands)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try:
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))

                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            action = actions[np.argmax(res)]
                            if action not in displayed_actions and action not in sentence:
                                displayed_actions.append(action)

                                # Add action to the sentence if it is not already present
                                if len(sentence) < 3:
                                    sentence.append(action)
                                    accuracy.append(str(res[np.argmax(res)] * 100))
                                else:
                                    sentence = [action]
                                    accuracy = [str(res[np.argmax(res)] * 100)]

                    if len(displayed_actions) > max_actions_displayed:
                        displayed_actions = displayed_actions[-max_actions_displayed:]

            except Exception as e:
                pass

            # Remove duplicate actions from displayed_actions list
            displayed_actions = list(set(displayed_actions))

            # Display recognized actions on the frame
            output_text = "Output: " + ' '.join(sentence)
            accuracy_text = "Accuracy: " + accuracy[-1] if accuracy else ""
            cv2.putText(frame, output_text, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, accuracy_text, (3, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()