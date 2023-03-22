import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import winsound

# Load the saved model
model = tf.keras.models.load_model('fire_detection_model.h5')
video = cv2.VideoCapture(1)
# Set the frequency and duration of the sound to be played
frequency = 2500  # in Hertz
duration = 500  # in milliseconds
while True:
    _, frame = video.read()

    # Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')

    # Resizing into 224x224 because we trained the model with this image size.
    im = im.resize((224, 224))
    img_array = img_to_array(im)
    img_array = np.expand_dims(img_array, axis=0) / 255

    # Calling the predict method on model to predict 'fire' on the image
   
    probabilities = model.predict(img_array)[0]
    prediction = np.argmax(probabilities)

    # if prediction is 0, which means there is fire in the frame.
    if prediction == 0:
        # Draw a red rectangle around the area where fire is detected
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), thickness=2)

        # Display the probability of fire on the frame as a percentage
        prob_percent = round(probabilities[prediction] * 100, 2)
        prob_text = f'Fire: {prob_percent}%'
        cv2.putText(frame, prob_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        winsound.Beep(frequency, duration)
    
    # Display the video frame
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
video.release()
cv2.destroyAllWindows()
