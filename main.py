import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Define custom objects if needed
custom_objects = {
    'DepthwiseConv2D': DepthwiseConv2D
}

# Load the model with custom objects if necessary
try:
    model = tf.keras.models.load_model("keras_model.h5", compile=False, custom_objects=custom_objects)
except Exception as e:
    print("Error loading model:", e)
    exit()

# Load the labels
with open("labels.txt", "r") as file:
    class_names = [line.strip() for line in file]

# Open the webcam
camera = cv2.VideoCapture(0)

# Initialize a list to store recent predictions
N = 5
recent_predictions = []

while True:
    # Grab the webcamera's image
    ret, image = camera.read()
    if not ret:
        print("Failed to grab image")
        break

    # Resize the raw image to (224, 224) pixels
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Prepare the image for model prediction
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1  # Normalize the image

    # Predict with the model
    try:
        prediction = model.predict(image_array)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Store the prediction
        recent_predictions.append(confidence_score)
        # Keep only the last N predictions
        if len(recent_predictions) > N:
            recent_predictions.pop(0)
        # Calculate the average confidence score
        average_confidence = np.mean(recent_predictions)

        # Get the dimensions of the original image
        height, width, _ = image.shape

        # Coordinates to cover the entire screen
        top_left = (0, 0)
        bottom_right = (width, height)

        # Draw rectangle on the original image
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Add text label on the original image
        label_text = f"{class_name} ({np.round(average_confidence * 100, 2)}%)"
        cv2.putText(image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Error during prediction:", e)

    # Show the image with rectangle and label
    cv2.imshow("Webcam Image", image)

    # Listen for keyboard input
    keyboard_input = cv2.waitKey(1)
    # 27 is the ASCII code for the ESC key
    if keyboard_input == 27:
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()

