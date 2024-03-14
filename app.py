import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('templates/final_model.h5')

# Define the labels for the classes
class_labels = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# index.html page
@app.route('/')
def index():
    return render_template('index.html')

# predict and display it on result.html page
@app.route('/upload', methods=['POST'])
def upload():
    try:
        image = request.files['upload-button']
        image = Image.open(image)
        image = image.resize((224, 224))  # Resize the image
        img_array = tf.keras.preprocessing.image.img_to_array(image)  # Convert the image to a NumPy array
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch
        img_array /= 255.0  # Normalize the pixel values

        # Make a prediction
        predictions = model.predict(img_array)

        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions)

        return render_template('results.html', result=predicted_class, percentage=confidence)

    except Exception as e:
        # Handle any exceptions, e.g., invalid file format
        error_message = str(e)
        return f"An error occurred: {error_message}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
