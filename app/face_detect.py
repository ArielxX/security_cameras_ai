import os
import numpy as np
from tensorflow.keras.layers import Layer
import tensorflow as tf
import base64


def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img

# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    
def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('face_recognition/application_data', 'verification_images')):
        input_img = preprocess(os.path.join('face_recognition/application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('face_recognition/application_data', 'verification_images', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) >= detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('face_recognition/application_data', 'verification_images'))) 
    print(verification)
    verified = verification >= verification_threshold
    
    return results, verified
    
def predict(photo_data):
    try:
        # print files in current directory
        print(os.listdir())
        siamese_model = tf.keras.models.load_model('face_recognition/siamesemodelv2.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
    except:
        print("Model not found")
        return False

    print("Model loaded")
    
    if not os.path.exists('face_recognition/application_data'):
        os.makedirs('face_recognition/application_data')
    if not os.path.exists('face_recognition/application_data/verification_images'):
        os.makedirs('face_recognition/application_data/verification_images')
    if not os.path.exists('face_recognition/application_data/input_image'):
        os.makedirs('face_recognition/application_data/input_image')

    #  Check that verification images are present
    if len(os.listdir('face_recognition/application_data/verification_images')) == 0:
        print("Verification images not found")
        return False
    
    print("Verification images found")

    base64_data = photo_data.split(",")[1]
    binary_data = base64.b64decode(base64_data)
    
    # Save image in input_image folder as byte data (photo_data is string data)
    with open('face_recognition/application_data/input_image/input_image.jpg', 'wb') as f:
        f.write(binary_data)

    print("Input image saved")

    results, verified = verify(siamese_model, 0.9, 0.7)
    return verified
    
