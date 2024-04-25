import streamlit as st
import tensorflow as tf
import numpy as np
import re
from PIL import Image
import os
from scipy.ndimage import median_filter

# Load your trained model
model_directory = '/Users/liqi/Desktop/23AUTUMNSEM/FYP/chinese_cnn_updated3'
loaded_model = tf.keras.models.load_model(model_directory)

def load_class_names(file_path):
    # A regex pattern to match 'character': index, including possible whitespace
    pattern = re.compile(r"'(.*?)': (\d+),?")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the whole file content at once
        file_content = file.read()
        
    # Find all matches in the file content
    matches = pattern.findall(file_content)

    # Calculate the maximum index to determine the size of the class_names list
    max_index = max(int(index) for _, index in matches)

    # Initialize the list with placeholders
    class_names = [None] * (max_index + 1)

    # Iterate over all matches and store the character at the correct index
    for character, index in matches:
        class_names[int(index)] = character

    return class_names

class_names = load_class_names('/Users/liqi/Desktop/23AUTUMNSEM/FYP/Original_dataset/CASIA-Classes.txt')

def median_filter_wrap(image_np):
    # This function expects a NumPy array as input and returns a NumPy array
    return median_filter(image_np, size=3)

def apply_median_filter(image):
    # Use tf.py_function to wrap the median_filter function
    # tf.py_function requires the operation to be defined in terms of NumPy arrays
    image_filtered = tf.py_function(median_filter_wrap, [image], Tout=tf.float32)
    # Make sure the output has the shape set, since tf.py_function does not infer it automatically
    image_filtered.set_shape(image.shape)
    return image_filtered

def preprocess_image(image_tensor, target_size=(64, 64)):
    # Resize the image to the target size
    image_resized = tf.image.resize(image_tensor, target_size)

    # Convert the image to grayscale if the model expects 1 channel
    image_gray = tf.image.rgb_to_grayscale(image_resized) if image_tensor.shape[-1] == 3 else image_resized

    # Apply median filter to the image
    image_median_filtered = apply_median_filter(image_gray)

    # Normalize the image
    image_normalized = image_median_filtered / 255.0

    # Add a batch dimension (model expects 4D input: [batch_size, height, width, channels])
    image_batch = tf.expand_dims(image_normalized, 0)
    return image_batch

def predict_character_chinese(uploaded_file):
    # Preprocess the image
    preprocessed_image = preprocess_image(uploaded_file)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(preprocessed_image)

    # Get the index of the highest probability
    predicted_index = np.argmax(predictions, axis=1)[0]

    # Retrieve the corresponding character from the class_names
    predicted_character = class_names[predicted_index]
    return predicted_character

# Streamlit code to upload and display the prediction
st.title('Chinese Character Recognition')
uploaded_file = st.file_uploader('Upload an image of a Chinese character', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Convert the uploaded file to an image tensor with 3 channels for color images
    image_tensor = tf.io.decode_image(uploaded_file.read(), channels=3)

    # Preprocess the image tensor
    preprocessed_image_tensor = preprocess_image(image_tensor)

    # Make predictions using the loaded model
    predictions = predict_character_chinese(image_tensor)

    # Display the predictions
    st.write("Predicted Chinese Character:", predictions)






model_directory_english = '/Users/liqi/Desktop/23AUTUMNSEM/FYP/english_hyper_with_regu_median_filter'

def load_and_preprocess_image(image_tensor, target_size=(128, 128)):
    # Resize the image to the target size
    image_resized = tf.image.resize(image_tensor, target_size)
    
    # Normalize the image
    image_normalized = image_resized / 255.0
    return image_normalized

# Directly define your labels
labels = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
}


def predict_character_english(uploaded_file_english, model):
    # Convert the uploaded file to an image tensor with 3 channels for color images
    image_tensor = tf.io.decode_image(uploaded_file_english.read(), channels=3)
    # Preprocess the image tensor
    preprocessed_image_tensor = load_and_preprocess_image(image_tensor)
    # Add batch dimension
    preprocessed_image_tensor = tf.expand_dims(preprocessed_image_tensor, axis=0)
    # Make predictions using the model
    predictions = model.predict(preprocessed_image_tensor)
    # Get the index of the highest probability
    predicted_index = np.argmax(predictions, axis=1)[0]
    # Retrieve the corresponding character from the labels list
    predicted_character = labels[predicted_index]
    return predicted_character

# Define the Streamlit app
st.title('English Character Recognition')
uploaded_file_english = st.file_uploader('Upload an image of an English character', type=['png', 'jpg', 'jpeg'])

# Load the model
loaded_model_english = tf.keras.models.load_model(model_directory_english)

# When a file is uploaded, make a prediction and display the result
if uploaded_file_english is not None:
    predicted_character = predict_character_english(uploaded_file_english, loaded_model_english)
    st.write("Predicted Character:", predicted_character)












MODEL_DIRECTORY_KOREAN = '/Users/liqi/Desktop/23AUTUMNSEM/FYP/korean_best_model'

# Update the preprocess_image function to handle grayscale conversion and normalization
def preprocess_image(image):
    image_pil = Image.open(image).convert('L') 
    image_resized = image_pil.resize((64, 64))
    image_np = np.array(image_resized) / 255.0  # Normalize the image
    image_np = np.expand_dims(image_np, axis=-1)  # Add channel dimension for grayscale
    image_batch = np.expand_dims(image_np, axis=0)  # Add batch dimension
    return image_batch

# Assuming dic is defined as shown in your message
dic = {
    0: ('a', 'ㅏ'), 1: ('ae', 'ㅐ'), 2: ('b', 'ㅂ'), 3: ('bb', 'ㅃ'), 4: ('ch', 'ㅊ'),
    5: ('d', 'ㄷ'), 6: ('e', 'ㅓ'), 7: ('eo', 'ㅕ'), 8: ('eu', 'ㅡ'), 9: ('g', 'ㄱ'),
    10: ('gg', 'ㄲ'), 11: ('h', 'ㅎ'), 12: ('i', 'ㅣ'), 13: ('j', 'ㅈ'), 14: ('k', 'ㅋ'),
    15: ('m', 'ㅁ'), 16: ('n', 'ㄴ'), 17: ('ng', 'ㅇ'), 18: ('o', 'ㅗ'), 19: ('p', 'ㅍ'),
    20: ('r', 'ㄹ'), 21: ('s', 'ㅅ'), 22: ('ss', 'ㅆ'), 23: ('t', 'ㅌ'), 24: ('u', 'ㅜ'),
    25: ('ya', 'ㅑ'), 26: ('yae', 'ㅒ'), 27: ('ye', 'ㅖ'), 28: ('yo', 'ㅛ'), 29: ('yu', 'ㅠ')
}


def predict_korean_character(image):
    # Load the model
    loaded_model_korean = tf.keras.models.load_model(MODEL_DIRECTORY_KOREAN)
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Make predictions using the loaded model
    predictions = loaded_model_korean.predict(preprocessed_image)
    # Get the index of the predicted character
    predicted_index = np.argmax(predictions)
    # Retrieve the corresponding character using the dictionary
    predicted_character = dic[predicted_index]
    return predicted_character


# Streamlit app interface
st.title('Korean Character Recognition')
uploaded_file_korean = st.file_uploader('Upload an image of a Korean character', type=['png', 'jpg', 'jpeg'])

if uploaded_file_korean is not None:
    predicted_character = predict_korean_character(uploaded_file_korean)
    st.write("Predicted Korean Character:", predicted_character)

