import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time
from inference_sdk import InferenceHTTPClient

# Load the fruit and vegetable classification model


st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #00000;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("")

model = load_model('FV1.h5')

# Define labels, fruits, vegetables, and calorie dictionary
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalapeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']

vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Radish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

calories_dict = {'Apple': 25, 'Banana': 89, 'Bell Pepper': 20, 'Chilli Pepper': 40, 'Grapes': 67, 'Jalapeno': 28, 'Kiwi': 61, 'Lemon': 29, 'Mango': 60, 'Orange': 47,
                 'Paprika': 282, 'Pear': 57, 'Pineapple': 50, 'Pomegranate': 68, 'Watermelon': 30, 'Beetroot': 43, 'Cabbage': 25, 'Capsicum': 40, 'Carrot': 41,
                 'Cauliflower': 25, 'Corn': 86, 'Cucumber': 15, 'Eggplant': 25, 'Ginger': 80,
                 'Lettuce': 15, 'Onion': 40, 'Peas': 81, 'Potato': 104, 'Radish': 16, 'Soy Beans': 446, 'Spinach': 23, 'Sweetcorn': 86, 'Sweetpotato': 86,
                 'Tomato': 18, 'Turnip': 28}

# Initialize the freshness prediction client
CLIENT = InferenceHTTPClient(
    api_url="http://detect.roboflow.com",
    api_key="q303WGUb9vRdxBHyxWiS"
)

def prepare_image(frame):
    img = cv2.resize(frame, (224, 224))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(y_class)
    res = labels[y]
    return res.capitalize()

def fetch_cal(pred):
    try:
        calory = calories_dict[pred]
        return calory
    except KeyError:
        return 'Unknown'

def run():
    st.title("Freshness and Calorie Detection")

    # Radio options on the right side
    option = st.sidebar.radio("Choose an option to input an image:", ["Webcam", "Upload"])

    if option == "Webcam":
        st.write("Click the button to capture an image from the webcam:")
        if st.button("Capture Image from Webcam"):
            video_capture = cv2.VideoCapture(0)
            st.write("Camera open. Get ready! The image will be captured in 5 seconds.")

            video_placeholder = st.empty()
            for i in range(5, 0, -1):
                video_placeholder.image(video_capture.read()[1], channels="BGR")
                st.write(f"Image will be captured in {i} seconds.")
                time.sleep(1)

            ret, frame = video_capture.read()
            video_capture.release()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            st.image(img, use_column_width=False)

            result = prepare_image(frame)
            if result in vegetables:
                st.info('Category: Vegetables')
            else:
                st.info('Category: Fruit')
            st.success("Predicted: " + result + '')
            cal = fetch_cal(result)
            st.warning(f'* {cal} (100 grams)*')

            freshness_result = CLIENT.infer(img, model_id="fvr/1")
            st.write("Freshness Prediction:", freshness_result)

    elif option == "Upload":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            img_resized = img.resize((720, 400))
            st.image(img_resized, use_column_width=False)

            frame = np.array(img_resized)
            result = prepare_image(frame)

            if result in vegetables:
                st.info('Category: Vegetables')
            else:
                st.info('Category: Fruit')
            st.success("Predicted: " + result + '')
            cal = fetch_cal(result)
            st.warning(f'* {cal} (100 grams)*')

            freshness_result = CLIENT.infer(img, model_id="fvr/1")
            st.write("Freshness Prediction:", freshness_result)

    # Predict button
    if st.button("Predict"):
        st.write("Click the 'Predict' button to see freshness prediction.")
        # You can add your freshness prediction logic here and display the result.

if __name__ == "__main__":
    run()