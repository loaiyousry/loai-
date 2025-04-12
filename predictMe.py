from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

model = tf.keras.models.load_model('myModel.h5') # the path of the model

df = pd.read_csv('p4.csv')  # the path of the csv file

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/Loai', methods=['POST'])
def Loai():
    return "Hello World"
    
@app.route('/predict', methods=['POST'])
def predict():
    # return "Hello World"

    # check if the post request has the file part
    if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)
    image_file = request.files['file']

    # image_file = request.files['image']
    # return "www"
    image_path = "./" + image_file.filename
    image_file.save(image_path)
    print("image_path:", image_path)
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    
    top3_indices = np.argsort(prediction[0])[-3:][::-1]
    top3_class_names = [df.iloc[i]['Label'] for i in top3_indices]
    top3_scores = prediction[0][top3_indices]
    top3_percentages = top3_scores / np.sum(top3_scores) * 100
    
    response = {}
    for i in range(3):
        index = top3_indices[i]
        treatment = df.iloc[index]['Treatment']
        if pd.isna(treatment):
            treatment = "No treatment needed"  

        response[f"prediction_{i+1}"] = {
            "class_name": top3_class_names[i],
            "confidence": f"{top3_percentages[i]:.2f}%",
            "example_picture": df.iloc[index]['Example Picture'],
            "description": df.iloc[index]['Description'],
            "prevention": df.iloc[index]['Prevention'],
            "treatment": treatment
        }
    
    return jsonify(response)
    # return top3_class_names[0]


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)