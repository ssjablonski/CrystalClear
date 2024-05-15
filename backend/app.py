from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Sprawdź, czy katalog uploads istnieje, jeśli nie, utwórz go
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Załaduj model
model = load_model('models/ADAM.keras')
# model = load_model('models/ADAM.keras')
class_names = ['amethyst', 'ametrine', 'aquamarine', 'black_onyx', 'blue_sapphire', 'citrine', 'diamond', 'emerald', 'lapis_lazuli', 'obsydian', 'pink_sapphire', 'quartz_clear', 'quartz_smoky', 'ruby', 'turquoise']
# Dozwolone rozszerzenia plików
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Przygotowanie obrazu do predykcji
        img = image.load_img(filepath, target_size=(96, 96))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # # Predykcja
        prediction = model.predict(img_array)
        top_three_indices = np.argsort(prediction[0])[-3:]  # get top 3 predictions
        predicted_classes = [class_names[i] for i in top_three_indices]
        # Zwróć wynik
        return jsonify({'prediction': predicted_classes[::-1]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
