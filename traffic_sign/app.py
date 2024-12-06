from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Proje klasörünün absolute path'ini al
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Upload klasörü yolunu absolute path olarak belirle
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SINIFLAR = {
    0:'Hız sınırı (20km/s)',
    1:'Hız sınırı (30km/s)', 
    2:'Hız sınırı (50km/s)', 
    3:'Hız sınırı (60km/s)', 
    4:'Hız sınırı (70km/s)', 
    5:'Hız sınırı (80km/s)', 
    6:'Hız sınırı sonu (80km/s)', 
    7:'Hız sınırı (100km/s)', 
    8:'Hız sınırı (120km/s)', 
    9:'Geçiş yasak', 
    10:'3.5 ton üstü araçlar geçemez', 
    11:'Kavşakta geçiş üstünlüğü', 
    12:'Anayol', 
    13:'Yol ver', 
    14:'Dur', 
    15:'Araç giremez', 
    16:'3.5 tondan ağır araç giremez', 
    17:'Giriş yasak', 
    18:'Genel tehlike', 
    19:'Sola tehlikeli viraj', 
    20:'Sağa tehlikeli viraj', 
    21:'Çift yönlü viraj', 
    22:'Kasisli yol', 
    23:'Kaygan yol', 
    24:'Sağdan daralan yol', 
    25:'Yol çalışması', 
    26:'Trafik ışıkları', 
    27:'Yaya geçidi', 
    28:'Okul geçidi', 
    29:'Bisiklet geçidi', 
    30:'Buzlanma tehlikesi',
    31:'Vahşi hayvan geçebilir', 
    32:'Hız ve geçiş sınırı sonu', 
    33:'Sağa dönülecek', 
    34:'Sola dönülecek', 
    35:'İleri git', 
    36:'İleri veya sağa', 
    37:'İleri veya sola', 
    38:'Sağdan git', 
    39:'Soldan git', 
    40:'Ada etrafında dönünüz', 
    41:'Geçiş yasağı sonu', 
    42:'3.5 ton üstü araçlar için geçiş yasağı sonu'
}


MODEL_PATH = os.path.join(BASE_DIR, 'modelim8.h5')
model = load_model(MODEL_PATH)
print(model.summary())

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path)
   
    img = img.convert('RGB')
    img = img.resize((30, 30))  
    img = np.array(img)
    
    img = img / 255.0  
    
    img = np.expand_dims(img, axis=0)  #
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html', prediction=None, image_path=None)
    
    prediction = None
    image_path = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Dosya seçilmedi'
        
        file = request.files['file']
        if file.filename == '':
            return 'Dosya seçilmedi'
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            processed_image = preprocess_image(filepath)
            pred = model.predict(processed_image)
            predicted_class = np.argmax(pred)
            prediction = SINIFLAR[predicted_class]
            
            # Image path'i düzelt
            image_path = filename  # Sadece dosya adını gönder
    
    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    # Upload klasörünü oluştur
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True) 