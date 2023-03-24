from urllib import request
from flask import Flask,render_template,request,jsonify
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from keras import applications  
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from datetime import datetime 

tf.get_logger().setLevel('INFO')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images'

model=tf.keras.models.load_model('animal10.hdf5', compile=False)
 


'''
     f = request.files['image']
    print("current path")
    # This extracts the filepath of the image uploaded
    basepath = os.path.dirname(__file__)
    print("current path", basepath)
    # This appends the original filepath to that of uploads
    filepath = os.path.join(basepath,'/images',f.filename)
    print("upload folder is ", filepath)
    # This saves the filepath of the image
    f.save(filepath)

'''

@app.route('/',methods=['GET'])
def home():
    #imagefile=request.files['imagefile']
    #image_path = "./images/" + imagefile.filename
    #imagefile.save(image_path)
       
        
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    name = str(datetime.now().microsecond) + str(datetime.now().month) + '-' + str(datetime.now().day) +  '.jpg'
    photo = request.files['image']
    path = os.path.join(app.config['UPLOAD_FOLDER'],name)
    photo.save(path)
        
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'],name)    
    
    # Testing the model

    img = image.load_img(filepath,target_size = (224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis =0)
    result_b = model.predict(x)  
    
    
        
    #print("prediction",preds)
            
    index = translate = ["Cat","Cow", "Dog",  "Elephant","Hen", "Sheep"]
        


        
    result = translate[np.argmax(result_b[0])]

    os.unlink(filepath)

    return jsonify(result)

if __name__ =='__main__':
    app.run(debug=False,host='0.0.0.0')
