import tensorflow as tf
import os
from PIL import Image
import numpy as np
import cv2
from bottle import route, run, template, request, static_file
import zipfile
import time
import asyncio
import shutil

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image  

def loadModel():
    print("loading")
    graph_def = tf.compat.v1.GraphDef()
    # These are set to the default names from exported models, update as needed.
    filename = "model.pb"
    labels_filename = "labels.txt"

    # Import the TF graph
    f=open("model.pb","rb")
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    # Create a list of labels.
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())
      
    print("done")

    
async def predict(path,imgDict,filename):
    result={}        
    # Load from a file

    imageFile = path
    image = Image.open(imageFile)

    # Update orientation based on EXIF tags, if the file has orientation info.
    image = update_orientation(image)

    # Convert to OpenCV format
    image = convert_to_opencv(image)      

    # We next get the largest center square
    h, w = image.shape[:2]
    min_dim = min(w,h)
    max_square_image = crop_center(image, min_dim, min_dim)

    # Resize that square down to 256x256
    augmented_image = resize_to_256_square(max_square_image)

    # Get the input size of the model
    input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
    network_input_size = input_tensor_shape[1]

    # Crop the center for the specified network_input_Size
    augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

    # These names are part of the model and cannot be changed.
    output_layer = 'loss:0'
    input_node = 'Placeholder:0'

    try:
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        predictions, = sess.run(prob_tensor, {input_node: [augmented_image] })
    except KeyError:
        print ("Couldn't find classification output layer: " + output_layer + ".")
        print ("Verify this a model exported from an Object Detection project.")
        return result
            
    # Print the highest probability label
    #highest_probability_index = np.argmax(predictions)
    #print('Classified as: ' + labels[highest_probability_index])
    
    # Or you can print out all of the results mapping labels to probabilities.
    label_index = 0
    preds=[]
    for p in predictions:
        pred={}
        truncated_probablity = np.float64(np.round(p,8))
        #print(labels[label_index], truncated_probablity)
        pred["tagName"]=labels[label_index]
        pred["probablity"]=truncated_probablity
        preds.append(pred)
        label_index += 1
    result["preds"]=preds
    imgDict[filename]=result    

@route('/classify', method='POST')
def classify():
    #username = request.forms.get('username')
    upload = request.files.get('upload')
    name, ext = os.path.splitext(upload.filename)

    if ext.lower() not in ('.png','.jpg','.jpeg','.zip'):
        return "File extension not allowed."
    timestamp=str(int(time.time()*1000))
    savedName=timestamp+ext
    save_path = "./uploaded/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = "{path}/{file}".format(path=save_path, file=savedName)
    if os.path.exists(file_path)==True:
        os.remove(file_path)
    upload.save(file_path)  

    loop =  asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    imgDict={}
    if ext.lower()==".zip":
        tasks=[]
        zipDst_path = "./uploaded/"+timestamp
        if not os.path.exists(zipDst_path):
            os.makedirs(zipDst_path)
        unzip_single(file_path,zipDst_path)
        for filename in os.listdir(zipDst_path):
            name, ext = os.path.splitext(filename)
            if ext.lower() in ('.png','.jpg','.jpeg'):
                img_path = "{path}/{file}".format(path=zipDst_path, file=filename)
                #print(img_path)
                tasks.append(predict(img_path,imgDict,filename))
                #loop.run_until_complete(predict(img_path,imgDict,filename))
        loop.run_until_complete(asyncio.wait(tasks))
        shutil.rmtree(zipDst_path)
    else:
        #print(file_path)
        loop.run_until_complete(predict(file_path,imgDict,upload.filename))
    loop.close()
    os.remove(file_path)
    
    return imgDict


def unzip_single(src_file, dest_dir):
    zf = zipfile.ZipFile(src_file)
    try:
        zf.extractall(path=dest_dir)
    except RuntimeError as e:
        print(e)
    zf.close()
    
@route('/<filepath:path>')
def server_static(filepath):
    return static_file(filepath, root='www')

@route('/test')
def test():
    start=time.time()
    imgDict={}
    loop =  asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks=[]
    for filename in os.listdir("./uploaded/"):
        name, ext = os.path.splitext(filename)
        if ext.lower() in ('.png','.jpg','.jpeg'):
            file_path="./uploaded/"+filename
            #tasks.append(predict(file_path,imgDict,filename))
            loop.run_until_complete(predict(file_path,imgDict,filename)) 
            
    #loop.run_until_complete(asyncio.wait(tasks))            
    end=time.time()
    print(end-start)
    loop.close()
    return imgDict
    
labels = []    
loadModel()    
sess=tf.compat.v1.Session()
#loop = asyncio.get_event_loop() 
run(server="paste",host='127.0.0.1', port=8081)    