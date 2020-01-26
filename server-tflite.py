import tensorflow as tf, sys
import os
import numpy as np
import cv2
from bottle import route, run, template, request, static_file
import zipfile
import time
import asyncio
import shutil

def imgData(filename):
    image = cv2.imread(filename)
    h=image.shape[0]
    w=image.shape[1]
    centerX=int(w/2)
    centerY=int(h/2)
    if h/w>1.5:
        image=image[max(0,int(centerY-w/2)):min(h-1,int(centerY+w/2))]
    if w/h>1.5:
        image=image[:,max(0,int(centerX-h/2)):min(w-1,int(centerX+h/2))]
    #cv2.imwrite("cropped.jpg",image)
    data=cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)
    input_data=data[np.newaxis,:,:,:]
    return input_data

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


async def predict(path,imgDict,filename):
    result={}        
    # Load from a file
    input_data=imgData(path)
    if floating_model:
        input_data = (np.float32(input_data) - 0) / 255
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels("class_labels.txt")
    preds=[]
    for i in top_k:
        pred={}
        probablity=1.0
        if floating_model:
            probablity=float(results[i])
        else:
            probablity=float(results[i] / 255.0)
        pred["tagName"]=labels[i]
        pred["probablity"]=probablity
        preds.append(pred)    
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
    savedName=name+"-"+timestamp+ext
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


labels = []    
interpreter = tf.lite.Interpreter(model_path="comics.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32
# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
new_size = (height, width)

#loop = asyncio.get_event_loop() 
run(server="paste",host='0.0.0.0', port=8082)    
