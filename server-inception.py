import tensorflow as tf, sys
import os
from PIL import Image
import numpy as np
import cv2
from bottle import route, run, template, request, static_file
import zipfile
import time
import asyncio
import shutil

def loadModel():
    filename = "comics_graph.pb"
    labels_filename = "comics_labels.txt"
    # Loads label file, strips off carriage return
    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())
    graph_def = tf.compat.v1.GraphDef()
    f=open(filename,"rb")
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


async def predict(path,imgDict,filename):
    result={}        
    # Load from a file
    f=open(path,"rb")
    image_data = f.read()
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    #print(predictions)
    prediction=predictions[0]
    #print(labels)
    label_index = 0
    preds=[]
    for p in prediction:
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
loadModel()    
sess=tf.compat.v1.Session()
#loop = asyncio.get_event_loop() 
run(server="paste",host='0.0.0.0', port=8082)    
