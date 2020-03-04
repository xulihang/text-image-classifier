import os
from shutil import copyfile

def copy(trainList,validationList,isText):
    index=0
    if isText==True:
        dirname="text"
    else:
        dirname="nontext"
    for filename in trainList:
        savedName=dirname+"."+str(index)+".jpg"
        copyfile("./"+dirname+"/"+filename,"./train/"+dirname+"/"+savedName)
        index=index+1
    for filename in validationList:
        savedName=dirname+"."+str(index)+".jpg"
        copyfile("./"+dirname+"/"+filename,"./validation/"+dirname+"/"+savedName)
        index=index+1        
        
textIndex=0
nontextIndex=0
for dirname in os.listdir("./"):
    if dirname=="nontext":
        os.mkdir("./train/nontext")
        os.mkdir("./validation/nontext")
        num=len(os.listdir("./"+dirname))
        trainList=[]
        validationList=[]
        for filename in os.listdir("./"+dirname):
            if str(nontextIndex)[-1]==str(5) or str(nontextIndex)[-1]==str(0):
                validationList.append(filename) 
            else:
                trainList.append(filename)
            nontextIndex=nontextIndex+1
        copy(trainList,validationList,False)            
    elif dirname=="text":
        os.mkdir("./train/text")
        os.mkdir("./validation/text")
        num=len(os.listdir("./"+dirname))
        trainList=[]
        validationList=[]
        for filename in os.listdir("./"+dirname):
            savedName="text."+str(textIndex)+".jpg"
            if str(textIndex)[-1]==str(5) or str(textIndex)[-1]==str(0):
                validationList.append(filename) 
            else:
                trainList.append(filename)
            textIndex=textIndex+1    
        copy(trainList,validationList,True)    