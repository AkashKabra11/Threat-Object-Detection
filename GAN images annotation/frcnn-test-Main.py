import pandas as pd
import os
import re
from sklearn.utils import shuffle
import numpy as np
import shutil
import cv2
import pickle as pkl
from pickle import Pickler
from PIL import Image

#please put images in img path
cwd = os.getcwd()
img_path = os.path.join(cwd,'images/')
print(img_path)
if not os.path.exists('../RPN/train_images'):
    os.mkdir('../RPN/train_images')
train_dir = os.path.join(os.path.dirname(os.getcwd()),'RPN/train_images/')
if not os.path.exists('./test_images'):
    os.mkdir('./test_images')
test_dir  = os.path.join(cwd,'test_images/')

df_v2 = pkl.load(open('df_final.pkl',"rb"))
df_v2['label'] = 'no_cls'
df_v3_shuff = shuffle(df_v2,random_state=42)
df_v3_shuff.reset_index(inplace=True)
df_v3_shuff.drop(['index'],axis=1,inplace=True)
df_v3_shuff = df_v3_shuff[df_v3_shuff['name'].isin(os.listdir(img_path))]
list_names  = list(df_v3_shuff['name'])
list_names  = list(dict.fromkeys(list_names))
num_train = int(len(os.listdir(img_path)))
df_v2.head()
df_v2.to_csv('csv_bboxes.csv',header=False,index=False)
DO_YOU_WANT_TO_COPY = True

if DO_YOU_WANT_TO_COPY:
    
    counter = -1
    for fname in list_names:
        counter += 1 
        if (counter<=num_train):
            src = os.path.join(img_path, fname)
            dst = os.path.join(train_dir, fname)
            shutil.copyfile(src, dst)
        if (counter>num_train):
            src = os.path.join(img_path, fname)
            dst = os.path.join(test_dir, fname)
            shutil.copyfile(src, dst)

 #copy to train and test folders un comment to copy files

list_names_tr = list_names[:num_train+1]
list_names_ts = list_names[num_train+1:]

len(list_names_tr)

df_tr = df_v3_shuff[df_v3_shuff['name'].isin(os.listdir(train_dir))]
df_ts = df_v3_shuff[df_v3_shuff['name'].isin(list_names_ts)]
df_tr.head()

df_ts.reset_index(inplace=True)
df_ts.drop(['index'],axis=1,inplace=True)


# make one true and the rest is false
mk_train_files = True
mk_all_file = True
col_names =  ['filename', 'file_size', 'file_attributes','region_count','region_id','region_shape_attributes','region_attributes']
print("===========pt0=============")
frame1 = pd.DataFrame(columns = col_names)
l = []
for i in range(len(df_tr)):
    l.append(df_tr.iloc[i][0])

freq={}
ans = 0

for items in l:
    freq[items] = l.count(items)
    print("{} {}".format(items,freq[items]))

for key, values in freq.items():
	ans = ans + values
print(ans)


print("==============pt1=================")
frame1.loc[len(frame1)] = col_names
curr={}
	
if mk_train_files:
    for i in range(len(df_tr)):
        ini = df_tr.iloc[i][0]
        if(ini in curr): 
            curr[ini] += 1
        else: 
            curr[ini] = 0
        fname = 'images/'+df_tr.iloc[i][0]
        df_tr.iloc[i][0] = 'train_images/'+df_tr.iloc[i][0]
        x1 = (float)(df_tr.iloc[i][1])
        y1 = (float)(df_tr.iloc[i][2])
        x2 = (float)(df_tr.iloc[i][3])
        y2 = (float)(df_tr.iloc[i][4])
        _cls = df_tr.iloc[i][5]
        #print("{} {} {} {}".format(x1,y1,x2,y2)) 
        size = os.stat(fname).st_size
        s = '{'+'"name":"rect","x":{},"y":{},"width":{},"height":{}'.format(x1, y1, x2-x1, y2-y1)+'}'
        row = [ini, size, '{}', freq[ini], curr[ini], s, _cls]
        frame1.loc[len(frame1)] = row
    frame1.to_csv(os.path.join(os.path.dirname(os.getcwd()),'RPN')+r'/annotate2.csv',header=None, index=None,mode='a')       	
    df_tr.to_csv(os.path.join(os.path.dirname(os.getcwd()),'RPN')+r'/annotate.txt',header=None, index=None,mode='a')
if mk_all_file:
    for i in range(len(df_v3_shuff)):
        df_v3_shuff.iloc[i][0] = 'train_images/'+df_v3_shuff.iloc[i][0]
    df_v3_shuff.to_csv(cwd+r'/all_data.txt',header=None, index=None,mode='a')
    
    




