from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import cv2 
import pickle as pkl
import pandas as pd

# Please put here the image path as : user/folder/images/
# so that the images are inside images folder
# Please put in job_path "in the same "form" as image path " the file df.pkl file


img_path="images/"
#job_path="df.pkl"
final_name = "df_final.pkl"

bboxes_df = pkl.load(open('df.pkl',"rb"))

print(bboxes_df)

ls = os.listdir(img_path)
ls.sort()

bboxes_df['label'] = bboxes_df['label'].apply(lambda x: 1 if x  == 'knife' else 0 )

ls_bboxes = list(bboxes_df.iloc[:,0])

arr_of_arrs = []
temp_1 = []
ls_copy = ls_bboxes
ls_imgs = os.listdir(img_path)
ls_imgs.sort()
for i in ls_imgs:
    temp_1 = []
    for k,j in enumerate(ls_copy):
        if i == j:
            temp_1.append(bboxes_df.iloc[k,1:6].values.astype('int'))
         
    arr_of_arrs.append(temp_1)
    
bboxes_df_copy = bboxes_df.copy()
counter = 0

for c,h in enumerate(ls_imgs):
    if h == "0087a.png":
        continue
    img = cv2.imread(img_path+h)
    bboxes_main = arr_of_arrs[c]
    bboxes_main = np.array(bboxes_main)
    bboxes_main = bboxes_main.astype("float64")
    img_, new_ = RandomHorizontalFlip(1)(img.copy(), bboxes_main.copy())
    ### Here we save image and add thier coordinates to the dataframe
    cv2.imwrite(img_path+h[:-4]+'hf.png',img_)
    #print(img_path+h[:-4]+'hf.png')

    for B in list(list(i) for i in new_):
            list_tobe_appended = [h[:-4]+'hf.png']
            list_tobe_appended = list_tobe_appended + B
            bboxes_df_copy.loc[len(bboxes_df_copy)+1] = list_tobe_appended
    ###
    img_, new_ = RandomScale(0.3, diff = True)(img.copy(), bboxes_main.copy())

    cv2.imwrite(img_path+h[:-4]+'RS.png',img_)

    for B in list(list(i) for i in new_):
            list_tobe_appended = [h[:-4]+'RS.png']
            list_tobe_appended = list_tobe_appended + B
            bboxes_df_copy.loc[len(bboxes_df_copy)+1] = list_tobe_appended
    ###
    img_, new_ = RandomScale(0.4, diff = True)(img.copy(), bboxes_main.copy())
    
    cv2.imwrite(img_path+h[:-4]+'RS2.png',img_)

    for B in list(list(i) for i in new_):
            list_tobe_appended = [h[:-4]+'RS2.png']
            list_tobe_appended = list_tobe_appended + B
            bboxes_df_copy.loc[len(bboxes_df_copy)+1] = list_tobe_appended

    img_, new_ = RandomRotate(20)(img.copy(), bboxes_main.copy())
    
    cv2.imwrite(img_path+h[:-4]+'RR.png',img_)

    for B in list(list(i) for i in new_):
            list_tobe_appended = [h[:-4]+'RR.png']
            list_tobe_appended = list_tobe_appended + B
            bboxes_df_copy.loc[len(bboxes_df_copy)+1] = list_tobe_appended
    ###
    img_, new_ = RandomRotate(45)(img.copy(), bboxes_main.copy())
    
    cv2.imwrite(img_path+h[:-4]+'RR2.png',img_)

    for B in list(list(i) for i in new_):
            list_tobe_appended = [h[:-4]+'RR2.png']
            list_tobe_appended = list_tobe_appended + B
            bboxes_df_copy.loc[len(bboxes_df_copy)+1] = list_tobe_appended
    ###
    img_, new_ = RandomShear(0.2)(img.copy(), bboxes_main.copy())
    
    cv2.imwrite(img_path+h[:-4]+'RS.png',img_)

    for B in list(list(i) for i in new_):
            list_tobe_appended = [h[:-4]+'RS.png']
            list_tobe_appended = list_tobe_appended + B
            bboxes_df_copy.loc[len(bboxes_df_copy)+1] = list_tobe_appended
            
import pickle as pkl
from pickle import Pickler
filename = 'df_final.pkl'
outfile = open(filename,'wb')
pkl.dump(bboxes_df_copy,outfile)
