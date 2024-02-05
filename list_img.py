import os
img_path = '../dataset/LeukoX/Fold_5/val/inputs/'

myfile = open('../dataset/LeukoX/Fold_5/val.txt','w')
imfiles = os.listdir(img_path)

for i in imfiles:
    myfile.write(img_path+i+'\n')

myfile.close()
