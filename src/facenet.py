## Import Dependencies
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random

import cv2
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from keras.models import load_model

model= load_model("./input/facenet/facenet_keras.h5")
print(model.inputs, model.outputs, sep='\n')
## Data & Embeddings
face_detector = MTCNN()

def pre_process(img):
    face_data = face_detector.detect_faces(img)
    if len(face_data) == 0:
        return
    x,y,w,h = face_data[0]['box']
    face = img[y:y+h, x:x+w]
    resized_face = cv2.resize(face, (160,160))
    normalized_face = (resized_face - resized_face.mean()) / resized_face.std()
    normalized_face = np.expand_dims(normalized_face, axis=0)
    return normalized_face, face_data

def get_embedding(train_dir):
    count = 0
    anchor_embedding = np.zeros(shape=(1,128))
    print(train_dir.split('/')[-1])
    for img_path in os.listdir(train_dir):
        img_path = os.path.join(train_dir, img_path)
        img = plt.imread(img_path)
        if img.shape[-1] > 3:
            continue
        processed_img, _ = pre_process(img)
        anchor_embedding += model.predict(processed_img)
        count += 1
    anchor_embedding = anchor_embedding/count
    
    return anchor_embedding

known_faces= {}
known_embeddings= []
parent_dir= "./input/5-celebrity-faces-dataset/train"
for i, dir_name in enumerate(os.listdir(parent_dir)):
    known_faces[i] = dir_name.replace("_", " ").title()
    known_embeddings.append(get_embedding(os.path.join(parent_dir, dir_name)))

known_embeddings= np.asarray(known_embeddings)
known_embeddings= np.squeeze(known_embeddings, axis=1)


with open("./dataset/known_embeddings.txt", "wb") as fp:
    pickle.dump(known_embeddings, fp)
with open("./dataset/known_faces.txt", "wb") as fp:
    pickle.dump(known_faces, fp)

with open("./dataset/known_embeddings.txt", "rb") as fp:
    known_embeddings = pickle.load(fp)
with open("./dataset/known_faces.txt", "rb") as gp:
    known_faces = pickle.load(gp)
    # known_faces = np.asarray(pickle.load(gp))

img1= plt.imread("./input/5-celebrity-faces-dataset/train/ben_afflek/httpssmediacacheakpinimgcomxedaedabcbefbcbabbjpg.jpg")
processed_img1, _ = pre_process(img1)
img1_embedding= model.predict(processed_img1)

img2= plt.imread("./input/5-celebrity-faces-dataset/train/madonna/httpiamediaimdbcomimagesMMVBMTANDQNTAxNDVeQTJeQWpwZBbWUMDIMjQOTYVUXCRALjpg.jpg")
processed_img1, _ = pre_process(img2)
img2_embedding = model.predict(processed_img1)

img1_list= [np.linalg.norm(known_embeddings[i] - img1_embedding) for i in range(len(known_embeddings))]
img2_list= [np.linalg.norm(known_embeddings[i] - img2_embedding) for i in range(len(known_embeddings))]

known_faces[np.argmin(img1_list)]
known_faces[np.argmin(img2_list)]

def plot_image(img, id_code, face_data):
    top_left_x, top_left_y, width, height = face_data[0]['box']
#     plot the bounding box on the input image
#     plt.figure(figsize=(6,6))
    plt.imshow(img)
#     (0, 0.8, 0.8) == aquablue
    plt.gca().add_patch(matplotlib.patches.Rectangle((top_left_x,top_left_y), width, height,
                                                     edgecolor=(0,0.8,0.8), facecolor='none', lw=3))
    plt.gca().add_patch(matplotlib.patches.Rectangle((top_left_x,top_left_y+height), width, 0.15*height,
                                                     edgecolor=(0,0.8,0.8), facecolor=(0,0.8,0.8), lw=3, fill=True))
    plt.text(top_left_x+0.05*width, top_left_y+1.1*height, s=known_faces[id_code].split(' ')[0], color='white', size=14, fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    return

def face_recognition(img_path, model, known_faces, known_embeddings):
    img = plt.imread(img_path)
    
    # refusinng RGBA format
    try:
        img.shape[-1] > 3
    except:
        print("Kindly input image with 3 color channels.")
        return
    
    #handling grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    #preprocessing the image
    processed_img, face_data = pre_process(img)
    top_left_x, top_left_y, width, height = face_data[0]['box']
    
    #getting the embeddings and the identity
    embedding = model.predict(processed_img)
    id_code = np.argmin([np.linalg.norm(known_embeddings[i] - embedding) for i in range(len(known_embeddings))])
    
    return img, id_code, face_data



val_img_paths = []
val_face_id_codes = []

val_dir = "./input/5-celebrity-faces-dataset/val"
for i, dir_name in enumerate(os.listdir(val_dir)):
    for img_path in os.listdir(os.path.join(val_dir, dir_name)):
        val_img_paths.append(os.path.join(val_dir, dir_name, img_path))
        val_face_id_codes.append(i)
        
#shuffle
temp = list(zip(val_img_paths, val_face_id_codes))
random.shuffle(temp)
val_img_paths, val_face_id_codes = zip(*temp)
val_img_paths, val_face_id_codes = list(val_img_paths), list(val_face_id_codes)

# the above process converts lists to tuples. Later on, we need to subtract, and subtrctrion betweena a tuple and a list is not allowed. Hence, we need to explicity convert the tuple back into a list
#illustrative example of above code
arr0 = ['ab', 'bc', 'cd', 'de']
arr1 = [0, 1, 2, 3]
print(arr0, arr1)

arr2 = list(zip(arr0, arr1))
random.shuffle(arr2)
arr0, arr1 = zip(*arr2)
arr0, arr1 = list(arr0), list(arr1)
print(arr0, arr1)

predicted_codes= []
for img_path in val_img_paths:
    img, id_code, fd = face_recognition(img_path, model, known_faces, known_embeddings)
    predicted_codes.append(id_code)

predicted_codes

np.sum(np.abs(np.asarray(val_face_id_codes) - np.asarray(predicted_codes)))

known_faces[predicted_codes[0]], known_faces[val_face_id_codes[0]]

conf_mat = tf.math.confusion_matrix(labels=val_face_id_codes, predictions=predicted_codes)
print(conf_mat)
# import seaborn as sns
sns.heatmap(conf_mat, annot=True, linewidth=1, linecolor='w')
plt.show()

add_file= os.listdir(os.path.join('input', 'additional-data'))

img_path1 = f"./input/additional-data/{add_file[0]}"
img1, id1, fd1 = face_recognition(img_path1, model, known_faces, known_embeddings)

img_path2 = f"./input/additional-data/{add_file[1]}"
img2, id2, fd2 = face_recognition(img_path2, model, known_faces, known_embeddings)

img_path3 = f"./input/additional-data/{add_file[2]}"
img3, id3, fd3 = face_recognition(img_path3, model, known_faces, known_embeddings)

plt.figure(figsize=(20, 10))
plt.subplot(1,3,1)
plot_image(img1, id1, fd1)
plt.subplot(1,3,2)
plot_image(img2, id2, fd2)
plt.subplot(1,3,3)
plot_image(img3, id3, fd3)
plt.show()