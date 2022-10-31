Reference
1. FaceNet: A Unified Embedding for Face Recognition and Clustering: https://arxiv.org/pdf/1503.03832.pdf
2. https://www.kaggle.com/code/armgaan/face-recognition-using-mtcnn-and-facenet/notebook
3. model download: https://www.kaggle.com/datasets/suicaokhoailang/facenet-keras?resource=download
4. dataset download: https://www.kaggle.com/datasets/dansbecker/5-celebrity-faces-dataset

Pseucode
1. Preprocess the input image using MTCNN.
    Return the processed image and the face data.
2. Get embeddings using FaceNet.
3. In order to make a prediction, pass the image through the model.
   Choose anchor embedding with the least distance to predicted embedding as the prediction.
4. Use the predicted embdedding's index to get the identity.
5. Use the face data to draw bounding box on original image and add identity as text below it.

Dependencies
- python 3.7
- virtual environment: conda

- pandas
- numpy
- matplotlib
- seaborn
- ipykernel
- opencv-python
- mtcnn 
- scikit-learn
- tensorflow
- keras