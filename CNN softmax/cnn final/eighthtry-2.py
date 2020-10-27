import cv2
import tensorflow as tf
import pickle
import numpy as np
# example making new class predictions for a classification problem
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.contrib.keras.python.keras.utils import np_utils
#from spark_sklearn import Converter

CATEGORIES = ["Dog", "Cat"]

DATA2 = "/Users/tejvir/Desktop/doggo.jpg"

def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#x = open('64x3-CNN.pkl','rb')
#model = pickle.load(x) #used to be '64x3-CNN2.pkl','rb'
#model = json.load(open('64x3-CNN.pkl'))
model = tf.keras.models.load_model("64x3-CNN2.model")

#predicSION = model.predict([prepare(DATA2)])


#preds = model.predict_proba([prepare(DATA2)])
#pred_classes = tf.argmax(preds)
#print(preds)
#print(pred_classes)

prediction = model.predict([prepare(DATA2)])  #used to be doggo.jpg
print(prediction)  # will be a list in a list.
#print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict_classes([prepare(DATA2)]) 
print(CATEGORIES[int(prediction)]) #BANG BANG 

import matplotlib.pyplot as plt
plt.imshow(np.squeeze([prepare(DATA2)]))
#plt.imshow( tf.shape( tf.squeeze(DATA2) ) )
plt.show()

#preds3 = tf.nn.sparse_softmax_cross_entropy_with_logits(prediction);
#preds4 = tf.nn.softmax(prediction)
#preds5 = tf.nn.sparse_softmax_cross_entropy_with_logits(preds);
#print(preds3)
   #print(preds4)
#print(preds5)


# y_proba = model.predict([prepare(DATA2)])
# y_classes = tf.keras.np_utils.probas_to_classes(y_proba) #np_
# print(y_classes)


# prediction = model.predict([prepare(DATA2)])  #used to be doggo.jpg
# print(prediction)  # will be a list in a list.
# print(CATEGORIES[int(prediction[0][0])])

# spark = SparkSession.builder.getOrCreate()

# # Convert to sklearn
# converter = Converter(spark.sparkContext)
# sk_model = converter.toSKLearn(model)

# sk_model.coef_ = sk_model.coef_.reshape(1, -1)  
# sk_model.classes_ = np.array([0, 1]) # This is needed!

# predictions = sk_model.predict(np.vstack(features))
# print(predictions)



# # make a prediction
# ynew = model.predict_proba(DATA2)
# X = model.predict_classes(DATA2)

# # show the inputs and predicted outputs
# for i in 2: #len(Xnew)
# 	print("X=%s, Predicted=%s" % (i, ynew[i]))

# for j in 2: #len(Xnew)
# 	print("X=%s, Predicted=%s" % (j, X[i]))

# #model.evaluate(X, y)
# #print model.evaluate(X, y)["accuracy"]
 
# # get the accuracy for each option: 
# numrows = len(prediction)    # 3 rows in your example
# numcols = len(prediction[0]) 
# print(numrows)
# print(numcols)


# #model.evaluate()
# #output = model(DATA2)
# #output = model.evaluate(DATA2)
# #print(output)

# sm = tf.nn.softmax(prediction)

# #probabilities = sm(prediction) #used to be output, now prediction
# print(sm)
