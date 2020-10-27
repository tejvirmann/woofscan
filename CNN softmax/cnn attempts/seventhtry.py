import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tqdm import tqdm #this used to be commented
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

DATADIR = "/Users/tejvir/Documents/PetImages"

CATEGORIES = ["Dog", "Cat"]

print("Hello1")

for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        #plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!
    
    #print(img_array)   #why commented? So it takes less time from here
    #print(img_array.shape)
    
    ##

    #print(img_array)
    #print(img_array.shape)  # to here 
    
    ##
print("Hello1.5")
IMG_SIZE = 100
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
print("Hello1.6")
#plt.show()

	##

print("Hello2")	

training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats
        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

##
print("Hello3")
print(len(training_data))  ##should print out 24946

#RANDOMIZE 

import random
random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
    
print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

##DONE RANDOMIZE

X = X/255.0

model = Sequential()

# define the model, copy paste 

model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('softmax')) #used to be sigmoid, now, softmax

# done defining the model 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #used to be 'binary_crossentropy' #trying sparse_categorical_crossentropy

#fixing model.fit 
#model.fit(X, y, epochs=1, batch_size=10)
#model.fit([data_a, data_b], y, batch_size=2, epochs=10)
#model.fit({'input_x': X}, y, batch_size=2, epochs=10)

X= np.asarray(X) # this converts X and y to arrays so .fit can work 
y= np.asarray(y)

model.fit(X, y, batch_size = 500, epochs = 1)
#model.fit(X, y, batch_size=32, epochs=4)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

model.save('64x3-CNN2.model')
#OR 
#pickle.dump([model], open('64x3-CNN2.pkl', 'wb' ))

# if the above doesnt work, then uses whats below: 
#from sklearn.externals import joblib 
# Save the model as a pickle in a file 
#joblib.dump(model, '64x3-CNN.pkl')

#NEXT
# convert to the number example. 
# save the trained model with pickle
# in a different file, open it, and start using it. Should output the percent probability of each possibility. (9)
# Make a website: So a user can drag and drop a jpg, then it is taken through the model, then the output is displayed nicely on 
# a website 
# at this point, now you need the actual data, so 1. find the list of diseases you are testing for. 

#CATEGORIES = ["Dog", "Cat"]

#DATA2 = "/Users/tejvir/Desktop/doggo.jpg"

#def prepare(filepath):
 #   IMG_SIZE = 70  # 50 in txt-based
 #   img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
 #   img_array = img_array/255
#    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    # may need to img_array = img_array/255
 #   return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


#model = tf.keras.models.load_model("64x3-CNN.model")
#model2 = pickle.load(open('64x3-CNN2.pkl','rb'))

#prediction = model.predict([prepare(DATA)])  #used to be doggo.jpg
#print(prediction)  # will be a list in a list.
#print(CATEGORIES[int(prediction[0][0])])

#model.evaluate(X, y)
#print model.evaluate(X, y)["accuracy"]
 
# get the accuracy for each option: 
#model.evaluate()
#output = model(DATA)
#print(output)

#sm = tf.nn.softmax()
#probabilities = sm(output)
#print(probabilities)

#import pickle
#pickle.dump(model,'64x3-CNN2.pkl' )

#model2 = pickle.load(open('64x3-CNN2.pkl','rb'))
print("hellodone")

