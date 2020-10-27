from sklearn.externals import joblib 
import tensorflow as tf
  
# Load the model from the file 
model_loaded = joblib.load('64x3-CNN.pkl')  
  
# Use the loaded model to make predictions 
DATA = "/Users/tejvir/Desktop/doggo.jpg"
prediction = model_loaded.predict(DATA) 
print(prediction)
print(CATEGORIES[int(prediction[0][0])])

#model_loaded.evaluate(X, y)
#print model_loaded.evaluate(X, y)["accuracy"]

# now same as the other: get the crossentropy list, then get the probs. 

model_loaded.evaluate()
output = model_loaded(DATA)
print(output)

sm = tf.nn.softmax()
probabilities = sm(output)
print(probabilities)
