from Matrix_computation import LoadVariables
import numpy as np
from scipy.sparse import hstack, vstack
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, LSTM,RepeatVector,TimeDistributed , InputLayer
import tensorflow as tf
import keras.backend as K
from keras.constraints import Constraint
from keras.callbacks import EarlyStopping
from scipy.spatial.distance import cosine


class CFE_FeedForwardANN():
  def __init__(self,dataset, features):
    self.vars = LoadVariables(dataset, features)
    self.vars.loadVariablesToTrain()


  def buildNetwork(self, loss):
    self.vars.loss = loss
    def cos(x, y):    
        n_x = tf.norm(x)      
        n_y = tf.norm(y)
        n=0
        if n_x > n_y:
          n= n_x
        else:
          n = n_y
        diff = (x-y)/tf.maximum(n_x, n_y) #tf.norm(x-y)
        return tf.multiply(diff,diff)-tf.reduce_sum(tf.multiply(x,y))/ (n_x*n_y)+1
    
    class CosineModel():
      def __new__(self, input, output, loss, activationFunction="sigmoid"):
          self.model = Sequential()
          self.loss = loss
          # self.model.add(Dense(int(input.shape[1]/100), input_dim=input.shape[1], activation=activationFunction))
          self.model.add(Input(shape=(input.shape[1],), sparse=True))

          self.model.add(Dense(np.sqrt(2)*output.shape[1], activation=activationFunction))
          self.model.add(Dropout(0.2))
          self.model.add(Dense(output.shape[1],  activation=activationFunction))
          if self.loss =='cosine':
            self.model.compile(loss=cos, optimizer='adam', metrics=[ 'accuracy'])
          else:
            self.model.compile(loss=self.loss, optimizer='adam', metrics=[ 'accuracy'])
          return self.model
    
      def getModel(self):
          return self.model  
    
    self.model = CosineModel(self.vars.TFIDF_TRAIN, self.vars.LABELS_TRAIN_MAPPED, self.vars.loss, 'relu')
 
  def train(self, n_epochs , batch_size=1):
      
      usualCallback = EarlyStopping()
      overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 20)
      self.model_input = vstack((self.vars.TF_TRAIN, self.vars.TFIDF_TRAIN))
      self.model_output = np.concatenate((self.vars.LABELS_TRAIN_MAPPED, self.vars.LABELS_TRAIN_MAPPED), axis=0)
      self.model.fit(
            self.model_input, self.model_output,
            epochs=n_epochs,
            batch_size=batch_size, callbacks=[overfitCallback])
  def predict(self, split):
    if split=='train':
      return self.model.predict(self.vars.TFIDF_TRAIN)
    else:
      return self.model.predict(self.vars.TFIDF_TEST)
    
      
  def score(self):
    mappingNewTest = self.predict("test")
    keys=  list(self.vars.mapping.keys())
    accuracy =[]
    labels_predicted=[]
    for i in range(self.vars.TFIDF_TEST.shape[0]):
      label =keys[np.argmax([1-cosine(mappingNewTest[i], self.vars.mapping[label]) if str(1-cosine(mappingNewTest[i], self.vars.mapping[label])) != 'nan' else 0 for label in self.vars.mapping.keys() ])]
      accuracy.append(1 if label==self.vars.LABELS_TEST[i] else 0 )
      labels_predicted.append(label)
    return np.mean(accuracy), labels_predicted
