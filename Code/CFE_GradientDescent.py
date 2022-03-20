from Matrix_computation import LoadVariables
from random import shuffle
from scipy.sparse import hstack
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, LSTM,RepeatVector,TimeDistributed , InputLayer
from random import shuffle
import copy
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, normalize
import keras.backend as K
from keras.constraints import Constraint

class NeuralNetwork():
  def __init__(self, input, output,activationFunction="softmax", validation=False):
      
      from keras.callbacks import EarlyStopping
      usualCallback = EarlyStopping()
      overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 2)
      self.input = input
      self.validation = validation
      self.output = output
      model = Sequential()
      model.add(InputLayer(input_shape=(input.shape[1],), sparse=True))
      model.add(Dense(output.shape[1], activation='softmax')) #sigmoid
      model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
      self.modelS =model

  def getModel(self):
      return self.modelS   
  def train(self, epochs, batch_size):

    from keras.callbacks import EarlyStopping
    usualCallback = EarlyStopping()
    overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 2)
   
    validation_split=0.33
    self.modelS.fit(self.input, self.output,validation_split=validation_split, epochs =epochs,  batch_size=batch_size  , callbacks=[overfitCallback])

def getModel(input, output):
  model = Sequential()
  model.add(Dense(output.shape[1],input_dim=input.shape[1],  activation='softmax')) #sigmoid
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
  return model

class CFE_GradientDescent():
  def __init__(self,dataset, features):
    self.vars = LoadVariables(dataset, features)
    self.vars.loadVariablesToTrain()
  

  def testingAccuracy(self):
    return self.vars.modelNN.getModel().evaluate(self.vars.TFIDF_TEST, self.vars.LABELS_TEST_ONE)

  def trainMatrix(self, n_epochs=100,batch_size = 20,  resetValue=False):
    #get all labels to initialize the One Hot Encoder
    labels =np.append(np.array(self.vars.LABELS_TRAIN),np.array(self.vars.LABELS_TEST))
    if hasattr(self.vars.dataset, 'validation_csv') : labels =  np.append( (labels ,np.array(self.vars.LABELS_VAL)))
    labels = labels.reshape(-1, 1)

    #convert labels to 1 hot encoding format
    self.vars.oneHot = OneHotEncoder()
    self.vars.oneHot.fit(labels)
    self.vars.LABELS_TRAIN_ONE = self.vars.oneHot.transform(np.array(self.vars.LABELS_TRAIN).reshape(-1, 1)).todense()
    self.vars.LABELS_TEST_ONE = self.vars.oneHot.transform(np.array(self.vars.LABELS_TEST).reshape(-1, 1)).todense()

    #construct neural network based on whether we have a validation preset
   
    self.vars.modelNN = NeuralNetwork(self.vars.TFIDF_TRAIN.todense(),  self.vars.LABELS_TRAIN_ONE, validation=False ) 
    
    #initiate the training process
    self.vars.modelNN.train(n_epochs, batch_size)

  def generateFeatures(self, norm, axis):
        #get the original TFICF matrix
        TFIDF_CAT_O = self.vars.TFIDF_CAT.todense()

        #extract the weight of the network trained then normalize (this represents the updated TFICF matrix)
        self.vars.TFIDF_CAT_NORMALIZED = self.vars.modelNN.getModel().layers[-1].get_weights()[0].T
        self.vars.TFIDF_CAT_NORMALIZED = normalize( self.vars.TFIDF_CAT_NORMALIZED , axis=0, norm='l1')

        #extract the new features that will be used to train the SVM

        #extract the features from the original matrix (features are sum and max)
        TrainModel =[]
        TrainModelO =[]

        for i in range(self.vars.TF_TRAIN.shape[0]):
          try:
            TrainModel.append(np.append(np.sum(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.TF_TRAIN[i].rows[0]], axis=0),np.max(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.TF_TRAIN[i].rows[0]], axis=0)))
          except Exception as e:
            print(e)
            TrainModel.append(np.append(np.sum(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.features.index('nan')], axis=0),np.max(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.features.index('nan')], axis=0)))
       
        for i in range(self.vars.TF_TRAIN.shape[0]):
          try:
            TrainModelO.append(np.append(np.sum(TFIDF_CAT_O.T[self.vars.TF_TRAIN[i].rows[0]], axis=0),np.max(TFIDF_CAT_O.T[self.vars.TF_TRAIN[i].rows[0]], axis=0), axis=1))
          except:
            TrainModelO.append(np.append(np.sum(TFIDF_CAT_O.T[self.vars.features.index('nan')], axis=0),np.max(TFIDF_CAT_O.T[self.vars.features.index('nan')], axis=0), axis=1))
        
        #extract the features from the trained matrix (features are sum and max)
       
        TrainModel =sparse.lil_matrix(np.array(TrainModel))
        TrainModelO =sparse.lil_matrix(np.array(TrainModelO))

        #normalize the matrices upon request
        if norm:
          TrainModelN = normalize(TrainModel ,axis=axis, norm=norm)
          TrainModelNO = normalize(TrainModelO ,axis=axis, norm=norm)
        else:
          TrainModelN = TrainModel 
          TrainModelNO =TrainModelO  


        ## extract the same features for the test set (to be able to compute the scores on the test set)
        ### then normalize
      
        TestModel =[]
        TestModelO =[]

        for i in range(self.vars.TF_TEST.shape[0]):
            try:
                TestModel.append(np.append(np.sum(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.TF_TEST[i].rows[0]], axis=0),np.max(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.TF_TEST[i].rows[0]], axis=0)))
            except:                
                TestModel.append(np.append(np.sum(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.features.index('nan')], axis=0),np.max(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.features.index('nan')], axis=0)))
        for i in range(self.vars.TF_TEST.shape[0]):
            try:
                TestModelO.append(np.append(np.sum(TFIDF_CAT_O.T[self.vars.TF_TEST[i].rows[0]], axis=0),np.max(TFIDF_CAT_O.T[self.vars.TF_TEST[i].rows[0]], axis=0), axis=1))
            except:
                TestModelO.append(np.append(np.sum(TFIDF_CAT_O.T[self.vars.features.index('nan')], axis=0),np.max(TFIDF_CAT_O.T[self.vars.features.index('nan')], axis=0), axis=1))
        
        TestModel =sparse.lil_matrix(np.array(TestModel))
        TestModelO =sparse.lil_matrix(np.array(TestModelO))
        
        

        if norm:
          TestModelN = normalize(TestModel,axis=axis, norm=norm)
          TestModelNO = normalize(TestModelO,axis=axis, norm=norm)
        else:
          TestModelN = TestModel 
          TestModelNO =TestModelO  

    
        #save the features
        self.features_train_1 =TrainModelN
        self.features_test_1 =TestModelN

        self.features_train_2 =TrainModelNO
        self.features_test_2 = TestModelNO 
  def train(self, model, features1, features2):
    #train the SVM
    train =[]
    test=[]

    #concatenate the tfidf with teh available extracted features
    if features1 and not features2:
        train = hstack((self.vars.TFIDF_TRAIN, self.features_train_1))
        test = hstack((self.vars.TFIDF_TEST, self.features_test_1))
        
    elif features1 and features2:
        train = hstack((self.vars.TFIDF_TRAIN,self.features_train_1, self.features_train_2))
        test = hstack((self.vars.TFIDF_TEST, self.features_test_1, self.features_test_2))
    elif not features1 and features2:
        train = hstack((self.vars.TFIDF_TRAIN, self.features_train_2))
        test = hstack((self.vars.TFIDF_TEST, self.features_test_2))
    else:
        train = self.vars.TFIDF_TRAIN
        test =self.vars.TFIDF_TEST     
   
    #fit the SVM on the data and evaluate the model    
    model.fit(train, self.vars.LABELS_TRAIN)
    results = model.predict(test)

    print("Testing", np.mean(results==self.vars.LABELS_TEST))

    return model, results,  np.mean(results==self.vars.LABELS_TEST)

