from random import shuffle
import copy
from scipy import sparse
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
from sklearn.svm import SVC ,LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from Matrix_computation import LoadVariables
import numpy as np


class CFE_IterativeAdditive():
  def __init__(self,dataset, features):
    self.vars = LoadVariables(dataset, features)
    self.vars.loadVariablesToTrain()
  

  def testingAccuracy(self):
    return self.vars.modelNN.getModel().evaluate(self.vars.TFIDF_TEST, self.vars.LABELS_TEST_ONE)
  def trainMatrix(self,validation= 0.98, splitP = 0.7, n_iterations_local= 400, n_epochs=100, resetValue=False):
    if resetValue:
      #reset to original matrix
      self.vars.TFIDF_CAT_TRAIN = self.vars.TFIDF_CAT.todense()
      self.vars.TF_CAT_TRAIN = self.vars.TF_CAT.todense()
      self.vars.ICF_RAW_TRAIN = copy.deepcopy(self.vars.ICF_RAW.copy())
      self.vars.ICF_TRAIN = copy.deepcopy(self.vars.ICF.copy())
    
    epochs =0
    allowed =list(range(len(self.vars.trainingLabelsIndexes)))
    val = 0
    while val <= validation and epochs < n_epochs:
          epochs+=1
          print('Epoch', epochs)
          shuffle(allowed)
          trainIndexing = allowed[:int(len(allowed)*splitP)]
          for i in trainIndexing:
              #train= np.zeros((1,self.vars.NbrOfColumns))[0]
              indexOfLabel = self.vars.trainingLabelsIndexes[i]
              how=0
              #print(self.vars.TF_TRAIN[i].rows[0], i)
              while(np.argmax(np.sum(self.vars.TFIDF_CAT_TRAIN.T[self.vars.TF_TRAIN[i].rows[0]], axis=0)) !=self.vars.trainingLabelsIndexes[i] ):
                    for ind in self.vars.TF_TRAIN[i].rows[0]:
                        self.vars.TF_CAT_TRAIN[indexOfLabel, ind] +=1
                        if self.vars.TF_CAT_TRAIN[indexOfLabel, ind]==1:
                            self.vars.ICF_RAW_TRAIN[ind]+=1
                            self.vars.ICF_TRAIN[ind] -= math.log10((self.vars.ICF_RAW_TRAIN[ind]+1)/self.vars.ICF_RAW_TRAIN[ind])
                        self.vars.TFIDF_CAT_TRAIN[indexOfLabel, ind] = self.vars.TF_CAT_TRAIN[indexOfLabel, ind]*self.vars.ICF_TRAIN[ind]
                    how+=1
                    if how ==n_iterations_local: break;
          
          t=0; c=0;
          for i in allowed:
                  if i not in trainIndexing:
                      t+=1
                      if(np.argmax(np.sum(self.vars.TFIDF_CAT_TRAIN.T[self.vars.TF_TRAIN[i].rows[0]], axis=0))==self.vars.trainingLabelsIndexes[i]):c+=1
          val = c/t 
          print("Matrix Validation", val)
                  
    self.vars.epochs = epochs


  def generateFeatures(self, norm, axis):
                
        self.vars.TF_CAT_NORMALIZED = self.vars.TF_CAT_TRAIN/(np.max(self.vars.TF_CAT_TRAIN))
        self.vars.TFIDF_CAT_NORMALIZED = np.multiply(self.vars.TF_CAT_TRAIN,self.vars.ICF_TRAIN)
        TFIDF_CAT_O = self.vars.TFIDF_CAT.todense()


        TrainModel =[]
        TrainModelO =[]

        for i in range(self.vars.TF_TRAIN.shape[0]):
          try:
            TrainModel.append(np.append(np.sum(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.TF_TRAIN[i].rows[0]], axis=0),np.max(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.TF_TRAIN[i].rows[0]], axis=0), axis=1))
          except Exception as e:
            print(e)
            TrainModel.append(np.append(np.sum(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.features.index('nan')], axis=0),np.max(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.features.index('nan')], axis=0), axis=1))
        
        for i in range(self.vars.TF_TRAIN.shape[0]):
          try:
            TrainModelO.append(np.append(np.sum(TFIDF_CAT_O.T[self.vars.TF_TRAIN[i].rows[0]], axis=0),np.max(TFIDF_CAT_O.T[self.vars.TF_TRAIN[i].rows[0]], axis=0), axis=1))
          except:
            TrainModelO.append(np.append(np.sum(TFIDF_CAT_O.T[self.vars.features.index('nan')], axis=0),np.max(TFIDF_CAT_O.T[self.vars.features.index('nan')], axis=0), axis=1))
        TrainModel =sparse.lil_matrix(np.array(TrainModel))
        TrainModelO =sparse.lil_matrix(np.array(TrainModelO))
        if norm:
          TrainModelN = normalize(TrainModel ,axis=axis, norm=norm)
          TrainModelNO = normalize(TrainModelO ,axis=axis, norm=norm)
        else:
          TrainModelN = TrainModel 
          TrainModelNO =TrainModelO  


        TestModel =[]
        TestModelO =[]
        for i in range(self.vars.TF_TEST.shape[0]):
            try:
                TestModel.append(np.append(np.sum(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.TF_TEST[i].rows[0]], axis=0),np.max(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.TF_TEST[i].rows[0]], axis=0), axis=1))
            except:
                
                TestModel.append(np.append(np.sum(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.features.index('nan')], axis=0),np.max(self.vars.TFIDF_CAT_NORMALIZED.T[self.vars.features.index('nan')], axis=0), axis=1))
                print(i)

        for i in range(self.vars.TF_TEST.shape[0]):
            try:
                TestModelO.append(np.append(np.sum(TFIDF_CAT_O.T[self.vars.TF_TEST[i].rows[0]], axis=0),np.max(TFIDF_CAT_O.T[self.vars.TF_TEST[i].rows[0]], axis=0), axis=1))
            except:
                
                TestModelO.append(np.append(np.sum(TFIDF_CAT_O.T[self.vars.features.index('nan')], axis=0),np.max(TFIDF_CAT_O.T[self.vars.features.index('nan')], axis=0), axis=1))

                print(i)
        TestModel =sparse.lil_matrix(np.array(TestModel))

        TestModelO =sparse.lil_matrix(np.array(TestModelO))
        if norm:
          TestModelN = normalize(TestModel,axis=axis, norm=norm)
          TestModelNO = normalize(TestModelO,axis=axis, norm=norm)
        else:
          TestModelN = TestModel 
          TestModelNO =TestModelO  
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

