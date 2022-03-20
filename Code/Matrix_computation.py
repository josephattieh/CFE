from Datasets import ProcessDocuments, Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import copy
import numpy as np

class TFIDF_Wrapper():
  """
  Compute TFIDF of training, testing and validation sets
  """
  def __init__(self, dataset,features=5000):
        self.dataset = dataset
        trainingDocuments = dataset.training_csv["Document"] 
        testingDocuments = dataset.testing_csv["Document"]

        
        self.vectorizer =CountVectorizer(token_pattern="[^\*]+", max_features=features)
        self.transformer =TfidfTransformer( smooth_idf=True) #idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1
        
        processor = ProcessDocuments()
        self.processedTrainingDocuments = trainingDocuments.apply(lambda x : 'nan***' + processor.preprocessDocument(x) )
        self.processedTestingDocuments = testingDocuments.apply(lambda x : processor.preprocessDocument(x) )
        

        self.training_tf = self.vectorizer.fit_transform(self.processedTrainingDocuments).tolil()
        self.training_tfidf = self.transformer.fit_transform(self.training_tf ).tolil()
        self.idf = self.transformer.idf_

          
        self.testing_tf = self.vectorizer.transform(self.processedTestingDocuments).tolil()
        self.testing_tfidf = self.transformer.transform(self.testing_tf ).tolil()
      
  
        self.features = self.vectorizer.get_feature_names()
        
        self.idf_raw = self.training_tf.copy()
        self.idf_raw [self.idf_raw >0] =1
        self.idf_raw  = self.idf_raw.sum(axis=0)

class TFICF_Wrapper():
  """
    Compute TFICF of training set

  """
  def __init__(self, tfidf_dataset):
    # Step 2 - Model 2: term-categories Model

      self.labels = tfidf_dataset.dataset.labels
      trainingLabels = tfidf_dataset.dataset.training_csv["Label"]
      LabelTerms = {label: [] for label in  self.labels}
      for i in range(len(trainingLabels)):
          LabelTerms[trainingLabels[i]].append(tfidf_dataset.processedTrainingDocuments[i])
      for k in LabelTerms.keys():
          LabelTerms[k]= "***".join(LabelTerms[k])
          
      self.labels = list(LabelTerms.keys())
      self.categoryDocuments = list(LabelTerms.values())
      self.tf = tfidf_dataset.vectorizer.transform(self.categoryDocuments).tolil()
      
      self.transformer =TfidfTransformer( smooth_idf=True) #idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1
      self.tficf = self.transformer.fit_transform(self.tf ).tolil()
      self.icf = self.transformer.idf_

      self.icf_raw = self.tf.copy()
      self.icf_raw [self.icf_raw >0] =1
      self.icf_raw  = self.icf_raw.sum(axis=0)



class LoadVariables():
  def __init__(self, dataset, features):

    self.dataset = Dataset(dataset) # Label, Document
    self.tfidf_data =TFIDF_Wrapper(self.dataset, features=features)
    self.tficf_data = TFICF_Wrapper(self.tfidf_data)

  def loadVariablesToTrain(self):

    self.TFIDF_CAT = self.tficf_data.tficf
    self.TF_CAT = self.tficf_data.tf
    self.ICF = self.tficf_data.icf
    self.ICF_RAW = self.tficf_data.icf_raw



    self.TF_TRAIN = self.tfidf_data.training_tf
    self.TF_TEST = self.tfidf_data.testing_tf

    
    self.TFIDF_TRAIN = self.tfidf_data.training_tfidf
    self.TFIDF_TEST = self.tfidf_data.testing_tfidf
    self.IDF = self.tfidf_data.idf

    self.LABELS_TRAIN = self.tfidf_data.dataset.training_csv["Label"]
    self.LABELS_TEST = self.tfidf_data.dataset.testing_csv["Label"]
    self.labels = self.tficf_data.labels
    self.mapping = {self.labels[i]: (self.TFIDF_CAT[i].toarray().flatten())*1.0 for i in range(len(self.labels))}
    self.features = self.tfidf_data.features

    #MAKES IT INCREDIBLY FASTER
    self.TFIDF_CAT_TRAIN = self.TFIDF_CAT.todense()
    self.TF_CAT_TRAIN = self.TF_CAT.todense()
    self.ICF_RAW_TRAIN = copy.deepcopy(self.ICF_RAW.copy())
    self.ICF_TRAIN = copy.deepcopy(self.ICF.copy())
    self.NbrOfColumns =self.TFIDF_CAT_TRAIN.shape[1]

    self.map = {self.labels[i]: i for i in range(len(self.labels)) }
    self.trainingLabelsIndexes = [self.map[label] for label in self.dataset.training_csv["Label"]]
    self.testingLabelsIndexes = [self.map[label] for label in self.dataset.testing_csv["Label"]]
    self.LABELS_TRAIN_MAPPED = np.array([ self.mapping[label] for label in self.LABELS_TRAIN])
    self.LABELS_TEST_MAPPED = np.array([ self.mapping[label] for label in self.LABELS_TEST])
    