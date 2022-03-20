import pandas as pd
import pickle, re
from nltk.stem import   WordNetLemmatizer


class Dataset:
    """
    This class represents our datasets:
      - name
      - training_csv
      - testing_csv
      - labels

    """
    def __init__(self,name):
        if name in ["R52" , "R8", "20NG",  "AGNews",]:
          
            self.name= name
            self.training_csv = pd.read_csv("./Datasets/{}/Train.csv".format(name))           
            self.testing_csv = pd.read_csv("./Datasets/{}/Test.csv".format(name))
            self.labels = self.training_csv["Label"].unique()
          

        else:
            raise Exception("{} does not correspond to a dataset supported".format(name))

class ProcessDocuments():
    """
    This class performs pre-processing of the documents.

    """
    def __init__(self):
        self.i=0
        with open('./Preprocessing/dictionary.pkl', 'rb') as f:
                self.dic = pickle.load(f)
        with open('./Preprocessing/stopwords.pkl', 'rb') as f:
                self.stopwords = pickle.load(f)
        self.special="***"
        self.stemmer = WordNetLemmatizer()
        self.map ={}
    def Lemmatize(self,word):
        if word in self.dic.keys():
            word = self.dic[word]
            stemmed = self.stemmer.lemmatize(word)
        else:
            stemmed = self.stemmer.lemmatize(word)

        if stemmed is None: return word
        return stemmed
    
    def preprocessDocument(self,text):
        #print(self.i)
        self.i+=1
        try:
          words = re.sub('\s+', ' ', re.sub('[^a-zA-Z0-9\s]+', ' ', text)).split(" ")
        except: 
          return "nan"
        tR =[]
        for word in words:
            if word not in self.map.keys() :
              if word in self.stopwords or word.strip()=="": 
                self.map[word]=""
              else: self.map[word]= self.Lemmatize(word).lower()
            tR.append(self.map[word])
        return self.special.join([a for a in tR if a!=""])
        #return [a for a in tR if a!=""]