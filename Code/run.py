from CFE_IterativeAdditive import CFE_IterativeAdditive
from CFE_GradientDescent import CFE_GradientDescent
from CFE_FeedForwardANN import CFE_FeedForwardANN
from Datasets import Dataset
from sklearn.svm import  LinearSVC

# d = 'R52'
# dataset = Dataset(d)
# labels = dataset.labels
# approach1= CFE_IterativeAdditive(d, features=1500)
# approach1.trainMatrix(validation= 0.9, splitP = 0.75, n_iterations_local= 100, n_epochs=100, resetValue=False)
# approach1.generateFeatures('l1', 1)
# model = LinearSVC()
# approach1.train(model, True, True)


d = 'R8'
dataset = Dataset(d)
labels = dataset.labels
approach1= CFE_GradientDescent(d, features=500)
approach1.trainMatrix( n_epochs=10, resetValue=False)
approach1.generateFeatures('l1', 1)
model = LinearSVC()
approach1.train(model, True, True)


# d='R52'
# approach2 = CFE_FeedForwardANN(d, features=500 )
# approach2.buildNetwork('cosine')
# approach2.train(n_epochs= 1)
# transformed = approach2.predict('test')
# acc , labels = (approach2.score())
# print(acc)