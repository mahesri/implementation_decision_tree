import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from sklearn import tree

# membuat dataset sederhana
X = np.array([[1.4, 0.2], [1.5, 0.3], [4.0, 1.2], [4.5, 1.5], [5.0, 1.7], [5.5, 2.0]])
y = np.array(['setosa', 'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica'])

# membuat dan train decision tree
clf = DecisionTreeClassifier()
clf.fit(X, y)

# memvisualisasikan decision tree
dot_data = export_graphviz(clf, out_file=None, 
                          feature_names=['Panjang kelopak (cm)', 'Lebar kelopak (cm)'],
                          class_names=['setosa', 'versicolor', 'virginica'],
                          filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('decision_tree.png')
