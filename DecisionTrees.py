import numpy as np

### RIGHT = TRUE

class DecisionTree():
    def __init__(self, n_features=None, max_depth=10):
        self.n_features = n_features
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.n_features = y.shape # maybe X.shape[0]?
        self.root = self.build_tree(X, y, self.max_depth)


    def gini_impurity(self): # preferable to entropy as logarithms are computationally taxing
        pass

    def information_gain(self, uncertainty, left, right):
        pass
    
    def partition_rows(self):
        return left_X, right_X
    
    def find_best_split(self, X, y):
        pass

    def build_tree(self, X, y, depth): # recursive, returns root
        pass

    def classify(self, x, node): # recursive, returns bool
        # base case: leaf reached
        if node.is_leaf_node():
            return node.value
        
        if node.decisionQuestion.compare(x):
            self.classify(x, node.right)
        else:
            self.classify(x, node.left)

    def accuracy_score(self):
        pass
        


class Node():
    def __init__(self, decisionQuestion, left, right, value=None,):
        self.decisionQuestion = decisionQuestion
        self.value = value
        self.left = left
        self.right = right

    def is_leaf_node(self):
        if self.value == None:
            return False
        return True



class DecisionQuestion():
    def __init__(self, feature, decision_value):
        self.feature = feature
        self.decision_value = decision_value
    
    # return true if datapoint's feature has a value greater than or equal to a fixed decision value
    def compare(self, datapoint):
        current_value = datapoint[self.feature]
        return current_value >= self.decision_value


