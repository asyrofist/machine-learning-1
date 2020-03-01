import numpy as np
from typing import List
from hw1_knn import KNN

import hw1_dt as decision_tree

from collections import defaultdict
import copy 

def entropy(s, return_counts=False):
    # s is the list containing labels of feature columns (ie array of labels)
    # s should contain values of final classification such as [1 0 0 1] and not individual attribute values

    x, counts = np.unique(s, return_counts=True)
    px=counts / np.sum(counts)
    if return_counts:
        return -np.sum(px * np.log2(px)), np.sum(counts)
    return -np.sum(px * np.log2(px))


# TODO: Information Gain function
def Information_Gain(S, branches):
#dim branches: num_attribute_values x num_cls
    Sc=[]
    pSc=[]
    for attribute in branches:
        cls_total=np.sum([attribute])
        if cls_total!=0:
            cls=np.array([a for a in attribute if a !=0])
            Sc.append(np.sum([-cls/cls_total*np.log2(cls/cls_total)]))
            pSc.append(cls_total)
    pSc=np.array(pSc)
    #print(pSc)
    #print(Sc)
    Sc=np.sum(-pSc/np.sum(pSc)*np.array(Sc))
    #print(Sc)
    return S+Sc


#----------------------------------



def training_validation_split(X_test, y_test, k=.7):
    # split data into training and validations sets with k=proportion of data to use for training
    train_count = round(k * len(y_test))
    ytest = np.reshape(y_test, (len(y_test), 1))
    arr = np.array([np.concatenate([X_test[i], ytest[i]]) for i in range(len(y_test))])
    arr=np.random.permutation(arr)
    train=arr[:train_count]
    validation=arr[train_count:]
    X_train=[a[:-1].tolist() for a in train]
    y_train=[a[-1].tolist() for a in train]
    X_val = [a[:-1].tolist() for a in validation]
    y_val = [a[-1].tolist() for a in validation]
    return X_train, y_train, X_val, y_val

def tree(): return defaultdict(tree)

def REP_preprocessing(decisionTree, nodes=None, curr_node=None,depth=None, idx=None):

    if curr_node is None:
        curr_node = decisionTree.root_node
        nodes = tree()
        depth = 0
        idx=0
        curr_node.depth = 0
        curr_node.idx = 0

    curr_node.depth = depth
    curr_node.idx = idx

    if curr_node.splittable:
        nodes[depth][idx] = curr_node

        for idx_child, child in enumerate(curr_node.children):
            REP_preprocessing(decisionTree, nodes=nodes, curr_node=child, depth=depth + 1, idx=idx_child)
    else:
        nodes[depth][idx]= curr_node.cls_max
        #nodes[depth][idx]=len(curr_node.labels)

    return nodes

def calc_accuracy(predictions, labels):
    l=list(zip(predictions,labels))
    num_predictions=len(predictions)
    num_correct=sum([1 for x,y in l if x==y])
    return float(num_correct)/num_predictions

def update_tree(decision_tree, pruneList):
    nodes=REP_preprocessing(decision_tree)
    for depth,idx in pruneList:
        nodes[depth][idx].children=[]
        nodes[depth][idx].splittable=False
    return

def update_tree(decisionTree, pruned_node): 
    nodes=REP_preprocessing(decisionTree)
    depth, idx=pruned_node
    nodes[depth][idx].children = []
    nodes[depth][idx].splittable = False
    #for depth,idx in pruneList:
    #    nodes[depth][idx].children=[]
    #    nodes[depth][idx].splittable=False

    return

def find_prunable(decisionTree, X_test, y_test, max_depth=None, start_idx=None):
    xtrain, ytrain, xval, yval = training_validation_split(X_test, y_test)
    pruningTree = copy.deepcopy(decisionTree)
    nodes = REP_preprocessing(pruningTree)
    if max_depth is None:
        max_depth = len(nodes.keys()) - 1
    if start_idx is None:
        start_idx=0
    most_prunable = None

    max_accuracy=calc_accuracy(decisionTree.predict(X_test), y_test) #decisionTree.root_node.labels)
    print('=========================')
    #print('Pre-pruning accuracy: ', max_accuracy)
    #orig_nodes=REP_preprocessing(decisionTree)


    for depth in np.arange(max_depth, 0, -1): #bottoms up
    #for depth in np.arange(1,max_depth):  # top down
        for idx_child, child in enumerate(nodes[depth]):
            if (depth==max_depth) and (idx_child<=start_idx):
                continue
            if (type(nodes[depth][idx_child])==decision_tree.TreeNode) and nodes[depth][idx_child].splittable:
                #subtree=copy.deepcopy(nodes[depth][idx_child])
                #temp=copy.deepcopy(pruningTree)
                xtrain, ytrain, xval, yval = training_validation_split(X_test, y_test)
                nodes[depth][idx_child].children=[]
                nodes[depth][idx_child].splittable=False
                pruned_accuracy=calc_accuracy(pruningTree.predict(X_test), y_test)
                #print('pruning_accuracy: [{}][{}] = {}'.format(depth, idx_child, pruned_accuracy))

                if pruned_accuracy>max_accuracy:
                    print('prunable: [{}][{}] = {}'.format(depth,idx_child,pruned_accuracy))
                    #pruneList.append([pruned_accuracy,(depth, idx_child)])
                    most_prunable=[depth, idx_child]
                    max_accuracy=pruned_accuracy
                    #nodes = REP_preprocessing(pruningTree)
                    break

    return max_accuracy,most_prunable #repeat until most_prunable is None

def reduced_error_prunning(decisionTree, X_test, y_test):
    orig_accuracy = calc_accuracy(decisionTree.predict(X_test), y_test)  # decisionTree.root_node.labels)
    # xtrain, ytrain, xval, yval = training_validation_split(X_test, y_test)
    print('=========================')
    print('Pre-pruning accuracy: ', orig_accuracy)
    pruneList = []
    updateList = []
    num_update = 0
    accuracy, node = find_prunable(decisionTree, X_test, y_test)

    pruneList.append([accuracy, node])
    while node is not None:
        accuracy, node = find_prunable(decisionTree, X_test, y_test, node[0], node[1])
        if node is not None:
            pruneList.append([accuracy, node])
    if len(pruneList) <= 1 and pruneList[0][1] is None:
        return
    pruneList = sorted(pruneList, reverse=True)

    best_node = pruneList[0][1]
    update_tree(decisionTree, best_node)

    print('new tree accuracy: ', calc_accuracy(decisionTree.predict(X_test), y_test))
    # +++++++++++++++++++++++++++++++++++++
    return reduced_error_prunning(decisionTree, X_test, y_test)



# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    labels = list(zip(real_labels, predicted_labels))
    tp = sum([1 for x, y in labels if x == 1. and y == 1.])
    fp = sum([1 for x, y in labels if x == 0. and y == 1.])
    fn = sum([1 for x, y in labels if x == 1. and y == 0.])
    if 2 * tp + fp + fn == 0:
        return 0
    f1 = 2 * tp / float(2 * tp + fp + fn)
    return f1


#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    assert len(point1) == len(point2)
    return sum([(p1 - p2) ** 2 for p1, p2 in zip(point1, point2)]) ** .5



#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    assert len(point1) == len(point2)
    return sum([float(p1 * p2) for p1, p2 in zip(point1, point2)])

#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    return -np.exp(-.5 * sum([(p1 - p2) ** 2 for p1, p2 in zip(point1, point2)]))


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    a = sum([p1 ** 2 for p1 in point1]) ** .5
    b = sum([p2 ** 2 for p2 in point2]) ** .5
    if (a * b) == 0:
        return 0
    return 1-inner_product_distance(point1, point2) / (a * b)



# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    
    best_distance = None
    best_function = ""
    best_k = 0
    max_score = 0.0
    kvals = np.arange(1, len(Xtrain), 2)
    train_f1_score = 1.0
    valid_f1_score = 0.0
    model = None

    for name, f in distance_funcs.items():
        for k in kvals:
            if k == 1:
                model = KNN(k, f)
                model.train(Xtrain, ytrain)
            else:
                model.k = k
            # train_f1_score =f1_score(ytrain,model.predict(Xtrain))
            # predicted=model.predict(Xval)
            valid_f1_score = f1_score(yval, model.predict(Xval))
            if valid_f1_score > max_score:
                max_score = valid_f1_score
                print("new valid score: ", valid_f1_score)
                best_distance = f
                best_function = name
                best_k = k
                print('**NEW BEST MODEL**')
            if valid_f1_score == max_score:
                if k<best_k:
                    max_score = valid_f1_score
                    print("new valid score: ", valid_f1_score)
                    best_distance = f
                    best_function = name
                    best_k = k
            

    # best_model=KNN(best_k,best_distance)
    # best_model.train(Xtrain,ytrain)
    model.k = best_k
    model.distance_function = best_distance
    model.f1 = max_score
    
    
    best_model=KNN(best_k,best_distance)
    best_model.train(Xtrain, ytrain)

    # Dont change any print statement
    print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) +
                  'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                  'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
    print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) +
          'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
          'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

    print()
    print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=name, best_k=best_k))
    return best_model, best_k, best_function


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    # ifthere are less than 30 points in dataset, choose n-1 as the upper bound of K. 
    #'n' is the number of points in dataset. 
    # You can choose N-1 as best k if N-1 is an odd number.
    best_scale = None
    best_score = 0.0
    best_distance = None
    best_func = ""
    best_k = 0
    max_score = 0.0
    #modified due to grading instructions
    if len(Xtrain)<30:
        kvals = np.arange(1, len(Xtrain), 2)
    kvals = np.arange(1, len(Xtrain), 2)
    train_f1_score = 1.0
    valid_f1_score = 0.0
    model = None
    for scaling_name, new_scaler in scaling_classes.items():
        print(scaling_name, new_scaler)
        scaler = new_scaler()
        scaled_Xtrain = scaler(Xtrain)
        scaled_Xval = scaler(Xval)
        for name, f in distance_funcs.items():
            for k in kvals:
                if k == 1:
                    model = KNN(k, f)
                    model.train(scaled_Xtrain, ytrain)
                else:
                    model.k = k
                valid_f1_score = f1_score(yval, model.predict(scaled_Xval))
                if valid_f1_score > max_score:
                    max_score = valid_f1_score
                    best_distance = f
                    best_k = k
                    best_func = name
                    best_scale = scaling_name
                if valid_f1_score == max_score:
                    if k<best_k:
                        max_score = valid_f1_score
                        best_distance = f
                        best_k = k
                        best_func = name
                        best_scale = scaling_name
                
    model.k = best_k
    model.distance_function = best_distance
    model.f1 = max_score
    model.scaler = scaling_classes[best_scale]

    
    best_model=KNN(best_k,best_distance)
    best_model.scaler= scaling_classes[best_scale]
    best_model.train(Xtrain, ytrain)
    #print('best score: ', best_score)
    #print('best k: ', best_k)
    #print('best scaler: ', scaling_name)

    # Dont change any print statement

    print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) +
                      'train:{train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                      'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))


    print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
          'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
          'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

    print()
    print('[part 1.2] {name}\t{scaling_name}\t'.format(name=name, scaling_name=scaling_name) +
          'best_k: {best_k:d}\t'.format(best_k=best_k))
    print()
    return best_model, best_k, best_func, best_scale
       

class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalized_vector = []
        for point in features:
            if all(p == 0 for p in point):
                normalized_vector.append(point)
            else:
                denom = float(np.sqrt(inner_product_distance(point, point)))
                normalized=[p / denom for p in point]
                normalized_vector.append(normalized)

        return normalized_vector

class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        
        self.feat_min = None
        self.feat_max = None
        self.max_min_diff = None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        
        feat_array = np.array(features)
        
        # set scaling factors using training data (assume all valid)
        # if self.feat_min is None:
        if (hasattr(self, 'feat_min') == False) or (self.feat_min is None):
            self.feat_min = np.nanmin(feat_array, axis=0)
            self.feat_max = np.nanmax(feat_array, axis=0)
          

            self.max_min_diff = self.feat_max - self.feat_min
            # self.constants=np.where(self.max_min_diff==0)
            # self.scaling_features=np.where(self.max_min_diff!=0)

        # norm_feat_numerator = (feat_array - self.feat_min)
        # norm_feat = norm_feat_numerator / self.max_min_diff
        norm_feat = (feat_array - self.feat_min) / self.max_min_diff

        # if all values for a feature are the same (feat_max==feat_min)
        if (self.max_min_diff == 0).any():
            constants = np.where(self.max_min_diff == 0)
            norm_feat[constants]=0
            #changed due to instructions to set 0
            #norm_feat[np.where(norm_feat == -np.inf)] = 0
            #norm_feat[np.where(norm_feat == np.inf)] = 0
            #norm_feat[np.where(np.isnan(norm_feat))] = 0
            #old
            # norm_feat[np.where(norm_feat_numerator[self.constants] > 0)] = 1
            # norm_feat[np.where(norm_feat_numerator[self.constants] < 0)] = 0
            # norm_feat[np.where(norm_feat_numerator[self.constants] == 0)] = .5
        return norm_feat.tolist()




