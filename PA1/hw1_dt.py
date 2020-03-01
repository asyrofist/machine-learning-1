import numpy as np
import utils as Util
from collections import defaultdict
import pandas as pd

# =================================================================================
class DecisionTree():

    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    # input: 
    #   - features: List[List[any]] traning data, num_cases*num_attributes
    #   - labels: List[any] traning labels, num_cases*1
    def train(self, features, labels):
        self.features=features
        self.labels=labels
        num_cls=len(np.unique(labels))

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        y_pred = []
        for feature in features:
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            #print ("feature: ", feature)
            #print ("pred: ", pred)
        return y_pred


# =================================================================================

class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]] --> all the points in current TreeNode
        # labels: List[int] -->correstponding labels for all data
        # num_cls: int
        # self.children = list of TreeNode after split current node based on best attributes
        self.features = features 
        self.labels = list(labels)

        self.children = [] 
        self.num_cls = num_cls
        
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label  # majority of current node


        # splittable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split
        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    # TODO: implement split function
    # Get list of possible splits
    # Sorted by decreasing gain and number of attributes (possibleSplits[0] is always best)
    def split(self):
        Sn=Util.entropy(self.labels)
        max_gain=-np.inf
        branch_vals=[]
        min_entropy=np.inf
        df=pd.DataFrame(self.features)
        if df.empty:
            self.splittable=False
        else:
            df['labels']=self.labels
            for col in df.drop(columns='labels'):
                branches = np.nan_to_num(df[[col, 'labels']].groupby(by=[col, 'labels']).size().unstack().fillna(value=0).values)
                # branches=df[col].value_counts().to_frame().reset_index().values
                gain = Util.Information_Gain(Sn, branches.tolist())

                branch_vals = sorted(df[col].unique().tolist())
                if gain > max_gain:
                    max_gain=gain
                    self.dim_split = col
                    self.feature_uniq_split = branch_vals
                    #branches[:,0] #list(d.keys())

            split_df=df.groupby(by=[self.dim_split])

            for feature_val in self.feature_uniq_split:
                if feature_val not in split_df.groups.keys():
                    continue

                child=split_df.get_group(feature_val).drop(columns=self.dim_split)

                new_node = TreeNode(child.drop(columns=['labels']).values.tolist(), child['labels'].values.tolist(),self.num_cls)
                if child.drop(columns=['labels']).empty:
                    new_node.splittable = False
                    #new_node.cls_max = self.cls_max
                if (len(new_node.features) <= 1) or (child.drop(columns=['labels']).values == []):
                    #new_node.cls_max = self.cls_max
                    new_node.splittable = False
                self.children.append(new_node)

            for child in self.children:
                if child.splittable:
                    child.split()

        return

    # TODO:treeNode predict function
    def predict(self, feature): # called once we create the tree structure by the split function.
        # input: datapoint
        # return: predicted label according to current node:
        #   - if leaf node: return current leaf node label
        #   - if non-leaf node: split it to child node

        if self.splittable:
            if type(feature)!=list:
                feature=feature.tolist()
            #if feature[self.dim_split] not in self.feature_uniq_split:
            if (len(feature)<self.dim_split) or (feature[self.dim_split] not in self.feature_uniq_split):

                return self.cls_max
           
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            

            return self.children[idx_child].predict(feature[:self.dim_split]+feature[self.dim_split+1:])
            #feature.pop(self.dim_split)
            #return self.children[idx_child].predict(feature)
        else:

            return self.cls_max

