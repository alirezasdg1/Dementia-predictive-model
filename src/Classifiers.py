import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, multilabel_confusion_matrix, confusion_matrix, log_loss, make_scorer, roc_auc_score, roc_curve, auc
from sklearn.inspection import permutation_importance

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle


def ROC(y_test_b,y_score_model,model_name):
    path_save = f'Outcomes/Figs/ROC_{model_name}_1.png'
    n_classes = y_test_b.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score_model[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_b.ravel(), y_score_model.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    class_names = ['Healthy',"Alzheimer's disease",'Vascular dementia',"Parkinson's disease"]
    # interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize = (10,10))
    # plt.plot(fpr["micro"], tpr["micro"],
    #         label='micro-average ROC curve (area = {0:0.2f})'
    #         ''.format(roc_auc["micro"]),
    #         color='deeppink', linestyle=':', linewidth=4)

    # plt.plot(fpr["macro"], tpr["macro"],
    #         label='macro-average ROC curve (area = {0:0.2f})'
    #         ''.format(roc_auc["macro"]),
    #         color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','slategrey'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                label='ROC of {0} (area = {1:0.2f})'
                ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.title(f'Extension of ROC for {model_name} model',fontsize=20)
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig(path_save)
    plt.show()

    return fpr, tpr, roc_auc


class DementiaClassifier():
    def __init__(self,X,y,model):
        self.model = model
        self.X = X
        self.y = y
        
    def score_cv(self, cvn=5):

        accuracy_val =  np.mean(cross_val_score(self.model, self.X, self.y, cv = cvn, scoring=make_scorer(accuracy_score)))
       
        return f"accuracy_val = {accuracy_val}"

    def score_model(self,X_test,y_test):
        
        self.model.fit(self.X,self.y)
        
        y_preb = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)

        accuracy = self.model.score(X_test,y_test)
        macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
        weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="weighted")
        macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

        confusion = multilabel_confusion_matrix(y_test, y_preb)
        print(f'Confusion_matrix: \n {confusion}')
        print(f'Confusion_matrix: \n {confusion.reshape(4,4)}')

        confusion2 = confusion_matrix(y_test, y_preb)

        return confusion, confusion2, pd.DataFrame({'accuracy':accuracy,
                              'One-vs-One ROC AUC scores':macro_roc_auc_ovo,
                              'weighted by prevalence_OVO': weighted_roc_auc_ovo,
                              'macro_roc_auc_ovr':macro_roc_auc_ovr,
                              'weighted by prevalence_ovr': weighted_roc_auc_ovr},index=[1])




    def Gsearch(self, parameters,searchtype):

        if searchtype == "G":
            grid = GridSearchCV(self.model,  parameters, n_jobs=-1, verbose=1, scoring=make_scorer(accuracy_score))
            grid.fit(self.X, self.y)
            out = grid

        else:
            grid_rand =  RandomizedSearchCV(self.model,  param_distributions = parameters,cv=3, n_iter=10, n_jobs=-1, verbose=1, scoring=make_scorer(accuracy_score))   
            grid_rand.fit(self.X, self.y)
            out = grid_rand

        return out.best_params_, out.best_estimator_


   




if __name__ == '__main__':
    from sklearn import datasets
    irs = datasets.load_iris()
    X = irs.data
    y = irs.target
    depth_parm = np.arange(1, 2, 1)
    num_samples_parm = np.arange(5,15,10)
    parameters = {'max_depth' : depth_parm,
              'min_samples_leaf' : num_samples_parm}
    output = DementiaClassifier(X,y,RandomForestClassifier()).score(5)
    print(DementiaClassifier(X,y,RandomForestClassifier()).Gsearch(parameters,"R"))