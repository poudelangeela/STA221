from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
import imblearn

import random
random_state = np.random.RandomState(42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import statistics
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.metrics import accuracy_score as acc
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
random_state = np.random.RandomState(42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,  make_scorer, precision_score, recall_score, \
average_precision_score, accuracy_score, average_precision_score
from sklearn.metrics import roc_curve, auc  , precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import random
import seaborn as sns
import joblib
from sklearn import metrics
from scipy.stats import ks_2samp
import numpy as np
from datetime import datetime
import ast
from sklearn.feature_selection import f_regression, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn import feature_selection
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import time
from os import listdir
from os.path import isfile, join



import warnings
warnings.filterwarnings("ignore")



from dataselectutils import get_dataset,statistical_filter,mutual_info, RFE_features,permutation_importance_features
from arguments import data_files,test_folder,train_folder,project_folder,data_folder,label_col,pt_col


repeat_flag = 'Y'

data_files = data_files

project_folder = project_folder
data_folder = data_folder
train_folder = train_folder
test_folder = test_folder
label_col,pt_col = label_col,pt_col

hyperparameter_catalog = {
    
    'RF': {
        'bootstrap': [True],
        'max_depth': [2, 5, 10], # maximum depth of the tree
        'max_features': ['log2','sqrt'], # maximum number of features to use at each split
        'min_samples_leaf': [5,10], # minimum number of samples to split a node
        'min_samples_split': range(2,10,2),
        'n_estimators': [100,200, 500], # number of trees
        'criterion' : ['gini','entropy']  # criterion for evaluating a split
    },
    # 'XGB': {
    #     'learning_rate': [0.2, 0.4, 0.6, 0.8],
    #     'n_estimators': [100,200, 500],
    #     'max_depth': [2, 5, 10],
    #     'max_features': ['auto','sqrt'],
    #     'min_samples_leaf': [5,10],
    #     'min_samples_split': range(2,10,2)
        
    # },
    # 'SVM': {
    #     'C': [0.5, 1, 1.5],
    #     'kernel': ['poly', 'sigmoid'],
    #     'degree': [3,4],
    #     'gamma': ['scale', 'auto']
        
    # },
    # 'LR': {
    #     'penalty': ['l1', 'l2', 'elasticnet'],
    #     'solver': ['lbfgs', 'liblinear'] 
        
    # }
}

from scipy.stats import ks_2samp



rp_list = [['n','n'], ['y', 'n'], ['n', 'y']]


# data_folder = 'bootstraps_sv2/'

filtered_col_list = []
fold_perf = []
only_comm = 'n'
comm_feats_90 = ['median_dia_area',
 'median_pulse_pres',
 'median_sys_dec_area',
 'median_t_dia',
 'median_t_sys',
 'std_avg_dia',
 'std_dias_pres',
 'std_dic_pres',
 'std_pp_area']
comm_feats_50 = ['std_dias_pres',
 'std_sys_dec_area_nor',
 'std_sys_pres',
 'std_sys_area_nor',
 'std_avg_dia',
 'std_avg_sys',
 'std_sys_rise_area_nor',
 'std_avg_sys_rise',
 'median_sys_dec_area',
 'std_dic_pres',
 'std_pp_area',
 'std_pp_area_nor']





from configs import num_splits
num_files = num_splits
data_folder = ''
train_or_test = 'train/'



#### Select Algorithm Selection Method ####
# set selection_method to: 
# 1 for random forest, 2 for XGBoost,
# 3 Logistic Regression, 4 for Support Vector Machine

algorithm_catalog = {
    'RF': 1,
    'XGB': 2,
    'LR': 3,
    'SVM': 4
}
from configs import algorithm,use_features


import argparse



algorithm = algorithm
algorithm_no = algorithm_catalog[algorithm]
hyperparameter_grid = hyperparameter_catalog[algorithm]



file_list = [f for f in listdir(data_folder+train_or_test) if isfile(join(data_folder+train_or_test, f))]

filtered_col_list = []
fold_perf = []
import argparse
parser = argparse.ArgumentParser()

# parser.add_argument("--use_features",default='N')
# parser.add_argument("--featselection",default='SFS')
# args = parser.parse_args()

from configs import feature_selection_method,feature_import_path,use_prefered_cols,prefered_columns

import_feature_list = use_features # 'Y' to use saved features from feature selection code
if import_feature_list == 'Y':
    #both should be user input
    feature_selection_method = feature_selection_method
    feature_import_path = feature_import_path #'pickled_features/{}/{}_top{}_features.pkl'.format(feature_selection_method,feature_selection_method,suffix_str)

resultdfcols = ["params","mean_fit_time","rank_test_roc_auc","mean_test_roc_auc","split0_test_roc_auc","split1_test_roc_auc","split2_test_roc_auc","split3_test_roc_auc","split4_test_roc_auc","rank_test_accuracy","mean_test_accuracy","rank_test_prc_auc","mean_test_prc_auc","rank_test_precision","mean_test_precision","rank_test_recall","mean_test_recall","rank_test_specificity","mean_test_specificity"]
classification_metrics = ["mean_test_roc_auc","mean_test_accuracy","mean_test_prc_auc","mean_test_precision","mean_test_recall","mean_test_specificity"]
# trainsplit_metrics = [m[:5] + 'train'+m[10:] for m in classification_metrics]
# "mean_train_roc_auc","split1_train_roc_auc","split2_train_roc_auc","split3_train_roc_auc","split4_train_roc_auc",
classification_metrics_ci = [m[5:] + ' 95% CI' for m in classification_metrics]
forcols = pd.read_excel(project_folder+"/algorithm_selection/RF/statistical_feature_selection/fullbestmodels_cv.xlsx")
fullmodels_cv = pd.DataFrame(columns = forcols.columns)
bestmodels_cv = pd.DataFrame(columns =resultdfcols+classification_metrics_ci)

# if args.featselection!='SFS':
#     feature_selection_method = 'RFE'
# else:
#     feature_selection_method = "statistical_feature_selection"
print(feature_selection_method)
for file_num in range(num_files):
    print('')
    # print(file_num)
    # if file_num==4:
    #     break

    start_time = time.time()
    
    if import_feature_list == 'N':
        All_file_pickle_folder = data_folder + 'algorithm_selection/' + \
                                 algorithm + '/all_features' + '/model_'+file_list[file_num][:-4] + '/'
    else:
        All_file_pickle_folder = data_folder + 'algorithm_selection/' + \
                                 algorithm + '/' + feature_selection_method + \
                                 '/model_' + file_list[file_num][:-4] + '/'
    data_file = file_list[file_num]
    print('Processing file ' + data_file)
    if not os.path.isdir(All_file_pickle_folder):
        os.makedirs(All_file_pickle_folder)
    

    
    param_grid = hyperparameter_grid
    hyperparameters = {'classification_model__' + key: param_grid[key] for key in param_grid}
    
    
    scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
               'specificity': make_scorer(recall_score,pos_label=0),\
               'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True)}
    
    
    X,y,df_dataset, cv = get_dataset(data_folder+train_or_test+data_file,file_num,label_col,pt_col)


    
    if import_feature_list == 'Y':
        feature_dict = joblib.load(feature_import_path)
        try:
            selected_features = feature_dict[data_file[:-4]]
        except:
            print("probably key error passing all features instead")
            selected_features = X.columns
        
        if selected_features==[]:
            print("no feature was selected, passing whole data instead")
            selected_features = X.columns
        if use_prefered_cols:
            selected_features = prefered_columns
        print("selected features are ",len(selected_features),selected_features)
        X = X[selected_features]#[list(X.columns[:51]) + list(selected_features)]
    column_list =[]
    column_list.append(X.columns.tolist())
    print(column_list)
    filtered_col_list.append(X.columns.tolist())

        
    inner_cv = cv
    
    
    if algorithm_no == 1:
        classification_model=RandomForestClassifier(random_state=1)
    elif algorithm_no == 2:
        classification_model=GradientBoostingClassifier(random_state=1)
    elif algorithm_no == 3:
        classification_model=LogisticRegression()
    elif algorithm_no == 4:
        classification_model=SVC(probability=True, random_state=1)
    
    
    noimb_pipeline = Pipeline([('classification_model', classification_model)])
    
    
    clf = GridSearchCV(noimb_pipeline, param_grid= hyperparameters, verbose =0,cv=inner_cv, scoring= scoring, refit = 'roc_auc', n_jobs=-1,error_score="raise",return_train_score=True)
    import time
    # startfit = time.time()
    clf.fit(X, y)
    
    # endfit = time.time()
    # print(endfit-startfit,"fitting took this much time")
    # print(clf.best_estimator_)
    # print("BEst results",clf.best_score_,"index",clf.best_index_,"best params",clf.best_params_)
    fold_perf.append(clf.cv_results_)
    
     
    results_df = pd.DataFrame(clf.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_roc_auc"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("kernel")

    # print(inner_cv)
    # clf.best_index_ = results_df["rank_test_roc_auc"].argmin()
    # print(results_df["params"][clf.best_index_])
    # from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,average_precision_score

    
    # noimb_pipeline.fit(X.loc[X.index[inner_cv[0][0]],:],y.loc[y.index[inner_cv[0][0]],:])
    # noimb_pipeline.set_params(**results_df["params"][clf.best_index_])
    # y_pred1,y_pred1_proba = noimb_pipeline.predict(X.loc[X.index[inner_cv[0][0]],:]),noimb_pipeline.predict_proba(X.loc[X.index[inner_cv[0][0]],:])
    
    # print("1",accuracy_score(y1,y_pred1),roc_auc_score(y1,y_pred1_proba))
    # exit()
    # y_pred2,y_pred2_proba = noimb_pipeline.predict(X.loc[X.index[inner_cv[1][0]],:]),noimb_pipeline.predict_proba(X.loc[X.index[inner_cv[1][0]],:])
    # y_pred3,y_pred3_proba = noimb_pipeline.predict(X.loc[X.index[inner_cv[2][0]],:]),noimb_pipeline.predict_proba(X.loc[X.index[inner_cv[2][0]],:])
    # y_pred4,y_pred4_proba = noimb_pipeline.predict(X.loc[X.index[inner_cv[3][0]],:]),noimb_pipeline.predict_proba(X.loc[X.index[inner_cv[3][0]],:])
    # y_pred5,y_pred5_proba = noimb_pipeline.predict(X.loc[X.index[inner_cv[4][0]],:]),noimb_pipeline.predict_proba(X.loc[X.index[inner_cv[4][0]],:])

    # y1,y2,y3,y4,y5 = y.loc[y.index[inner_cv[0][0]],:],y.loc[y.index[inner_cv[1][0]],:],y.loc[y.index[inner_cv[2][0]],:],y.loc[y.index[inner_cv[3][0]],:],y.loc[y.index[inner_cv[4][0]],:]
    # print("1",accuracy_score(y1,y_pred1),roc_auc_score(y1,y_pred1_proba))
    # print("2",accuracy_score(y2,y_pred2),roc_auc_score(y2,y_pred2_proba))
    # print("3",accuracy_score(y3,y_pred3),roc_auc_score(y3,y_pred3_proba))
    # print("4",accuracy_score(y4,y_pred4),roc_auc_score(y4,y_pred4_proba))
    # print("5",accuracy_score(y5,y_pred5),roc_auc_score(y5,y_pred5_proba))
    # print(results_df)
    # print(results_df[results_df["params"]==clf.best_params_])
    # results_df[results_df["params"]==clf.best_params_].to_excel("temp.xlsx")
    # print(results_df.loc[clf.best_index_,:])
    # exit()
    
    temp = results_df[results_df["params"]==clf.best_params_]

    tmp1 = results_df[results_df["params"]==clf.best_params_][resultdfcols].values.tolist()
    
    
    # print(tmp1)
    splitlevel = {j:["split" + str(i) + "_"+j[5:] for i in range(5) ] for j in classification_metrics}
    # print(splitlevel) 
    # calc1 = time.time()

    for metric in classification_metrics:
        print(metric)
        metric_ci = round(1.96*np.std(temp[splitlevel[metric]].values.tolist())/np.sqrt(len(temp[splitlevel[metric]].values.tolist())),3)
        # print(temp[metric].values.tolist(),temp[splitlevel[metric]].values.tolist(),np.std(temp[splitlevel[metric]].values.tolist()),np.sqrt(len(temp[splitlevel[metric]].values.tolist()[0])),metric_ci)    

        tmp1[0].append(metric_ci)
    # endcal =time.time()
    # print("\n\n\n\n",endcal-calc1,"calculation time ")
    # print(tmp1)
    # exit() 
    tmp_bootstrap_summary = pd.DataFrame([tmp1[0]],columns=resultdfcols+classification_metrics_ci)
    tmp_bootstrap_summary.to_excel("temp.xlsx")
    # exit()
    bestmodels_cv=bestmodels_cv.append(tmp_bootstrap_summary,ignore_index= True)
    
    # print("best model",temp[temp["params"]==clf.best_params_])
    # print(results_df.columns)
    # results_df.to_excel("prelimresults.xlsx")
    # exit()
    model_to_choose =clf.best_estimator_ 
    
    model_file = "classification_model_"+file_list[file_num][:-4]+".pkl"
    
    
    temppath = data_folder + 'algorithm_selection/' + algorithm + '/' + feature_selection_method 
    bestmodels_cv.to_excel(temppath+"/bestmodels_cv.xlsx")
    
    fullmodels_cv=fullmodels_cv.append(temp,ignore_index=True)
    fullmodels_cv.to_excel(temppath+"/fullbestmodels_cv.xlsx")
    # exit()
    joblib.dump(model_to_choose, All_file_pickle_folder+model_file)
    print('')
    processing_time = (round(time.time() - start_time, 2))
    if processing_time > 60:
        print('Processing time for file ' + data_file + ': ' + str(round(processing_time/60, 2)) + ' minutes')
    else:
        print('Processing time for file ' + data_file + ': ' + str(processing_time/60) + ' seconds')
    
    print('')
    print('Processed file number ' + str(file_num + 1))
    print(str(round(100*(file_num+1)/num_files, 2)) + '% complete')
    print('')
    print('-------------------------------------------------------')
performance_file = "gridsearch_classification_performance_"+file_list[file_num][:-4]+".pkl"
joblib.dump(fold_perf, All_file_pickle_folder+performance_file)




