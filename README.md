
Keep the zipcode folder in ur project directory. Make new folders - algorithm_selection, train, test and datafile. Keep your data inside the folder datafile.  

In arguments.py change the path to your project directory
In configs.py change variables to reflect your choice of ML algorithm and feature selection methods.

Then, to train the model and get results on test set for the chosen configuration of experiment run 


bash scripts.sh 

Or 

The following files sequentially 

To generate train and test split and allocate k folds. This will create files inside train and test folders and generate a summary of the k folds allocation 

python split_fold_form.py


To perform feature selection. This will create a new folder called pickled_features and store selected features inside them 

python feature_engg.py


To train the model and perform GridSearchCV and hyper parameter fine tuning. This will save models and their training result summaries 

python featselect.py


To run the trained model on holdout test set. This will save results inside results folder

python holdout.py 
