feature_selection_method = 'statistical_feature_selection'
use_features='Y'
if feature_selection_method=='statistical_feature_selection':
    suffix_str=''
else:
    suffix_str='_10'
    
feature_import_path = 'pickled_features/{}/{}_top{}_features.pkl'.format(feature_selection_method,feature_selection_method,suffix_str)
algorithm = 'RF'
num_splits = 2
prefered_columns =[]#['Delta BMI', 'ACS_PCT_NO_WORK_NO_SCHL_16_19_ZC', 'Yes Induction', 'POS_DIST_TRAUMA_ZP', 'Y_ECG', 'ACS_PCT_OTH_LANG_ZC']

use_prefered_cols = False


