"""
  @Content: demo for MESA
  @Author: Chaorong Chen
  @Date: 2022-07-06 17:20:08
  @Last Modified by: Chaorong Chen
  @Last Modified time: 2022-07-06 17:20:08
"""

from MESA import *
# Load data sets (cohort 1 as example)
X_c1_t1 = pd.read_csv("dir_to_data/cohort1_type1.csv",index_col=0)
X_c1_t2 = pd.read_csv("dir_to_data/cohort1_type2.csv",index_col=0)
X_c1_t3 = pd.read_csv("dir_to_data/cohort1_type3.csv",index_col=0)
X_c1_t4 = pd.read_csv("dir_to_data/cohort1_type4.csv",index_col=0)
X_c1_t5 = pd.read_csv("dir_to_data/cohort1_type5.csv",index_col=0)
X_c1_t6 = pd.read_csv("dir_to_data/cohort1_type6.csv",index_col=0)
y_cohort1 = pd.read_csv("dir_to_data/cohort1_label.csv",index_col=0).T.values[0]

# Example
random_state = 0

# Train-test split inside SBS
cv_sbs = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)

# Classifiers used for final evaluation on test sets
svm = SVC(kernel="linear",random_state=random_state,probability=True)
rf = RandomForestClassifier(random_state=random_state,n_jobs=-1)

# Run pipeline for cohort 1, type 1 feature
sbs_c1_t1 = SBS_LOO(X=X_c1_t1,
             y=y_cohort1,
             estimator=svm,
             cv=cv_sbs,
             classifiers=[svm, rf],
             min_feature=10,
             boruta_top_n_feature=1000)

# Run pipeline for cohort 1, type 2 feature
sbs_c1_t2 = SBS_LOO(X=X_c1_t2,
             y=y_cohort1,
             estimator=svm,
             cv=cv_sbs,
             classifiers=[svm, rf],
             min_feature=10,
             boruta_top_n_feature=1000)

#Integrate results(feature selected) of type 1 and type 2 features for cohort 1
sbs_c1_combine = calculate_integration(
    X=[X_c1_t1, X_c1_t2],
    y=y_cohort1,
    feature_selected=[sbs_c1_t1[0], sbs_c1_t2[0]],
    classifiers=[svm, rf])
