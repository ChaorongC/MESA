"""
 # @ Author: Chaorong Chen
 # @ Create Time: 2022-07-14 21:12:52
 # @ Modified by: Chaorong Chen
 # @ Modified time: 2022-07-19 21:47:00
 # @ Description: reproductivity-MESA
 """

from matplotlib.pyplot import cla
import pandas as pd
import numpy as np
import glob
from MESA import *


random_state = 0
rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
lr = LogisticRegression(
    multi_class="ovr", solver="liblinear", random_state=random_state, n_jobs=-1
)
df = CascadeForestClassifier(n_jobs=-1, random_state=0, use_predictor=True)
ensembling_clf = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("df", CascadeForestClassifier(n_jobs=-1, random_state=0, use_predictor=True)),
    ],
    voting="soft",
)

svc = SVC(kernel="linear", random_state=0, probability=True)
cv_sbs = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)


def readNconcat(dir, featureType):
    cancer_data = pd.read_table(
        glob.glob(dir + "/*.Cancer." + featureType + "*")[0], header=0, index_col=0
    )
    noncancer_data = pd.read_table(
        glob.glob(dir + "/*.Non-Cancer." + featureType + "*")[0], header=0, index_col=0
    )
    label = cancer_data.shape[1] * [1] + noncancer_data.shape[1] * [0]
    return pd.concat([cancer_data, noncancer_data], axis=1), label


"""
Corhort 1
"""
# Load data from files
cohort1_dir = "./processed_data/Cohort 1"
c1_methylation = readNconcat(cohort1_dir, "siteMethyRatio")
c1_fragmentation = readNconcat(cohort1_dir, "fragmentation_150_StoL")
c1_occupancy_pas = readNconcat(cohort1_dir, "PAS.occupancy-1kbWindow")
c1_occupancy_tss = readNconcat(cohort1_dir, "TSS.occupancy-1kbWindow")
c1_fuzziness_pas = readNconcat(cohort1_dir, "PAS.fuzziness-1kbWindow")
c1_fuzziness_tss = readNconcat(cohort1_dir, "TSS.fuzziness-1kbWindow")

# Single-modality: run analysis for each feature type
c1_methylation_result = SBS_LOO(
    X=c1_methylation[0],
    y=c1_methylation[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)
c1_fragmentation_result = SBS_LOO(
    X=c1_fragmentation[0],
    y=c1_fragmentation[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)
c1_occupancy_pas_result = SBS_LOO(
    X=c1_occupancy_pas[0],
    y=c1_occupancy_pas[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)
c1_occupancy_tss_result = SBS_LOO(
    X=c1_occupancy_tss[0],
    y=c1_occupancy_tss[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)
c1_fuzziness_pas_result = SBS_LOO(
    X=c1_fuzziness_pas[0],
    y=c1_fuzziness_pas[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)
c1_fuzziness_tss_result = SBS_LOO(
    X=c1_fuzziness_tss[0],
    y=c1_fuzziness_tss[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)

c1_occupancy_result = calculate_integration(
    X_list=[c1_occupancy_pas[0], c1_occupancy_tss[0]],
    y=c1_occupancy_pas[1],
    feature_selected=[c1_occupancy_pas_result[0], c1_occupancy_tss_result[0]],
    classifiers=[ensembling_clf],
)

c1_fuzziness_result = calculate_integration(
    X_list=[c1_fuzziness_pas[0], c1_fuzziness_tss[0]],
    y=c1_occupancy_pas[1],
    feature_selected=[c1_fuzziness_pas_result[0], c1_fuzziness_tss_result[0]],
    classifiers=[ensembling_clf],
)

# Multimodal integtation
all_X = [
    c1_methylation[0],
    c1_fragmentation[0],
    c1_occupancy_pas[0],
    c1_occupancy_tss[0],
    c1_fuzziness_pas[0],
    c1_fuzziness_tss[0],
]
all_feature_selected = [
    c1_methylation_result[0],
    c1_fragmentation_result[0],
    c1_occupancy_pas_result[0],
    c1_occupancy_tss_result[0],
    c1_fuzziness_pas_result[0],
    c1_fuzziness_tss_result[0],
]

c1_integration = calculate_integration(
    X_list=all_X,
    y=c1_methylation[1],
    feature_selected=all_feature_selected,
    classifiers=[ensembling_clf],
)

# AUC summary for cohort 1

c1_auc = pd.DataFrame(
    [
        _[-1]
        for _ in [
            c1_methylation_result,
            c1_fragmentation_result,
            c1_occupancy_result,
            c1_fuzziness_result,
            c1_integration,
        ]
    ],
    index=["Methylation", "Fragmentation", "Occupancy", "Fuzziness", "Integration"],
)

"""
Corhort 2
"""
# Load data from files
cohort2_dir = "./processed_data/Cohort 2"
c2_methylation = readNconcat(cohort2_dir, "siteMethyRatio")
c2_fragmentation = readNconcat(cohort2_dir, "fragmentation_150_StoL")
c2_occupancy_pas = readNconcat(cohort2_dir, "PAS.occupancy")
c2_occupancy_tss = readNconcat(cohort2_dir, "TSS.occupancy")
c2_fuzziness_pas = readNconcat(cohort2_dir, "PAS.fuzziness")
c2_fuzziness_tss = readNconcat(cohort2_dir, "TSS.fuzziness")


# Single-modality: run analysis for each feature type
c2_methylation_result = SBS_LOO(
    X=c2_methylation[0],
    y=c2_methylation[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)
c2_fragmentation_result = SBS_LOO(
    X=c2_fragmentation[0],
    y=c2_fragmentation[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)
c2_occupancy_pas_result = SBS_LOO(
    X=c2_occupancy_pas[0],
    y=c2_occupancy_pas[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)
c2_occupancy_tss_result = SBS_LOO(
    X=c2_occupancy_tss[0],
    y=c2_occupancy_tss[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)
c2_fuzziness_pas_result = SBS_LOO(
    X=c2_fuzziness_pas[0],
    y=c2_fuzziness_pas[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)
c2_fuzziness_tss_result = SBS_LOO(
    X=c2_fuzziness_tss[0],
    y=c2_fuzziness_tss[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)

c2_occupancy_result = calculate_integration(
    X_list=[c2_occupancy_pas[0], c2_occupancy_tss[0]],
    y=c2_occupancy_pas[1],
    feature_selected=[c2_occupancy_pas_result[0], c2_occupancy_tss_result[0]],
    classifiers=[ensembling_clf],
)

c2_fuzziness_result = calculate_integration(
    X_list=[c2_fuzziness_pas[0], c2_fuzziness_tss[0]],
    y=c2_occupancy_pas[1],
    feature_selected=[c2_fuzziness_pas_result[0], c2_fuzziness_tss_result[0]],
    classifiers=[ensembling_clf],
)

# Multimodal integtation
all_X = [
    c2_methylation[0],
    c2_fragmentation[0],
    c2_occupancy_pas[0],
    c2_occupancy_tss[0],
    c2_fuzziness_pas[0],
    c2_fuzziness_tss[0],
]
all_feature_selected = [
    c2_methylation_result[0],
    c2_fragmentation_result[0],
    c2_occupancy_pas_result[0],
    c2_occupancy_tss_result[0],
    c2_fuzziness_pas_result[0],
    c2_fuzziness_tss_result[0],
]

c2_integration = calculate_integration(
    X_list=all_X,
    y=c2_methylation[1],
    feature_selected=all_feature_selected,
    classifiers=[ensembling_clf],
)

# AUC summary for cohort 2
c2_auc = pd.DataFrame(
    [
        _[-1]
        for _ in [
            c2_methylation_result,
            c2_fragmentation_result,
            c2_occupancy_result,
            c2_fuzziness_result,
            c2_integration,
        ]
    ],
    index=["Methylation", "Fragmentation", "Occupancy", "Fuzziness", "Integration"],
)

"""
cfTAPS
"""


def readNconcat_cftaps(dir, featureType):
    ctrl_data = pd.read_table(
        glob.glob(dir + "/Ctrl." + featureType + "*")[0], header=0, index_col=0
    )
    hcc_data = pd.read_table(
        glob.glob(dir + "/HCC." + featureType + "*")[0], header=0, index_col=0
    )
    pdac_data = pd.read_table(
        glob.glob(dir + "/PDAC." + featureType + "*")[0], header=0, index_col=0
    )
    label = (
        ctrl_data.shape[1] * [0] + hcc_data.shape[1] * [1] + pdac_data.shape[1] * [2]
    )
    return pd.concat([ctrl_data, hcc_data, pdac_data], axis=1), label


cftaps_dir = "./processed_data/cfTAPS dataset"
cftaps_methylation = readNconcat(cftaps_dir, "promoterEnhancer.methyRatio")
cftaps_fragmentation = readNconcat(cftaps_dir, "Frac300-500.fragmentation")
cftaps_occupancy = readNconcat(cftaps_dir, "allRefGene.TSS-PAS.fl500.occupancy.meanF")

# 2-class classification: Control VS PDAC
## Single-modality

cftaps_PDAC_methylation_result = SBS_LOO(
    X=cftaps_methylation[0].iloc[:, np.where(np.array(cftaps_methylation[1]) != 1)[0]],
    y=np.array(cftaps_methylation[1])[
        np.where(np.array(cftaps_methylation[1]) != 1)[0]
    ],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
    boruta_top_n_feature=300,
)

cftaps_PDAC_fragmentation_result = SBS_LOO(
    X=cftaps_fragmentation[0].iloc[
        :, np.where(np.array(cftaps_fragmentation[1]) != 1)[0]
    ],
    y=np.array(cftaps_fragmentation[1])[
        np.where(np.array(cftaps_fragmentation[1]) != 1)[0]
    ],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
    boruta_top_n_feature=300,
)

cftaps_PDAC_occupancy_result = SBS_LOO(
    X=cftaps_occupancy[0].iloc[:, np.where(np.array(cftaps_occupancy[1]) != 1)[0]],
    y=np.array(cftaps_occupancy[1])[np.where(np.array(cftaps_occupancy[1]) != 1)[0]],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
    boruta_top_n_feature=300,
)

## Multimodal integtation
all_X = [
    cftaps_methylation[0].iloc[:, np.where(np.array(cftaps_methylation[1]) != 1)[0]],
    cftaps_fragmentation[0].iloc[
        :, np.where(np.array(cftaps_fragmentation[1]) != 1)[0]
    ],
    cftaps_occupancy[0].iloc[:, np.where(np.array(cftaps_occupancy[1]) != 1)[0]],
]
all_feature_selected = [
    cftaps_PDAC_methylation_result[0],
    cftaps_PDAC_fragmentation_result[0],
    cftaps_PDAC_occupancy_result[0],
]
cfTAPS_PDAC_integration = calculate_integration(
    X_list=all_X,
    y=cftaps_PDAC_methylation_result[1],
    feature_selected=all_feature_selected,
    classifiers=[ensembling_clf],
)

# AUC summary for cfTAPS-Normal Vs PDAC
cfTAPS_PDAC_auc = pd.DataFrame(
    [
        _[-1]
        for _ in [
            cftaps_PDAC_methylation_result,
            cftaps_PDAC_fragmentation_result,
            cftaps_PDAC_occupancy_result,
            cfTAPS_PDAC_integration,
        ]
    ],
    index=["Methylation", "Fragmentation", "Occupancy", "Integration"],
)

# 2-class classification: Control VS HCC
## Single-modality
cftaps_HCC_methylation_result = SBS_LOO(
    X=cftaps_methylation[0].iloc[:, np.where(np.array(cftaps_methylation[1]) != 2)[0]],
    y=np.array(cftaps_methylation[1])[
        np.where(np.array(cftaps_methylation[1]) != 2)[0]
    ],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)

cftaps_HCC_fragmentation_result = SBS_LOO(
    X=cftaps_fragmentation[0].iloc[
        :, np.where(np.array(cftaps_fragmentation[1]) != 2)[0]
    ],
    y=np.array(cftaps_fragmentation[1])[
        np.where(np.array(cftaps_fragmentation[1]) != 2)[0]
    ],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)

cftaps_HCC_occupancy_result = SBS_LOO(
    X=cftaps_occupancy[0].iloc[:, np.where(np.array(cftaps_occupancy[1]) != 2)[0]],
    y=np.array(cftaps_occupancy[1])[np.where(np.array(cftaps_occupancy[1]) != 2)[0]],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)

## Multimodal integtation
all_X = [
    cftaps_methylation[0].iloc[:, np.where(np.array(cftaps_methylation[1]) != 2)[0]],
    cftaps_fragmentation[0].iloc[
        :, np.where(np.array(cftaps_fragmentation[1]) != 2)[0]
    ],
    cftaps_occupancy[0].iloc[:, np.where(np.array(cftaps_occupancy[1]) != 2)[0]],
]
all_feature_selected = [
    cftaps_HCC_methylation_result[0],
    cftaps_HCC_fragmentation_result[0],
    cftaps_HCC_occupancy_result[0],
]
cfTAPS_HCC_integration = calculate_integration(
    X_list=all_X,
    y=cftaps_HCC_methylation_result[1],
    feature_selected=all_feature_selected,
    classifiers=[ensembling_clf],
)

# AUC summary for cfTAPS-Normal Vs HCC
cfTAPS_HCC_auc = pd.DataFrame(
    [
        _[-1]
        for _ in [
            cftaps_HCC_methylation_result,
            cftaps_HCC_fragmentation_result,
            cftaps_HCC_occupancy_result,
            cfTAPS_HCC_integration,
        ]
    ],
    index=["Methylation", "Fragmentation", "Occupancy", "Integration"],
)

# 3-class classification: Control VS HCC VS PDAC
## Single-modality
cftaps_3class_methylation_result = SBS_LOO_3class(
    X=cftaps_methylation[0],
    y=cftaps_methylation[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)

cftaps_3class_fragmentation_result = SBS_LOO_3class(
    X=cftaps_fragmentation[0],
    y=cftaps_fragmentation[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)

cftaps_3class_occupancy_result = SBS_LOO_3class(
    X=cftaps_occupancy[0],
    y=cftaps_occupancy[1],
    estimator=svc,
    cv=cv_sbs,
    classifiers=[ensembling_clf],
    min_feature=2,
)

## Multimodal integtation
all_X = [
    cftaps_methylation[0],
    cftaps_fragmentation[0],
    cftaps_occupancy[0],
]
all_feature_selected = [
    cftaps_3class_methylation_result[0],
    cftaps_3class_fragmentation_result[0],
    cftaps_3class_occupancy_result[0],
]
cfTAPS_3class_integration = calculate_integration_3class(
    X_list=all_X,
    y=cftaps_3class_methylation_result[1],
    feature_selected=all_feature_selected,
    classifiers=[ensembling_clf],
)

# Accuracy summary for cfTAPS-Normal Vs HCC
cfTAPS_3class_auc = pd.DataFrame(
    [
        _[-1]
        for _ in [
            cftaps_3class_methylation_result,
            cftaps_3class_fragmentation_result,
            cftaps_3class_occupancy_result,
            cfTAPS_3class_integration,
        ]
    ],
    index=["Methylation", "Fragmentation", "Occupancy", "Integration"],
)