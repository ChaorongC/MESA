"""
 # @ Author: Chaorong Chen
 # @ Create Time: 2022-07-14 21:12:52
 # @ Modified by: Chaorong Chen
 # @ Modified time: 2023-02-11 17:29:05
 # @ Description: reproductivity-MESA
 """

import pandas as pd
import numpy as np
import glob
from MESA_util import *


random_state = 0


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
# c1_fragmentation = readNconcat(cohort1_dir, "fragmentation_150_StoL")
c1_occupancy_pas = readNconcat(cohort1_dir, "PAS.occupancy-1kbWindow")
c1_occupancy_tss = readNconcat(cohort1_dir, "TSS.occupancy-1kbWindow")
c1_fuzziness_pas = readNconcat(cohort1_dir, "PAS.fuzziness-1kbWindow")
c1_fuzziness_tss = readNconcat(cohort1_dir, "TSS.fuzziness-1kbWindow")
c1_wps = readNconcat(cohort1_dir, "1kbSlidingWindow.Cancer.WPS")

# Single-modality: run analysis for each feature type
c1_methylation_result = MESA_single(
    X=c1_methylation[0],
    y=c1_methylation[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)

c1_occupancy_pas_result = MESA_single(
    X=c1_occupancy_pas[0],
    y=c1_occupancy_pas[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)

c1_occupancy_tss_result = MESA_single(
    X=c1_occupancy_tss[0],
    y=c1_occupancy_tss[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)


c1_fuzziness_pas_result = MESA_single(
    X=c1_fuzziness_pas[0],
    y=c1_fuzziness_pas[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)


c1_fuzziness_tss_result = MESA_single(
    X=c1_fuzziness_tss[0],
    y=c1_fuzziness_tss[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)

c1_occupancy_result = MESA_integration(
    X_list=[c1_occupancy_pas[0], c1_occupancy_tss[0]],
    y=c1_occupancy_pas[1],
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
    ],
    random_state=0,
    meta_estimator=LogisticRegression(random_state=random_state, n_jobs=-1),
    feature_selected=[c1_occupancy_pas_result[0], c1_occupancy_tss_result[0]],
)

c1_fuzziness_result = MESA_integration(
    X_list=[c1_fuzziness_pas[0], c1_fuzziness_tss[0]],
    y=c1_fuzziness_pas[1],
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
    ],
    random_state=0,
    meta_estimator=LogisticRegression(random_state=random_state, n_jobs=-1),
    feature_selected=[c1_fuzziness_pas_result[0], c1_fuzziness_tss_result[0]],
)

c1_nucleosome_result = MESA_integration(
    X_list=[
        c1_occupancy_pas[0],
        c1_occupancy_tss[0],
        c1_fuzziness_pas[0],
        c1_fuzziness_tss[0],
    ],
    y=c1_occupancy_pas[1],
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
    ],
    random_state=0,
    meta_estimator=LogisticRegression(random_state=random_state, n_jobs=-1),
    feature_selected=[
        c1_occupancy_pas_result[0],
        c1_occupancy_tss_result[0],
        c1_fuzziness_pas_result[0],
        c1_fuzziness_tss_result[0],
    ],
)


c1_wps_result = MESA_single(
    X=c1_wps[0],
    y=c1_wps[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)

# Multimodal integtation
all_X = [
    c1_methylation[0],
    c1_occupancy_pas[0],
    c1_occupancy_tss[0],
    c1_fuzziness_pas[0],
    c1_fuzziness_tss[0],
    c1_wps[0],
]
all_feature_selected = [
    c1_methylation_result[0],
    c1_occupancy_pas_result[0],
    c1_occupancy_tss_result[0],
    c1_fuzziness_pas_result[0],
    c1_fuzziness_tss_result[0],
    c1_wps_result[0],
]


c1_integration = MESA_integration(
    X_list=all_X,
    y=c1_occupancy_pas[1],
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[RandomForestClassifier(random_state=random_state, n_jobs=-1)] * 7,
    random_state=0,
    meta_estimator=LogisticRegression(random_state=random_state, n_jobs=-1),
    feature_selected=all_feature_selected,
)

# AUC summary for cohort 1

c1_auc = pd.DataFrame(
    [
        MESA_summary(_, 1)[-1]
        for _ in [
            c1_methylation_result,
            c1_occupancy_pas_result,
            c1_occupancy_tss_result,
            c1_fuzziness_pas_result,
            c1_fuzziness_tss_result,
            c1_wps_result,
        ]
    ]
    + [
        MESA_integration_summary(_)[-1]
        for _ in [c1_occupancy_result, c1_fuzziness_result, c1_integration]
    ],
    index=[
        "Methylation",
        "Occupancy_pas",
        "Occupancy_tss",
        "Fuzziness_pas",
        "Fuzziness_tss",
        "WPS",
        "Occupancy",
        "Fuzziness",
        "Integration",
    ],
    columns=["Cohort1_RandomForest_AUC"],
)


"""
Corhort 2
"""
# Load data from files
cohort2_dir = "./processed_data/Cohort 2"
c2_methylation = readNconcat(cohort2_dir, "siteMethyRatio")
# c2_fragmentation = readNconcat(cohort2_dir, "fragmentation_150_StoL")
c2_occupancy_pas = readNconcat(cohort2_dir, "PAS.occupancy")
c2_occupancy_tss = readNconcat(cohort2_dir, "TSS.occupancy")
c2_fuzziness_pas = readNconcat(cohort2_dir, "PAS.fuzziness")
c2_fuzziness_tss = readNconcat(cohort2_dir, "TSS.fuzziness")
c2_wps = readNconcat(cohort2_dir, "1kbSlidingWindow.Cancer.WPS")


# Single-modality: run analysis for each feature type
c2_methylation_result = MESA_single(
    X=c2_methylation[0],
    y=c2_methylation[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)

c2_occupancy_pas_result = MESA_single(
    X=c2_occupancy_pas[0],
    y=c2_occupancy_pas[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)

c2_occupancy_tss_result = MESA_single(
    X=c2_occupancy_tss[0],
    y=c2_occupancy_tss[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)


c2_fuzziness_pas_result = MESA_single(
    X=c2_fuzziness_pas[0],
    y=c2_fuzziness_pas[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)


c2_fuzziness_tss_result = MESA_single(
    X=c2_fuzziness_tss[0],
    y=c2_fuzziness_tss[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)

c2_occupancy_result = MESA_integration(
    X_list=[c2_occupancy_pas[0], c2_occupancy_tss[0]],
    y=c2_occupancy_pas[1],
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
    ],
    random_state=0,
    meta_estimator=LogisticRegression(random_state=random_state, n_jobs=-1),
    feature_selected=[c2_occupancy_pas_result[0], c2_occupancy_tss_result[0]],
)

c2_fuzziness_result = MESA_integration(
    X_list=[c2_fuzziness_pas[0], c2_fuzziness_tss[0]],
    y=c2_fuzziness_pas[1],
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
    ],
    random_state=0,
    meta_estimator=LogisticRegression(random_state=random_state, n_jobs=-1),
    feature_selected=[c2_fuzziness_pas_result[0], c2_fuzziness_tss_result[0]],
)

c2_nucleosome_result = MESA_integration(
    X_list=[
        c2_occupancy_pas[0],
        c2_occupancy_tss[0],
        c2_fuzziness_pas[0],
        c2_fuzziness_tss[0],
    ],
    y=c2_occupancy_pas[1],
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
    ],
    random_state=0,
    meta_estimator=LogisticRegression(random_state=random_state, n_jobs=-1),
    feature_selected=[
        c2_occupancy_pas_result[0],
        c2_occupancy_tss_result[0],
        c2_fuzziness_pas_result[0],
        c2_fuzziness_tss_result[0],
    ],
)

c2_wps_result = MESA_single(
    X=c2_wps[0],
    y=c2_wps[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)

# Multimodal integtation
all_X = [
    c2_methylation[0],
    c2_occupancy_pas[0],
    c2_occupancy_tss[0],
    c2_fuzziness_pas[0],
    c2_fuzziness_tss[0],
    c2_wps[0],
]
all_feature_selected = [
    c2_methylation_result[0],
    c2_occupancy_pas_result[0],
    c2_occupancy_tss_result[0],
    c2_fuzziness_pas_result[0],
    c2_fuzziness_tss_result[0],
    c2_wps_result[0],
]


c2_integration = MESA_integration(
    X_list=all_X,
    y=c2_occupancy_pas[1],
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[RandomForestClassifier(random_state=random_state, n_jobs=-1)] * 7,
    random_state=0,
    meta_estimator=LogisticRegression(random_state=random_state, n_jobs=-1),
    feature_selected=all_feature_selected,
)

# AUC summary for cohort 2
c2_auc = pd.DataFrame(
    [
        MESA_summary(_, 1)[-1]
        for _ in [
            c2_methylation_result,
            c2_occupancy_pas_result,
            c2_occupancy_tss_result,
            c2_fuzziness_pas_result,
            c2_fuzziness_tss_result,
            c2_wps_result,
        ]
    ]
    + [
        MESA_integration_summary(_)[-1]
        for _ in [c2_occupancy_result, c2_fuzziness_result, c2_integration]
    ],
    index=[
        "Methylation",
        "Occupancy_pas",
        "Occupancy_tss",
        "Fuzziness_pas",
        "Fuzziness_tss",
        "WPS",
        "Occupancy",
        "Fuzziness",
        "Integration",
    ],
    columns=["Cohort2_RandomForest_AUC"],
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
cftaps_occupancy = readNconcat(cftaps_dir, "allRefGene.TSS-PAS.fl500.occupancy.meanF")
# cftaps_fragmentation = readNconcat(cftaps_dir, "Frac300-500.fragmentation")
cftaps_wps = readNconcat(cftaps_dir, "allRefGene.TSS-PAS.fl500.WPS")


# 2-class classification: Control VS PDAC
## Single-modality

cftaps_PDAC_methylation_result = MESA_single(
    X=cftaps_methylation[0].iloc[:, np.where(np.array(cftaps_methylation[1]) != 1)[0]],
    y=np.array(cftaps_methylation[1])[
        np.where(np.array(cftaps_methylation[1]) != 1)[0]
    ],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)


cftaps_PDAC_occupancy_result = MESA_single(
    X=cftaps_occupancy[0].iloc[:, np.where(np.array(cftaps_occupancy[1]) != 1)[0]],
    y=np.array(cftaps_occupancy[1])[np.where(np.array(cftaps_occupancy[1]) != 1)[0]],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)


cftaps_PDAC_wps_result = MESA_single(
    X=cftaps_wps[0].iloc[:, np.where(np.array(cftaps_wps[1]) != 1)[0]],
    y=np.array(cftaps_wps[1])[np.where(np.array(cftaps_wps[1]) != 1)[0]],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)

## Multimodal integtation
all_X = [
    cftaps_methylation[0].iloc[:, np.where(np.array(cftaps_methylation[1]) != 1)[0]],
    cftaps_wps[0].iloc[:, np.where(np.array(cftaps_wps[1]) != 1)[0]],
    cftaps_occupancy[0].iloc[:, np.where(np.array(cftaps_occupancy[1]) != 1)[0]],
]
all_feature_selected = [
    cftaps_PDAC_methylation_result[0],
    cftaps_PDAC_wps_result[0],
    cftaps_PDAC_occupancy_result[0],
]

cfTAPS_PDAC_integration = MESA_integration(
    X_list=all_X,
    y=c2_occupancy_pas[1],
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[RandomForestClassifier(random_state=random_state, n_jobs=-1)] * 3,
    random_state=0,
    meta_estimator=LogisticRegression(random_state=random_state, n_jobs=-1),
    feature_selected=all_feature_selected,
)

# AUC summary for cfTAPS-Normal Vs PDAC
cfTAPS_PDAC_auc = pd.DataFrame(
    [
        MESA_summary(_, 1)[-1]
        for _ in [
            cftaps_PDAC_methylation_result,
            cftaps_PDAC_wps_result,
            cftaps_PDAC_occupancy_result,
        ]
    ]
    + [MESA_integration_summary(_)[-1] for _ in [cfTAPS_PDAC_integration]],
    index=["Methylation", "WPS", "Occupancy", "Integration"],
    columns=["PDACvsCtrl_RandomForest_AUC"],
)


# 2-class classification: Control VS HCC
## Single-modality
cftaps_HCC_methylation_result = MESA_single(
    X=cftaps_methylation[0].iloc[:, np.where(np.array(cftaps_methylation[1]) != 1)[0]],
    y=np.array(cftaps_methylation[1])[
        np.where(np.array(cftaps_methylation[1]) != 1)[0]
    ],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)


cftaps_HCC_occupancy_result = MESA_single(
    X=cftaps_occupancy[0].iloc[:, np.where(np.array(cftaps_occupancy[1]) != 1)[0]],
    y=np.array(cftaps_occupancy[1])[np.where(np.array(cftaps_occupancy[1]) != 1)[0]],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)

cftaps_HCC_wps_result = MESA_single(
    X=cftaps_wps[0].iloc[:, np.where(np.array(cftaps_wps[1]) != 1)[0]],
    y=np.array(cftaps_wps[1])[np.where(np.array(cftaps_wps[1]) != 1)[0]],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
)
## Multimodal integtation
all_X = [
    cftaps_methylation[0].iloc[:, np.where(np.array(cftaps_methylation[1]) != 1)[0]],
    cftaps_wps[0].iloc[:, np.where(np.array(cftaps_wps[1]) != 1)[0]],
    cftaps_occupancy[0].iloc[:, np.where(np.array(cftaps_occupancy[1]) != 1)[0]],
]
all_feature_selected = [
    cftaps_HCC_methylation_result[0],
    cftaps_HCC_wps_result[0],
    cftaps_HCC_occupancy_result[0],
]

cfTAPS_HCC_integration = MESA_integration(
    X_list=all_X,
    y=c2_occupancy_pas[1],
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[RandomForestClassifier(random_state=random_state, n_jobs=-1)] * 3,
    random_state=0,
    meta_estimator=LogisticRegression(random_state=random_state, n_jobs=-1),
    feature_selected=all_feature_selected,
)

# AUC summary for cfTAPS-Normal Vs HCC
cfTAPS_HCC_auc = pd.DataFrame(
    [
        MESA_summary(_, 1)[-1]
        for _ in [
            cftaps_HCC_methylation_result,
            cftaps_HCC_wps_result,
            cftaps_HCC_occupancy_result,
        ]
    ]
    + [MESA_integration_summary(_)[-1] for _ in [cfTAPS_HCC_integration]],
    index=["Methylation", "WPS", "Occupancy", "Integration"],
    columns=["HCCvsCtrl_RandomForest_AUC"],
)

# 3-class classification: Control VS HCC VS PDAC
## Single-modality
cftaps_3class_methylation_result = MESA_single(
    X=cftaps_methylation[0],
    y=cftaps_methylation[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
    multiclass=True,
)


cftaps_3class_wps_result = MESA_single(
    X=cftaps_wps[0],
    y=cftaps_wps[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
    multiclass=True,
)


cftaps_3class_occupancy_result = MESA_single(
    X=cftaps_occupancy[0],
    y=cftaps_occupancy[1],
    boruta_est=RandomForestClassifier(random_state=random_state, n_jobs=-1),
    cv=LeaveOneOut(),
    classifiers=[RandomForestClassifier(random_state=random_state, n_jobs=-1)],
    random_state=0,
    boruta_top_n_feature=100,
    variance_threshold=0,
    missing_ratio=1,
    normalization=False,
    multiclass=True,
)

## Multimodal integtation
all_X = [
    cftaps_methylation[0],
    cftaps_wps[0],
    cftaps_occupancy[0],
]
all_feature_selected = [
    cftaps_3class_methylation_result[0],
    cftaps_3class_wps_result[0],
    cftaps_3class_occupancy_result[0],
]

cfTAPS_3class_integration = MESA_integration(
    X_list=all_X,
    y=cftaps_3class_methylation_result[1],
    cv=LeaveOneOut(),
    missing_ratio=1,
    normalization=False,
    estimator_list=[RandomForestClassifier(random_state=random_state, n_jobs=-1)] * 7,
    random_state=0,
    meta_estimator=LogisticRegression(random_state=random_state, n_jobs=-1),
    feature_selected=all_feature_selected,
    multiclass=True,
)

# Accuracy summary for cfTAPS-Normal Vs HCC
cfTAPS_3class_auc = pd.DataFrame(
    [
        MESA_summary(_, 1, True)[-1]
        for _ in [
            cftaps_3class_methylation_result,
            cftaps_3class_wps_result,
            cftaps_3class_occupancy_result,
        ]
    ]
    + [MESA_integration_summary(_, True)[-1] for _ in [cfTAPS_3class_integration]],
    index=["Methylation", "WPS", "Occupancy", "Integration"],
    columns=["cfTAPS_3class_RandomForest_ACC"],
)


"""
Probability for ROC Curves
"""
# Figure 4
## Cohort 1
c1_nucleosome_probability = pd.DataFrame(
    [MESA_summary(c1_occupancy_result, 1)[0]]
    + [
        MESA_summary(_, 1)[1]
        for _ in [
            c1_occupancy_pas_result,
            c1_occupancy_tss_result,
            c1_fuzziness_pas_result,
            c1_fuzziness_tss_result,
        ]
    ]
    + [
        MESA_integration_summary(_)[0]
        for _ in [
            c1_occupancy_result,
            c1_fuzziness_result,
            c1_nucleosome_result,
        ]
    ],
    index=[
        "Label",
        "Occupancy_PAS",
        "Occupancy_TSS",
        "Fuzziness_PAS",
        "Fuzziness_TSS",
        "Occupancy_PAS+TSS",
        "Fuzziness_PAS+TSS",
        "Occupancy+Fuzziness",
    ],
    columns=c1_occupancy_result[0].columns,
).T.to_csv("Cohort1_nucleosome_probability.csv", index=True)
## Cohort 2
c2_nucleosome_probability = pd.DataFrame(
    [MESA_summary(c2_occupancy_result, 1)[0]]
    + [
        MESA_summary(_, 1)[1]
        for _ in [
            c2_occupancy_pas_result,
            c2_occupancy_tss_result,
            c2_fuzziness_pas_result,
            c2_fuzziness_tss_result,
        ]
    ]
    + [
        MESA_integration_summary(_)[0]
        for _ in [
            c2_occupancy_result,
            c2_fuzziness_result,
            c2_nucleosome_result,
        ]
    ],
    index=[
        "Label",
        "Occupancy_PAS",
        "Occupancy_TSS",
        "Fuzziness_PAS",
        "Fuzziness_TSS",
        "Occupancy_PAS+TSS",
        "Fuzziness_PAS+TSS",
        "Occupancy+Fuzziness",
    ],
    columns=c2_occupancy_result[0].columns,
).T.to_csv("Cohort2_nucleosome_probability.csv", index=True)

# Figure 5
## Cohort 1
c1_probability = pd.DataFrame(
    [MESA_summary(c1_methylation_result, 1)[0]]
    + [
        MESA_summary(_, 1)[1]
        for _ in [
            c1_methylation_result,
            c1_wps_result,
        ]
    ]
    + [
        MESA_integration_summary(_)[0]
        for _ in [
            c1_occupancy_result,
            c1_fuzziness_result,
            c1_integration,
        ]
    ],
    index=[
        "Label",
        "Methylation",
        "WPS",
        "Occupancy",
        "Fuzziness",
        "Multimodal",
    ],
    columns=c1_methylation[0].columns,
).T.to_csv("Cohort1_probability.csv", index=True)

## Cohort 2
c2_probability = pd.DataFrame(
    [MESA_summary(c2_methylation_result, 1)[0]]
    + [
        MESA_summary(_, 1)[1]
        for _ in [
            c2_methylation_result,
            c2_wps_result,
        ]
    ]
    + [
        MESA_integration_summary(_)[0]
        for _ in [
            c2_occupancy_result,
            c2_fuzziness_result,
            c2_integration,
        ]
    ],
    index=[
        "Label",
        "Methylation",
        "WPS",
        "Occupancy",
        "Fuzziness",
        "Multimodal",
    ],
    columns=c2_methylation[0].columns,
).T.to_csv("Cohort2_probability.csv", index=True)

# Figure 6
## HCC vs. Control
cftaps_HCC_probability = pd.DataFrame(
    [MESA_summary(cftaps_HCC_methylation_result, 1)[0]]
    + [
        MESA_summary(_, 1)[1]
        for _ in [
            cftaps_HCC_methylation_result,
            cftaps_HCC_wps_result,
            cftaps_HCC_occupancy_result,
        ]
    ]
    + [MESA_integration_summary(_)[0] for _ in [cfTAPS_HCC_integration]],
    index=["Label", "Methylation", "WPS", "Occupancy", "Multimodal"],
    columns=cftaps_methylation[0]
    .iloc[:, np.where(np.array(cftaps_methylation[1]) != 2)[0]]
    .columns,
).T.to_csv("cfTAPS_HCC_probability.csv", index=True)

## PDAC vs. Control
cftaps_PDAC_probability = pd.DataFrame(
    [MESA_summary(cftaps_PDAC_methylation_result, 1)[0]]
    + [
        MESA_summary(_, 1)[1]
        for _ in [
            cftaps_PDAC_methylation_result,
            cftaps_PDAC_wps_result,
            cftaps_PDAC_occupancy_result,
        ]
    ]
    + [MESA_integration_summary(_)[0] for _ in [cfTAPS_PDAC_integration]],
    index=["Label", "Methylation", "WPS", "Occupancy", "Multimodal"],
    columns=cftaps_methylation[0]
    .iloc[:, np.where(np.array(cftaps_methylation[1]) != 2)[0]]
    .columns,
).T.to_csv("cfTAPS_PDAC_probability.csv", index=True)

## 3-class predicted label
cftaps_3class_probability = pd.DataFrame(
    [MESA_summary(cftaps_3class_methylation_result, 1, True)[0]]
    + [
        MESA_summary(_, 1, True)[1]
        for _ in [
            cftaps_3class_methylation_result,
            cftaps_3class_wps_result,
            cftaps_3class_occupancy_result,
        ]
    ]
    + [MESA_integration_summary(_, True)[0] for _ in [cfTAPS_3class_integration]],
    index=["Label", "Methylation", "WPS", "Occupancy", "Multimodal"],
    columns=cftaps_methylation[0].columns,
).T.to_csv("cfTAPS_3class_probability.csv", index=True)
