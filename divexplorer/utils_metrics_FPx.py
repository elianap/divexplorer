import numpy as np


def averageScore(df_score_res, score_col, count_col):
    return (df_score_res[score_col] / (df_score_res[count_col])).fillna(0)


def tpr_df(df_cm):
    return (df_cm["tp"] / (df_cm["tp"] + df_cm["fn"])).fillna(0)


def fpr_df(df_cm):
    return (df_cm["fp"] / (df_cm["fp"] + df_cm["tn"])).fillna(0)


def fnr_df(df_cm):
    return (df_cm["fn"] / (df_cm["tp"] + df_cm["fn"])).fillna(0)


def tnr_df(df_cm):
    return (df_cm["tn"] / (df_cm["fp"] + df_cm["tn"])).fillna(0)


def posr_df(df_cm):
    return (df_cm["tp"] + df_cm["fn"]) / (
        df_cm["tp"] + df_cm["fp"] + df_cm["tn"] + df_cm["fn"]
    )


def negr_df(df_cm):
    return (df_cm["tn"] + df_cm["fp"]) / (
        df_cm["tp"] + df_cm["fp"] + df_cm["tn"] + df_cm["fn"]
    )


def accuracy_df(df_cm):
    return (df_cm["tn"] + df_cm["tp"]) / (
        df_cm["tp"] + df_cm["fp"] + df_cm["tn"] + df_cm["fn"]
    )


def classiferror_df(df_cm):
    return (df_cm["fn"] + df_cm["fp"]) / (
        df_cm["tp"] + df_cm["fp"] + df_cm["tn"] + df_cm["fn"]
    )


def positive_predicted_value_df(df_cm):
    return (df_cm["tp"] / (df_cm["tp"] + df_cm["fp"])).fillna(0)


def classification_error_df(df_cm):
    return 1 - (df_cm["tn"] + df_cm["tp"]) / (
        df_cm["tp"] + df_cm["fp"] + df_cm["tn"] + df_cm["fn"]
    )


def true_positive_rate_df(df_cm):
    return (df_cm["tp"] / (df_cm["tp"] + df_cm["fn"])).fillna(0)


def true_negative_rate_df(df_cm):
    return (df_cm["tn"] / (df_cm["tn"] + df_cm["fp"])).fillna(0)


def negative_predicted_value_df(df_cm):
    return (df_cm["tn"] / (df_cm["tn"] + df_cm["fn"])).fillna(0)


def false_discovery_rate_df(df_cm):
    return (df_cm["fp"] / (df_cm["fp"] + df_cm["tp"])).fillna(0)


def false_omission_rate_df(df_cm):
    return (df_cm["fn"] / (df_cm["fn"] + df_cm["tn"])).fillna(0)


def get_pos(df_cm):
    return df_cm["tp"] + df_cm["fn"]


def get_neg(df_cm):
    return df_cm["tn"] + df_cm["fp"]

def precision_df(df_cm):
    return (df_cm["tp"]) / (
        df_cm["tp"] + df_cm["fp"]
    )

def recall_df(df_cm):
    return (df_cm["tp"]) / (
        df_cm["tp"] + df_cm["fn"]
    )

def f1_score_df(df_cm):
    return (2*df_cm["tp"]) / (
        2*df_cm["tp"] + df_cm["fp"] + df_cm["fn"]
    )


def getInfoRoot(df):
    return df.loc[df["itemsets"] == frozenset()]


def statParitySubgroupFairness(df):
    root_info = getInfoRoot(df)
    alfaSP = df["support"]
    SP_D = (root_info["tp"].values[0] + root_info["fp"].values[0]) / root_info[
        "support_count"
    ].values[0]
    SP_DG = (df["tp"] + df["fp"]) / df["support_count"]
    betaSP = abs(SP_D - SP_DG)
    df["SPsf"] = alfaSP * betaSP
    return df


def FPSubgroupFairness(df):
    root_info = getInfoRoot(df)
    alfaFP = (df["tn"] + df["fp"]) / root_info["support_count"].values[0]
    FP_D = root_info["fp"].values[0] / (
        root_info["fp"].values[0] + root_info["tn"].values[0]
    )
    # Redundant, equalt to fpr_rate
    FP_DG = (df["fp"]) / (df["fp"] + df["tn"])
    betaFP = abs(FP_D - FP_DG)
    df["FPsf"] = alfaFP * betaFP
    return df


def FNSubgroupFairness(df):
    root_info = getInfoRoot(df)
    alfaFN = (df["tp"] + df["fn"]) / root_info["support_count"].values[0]
    FN_D = root_info["fn"].values[0] / (
        root_info["fn"].values[0] + root_info["tp"].values[0]
    )
    # Redundant, equalt to fnr_rate
    FN_DG = (df["fn"]) / (df["fn"] + df["tp"])
    betaFN = abs(FN_D - FN_DG)
    df["FNsf"] = alfaFN * betaFN
    return df


def getAccuracyDF(df):
    df["accuracy"] = (df["tp"] + df["tn"]) / (df["tp"] + df["tn"] + df["fn"] + df["fp"])
    return df


def AccuracySubgroupFairness(df):
    if "accuracy" not in df.columns:
        df = getAccuracyDF(df)
    root_info = getInfoRoot(df)
    alfaAC = df["support"]
    AC_D = (root_info["tp"].values[0] + root_info["tn"].values[0]) / (
        root_info["support_count"].values[0]
    )
    AC_DG = df["accuracy"]
    if "d_accuracy" not in df:
        df["d_accuracy"] = df["accuracy"] - AC_D
        # df["d_accuracy_abs"]=abs(df["accuracy"]-AC_D)

    betaAC = abs(AC_D - AC_DG)
    df["ACsf"] = alfaAC * betaAC
    return df
