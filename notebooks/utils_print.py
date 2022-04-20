# Item name in the paper
i_name = "α"  # or i
# Pattern or itemset name in the paper
p_name = "I"
# Name for diverge in the paper
div_name = "Δ"


def printable(df_print, cols=["itemsets"], abbreviations={}, n_rows=3, decimals=(2, 3)):

    if type(decimals) is tuple:
        r1, r2 = decimals[0], decimals[1]
    else:
        r1, r2 = decimals, decimals
    df_print = df_print.copy()
    if "support" in df_print.columns:
        df_print["support"] = df_print["support"].round(r1)
    t_v = [c for c in df_print.columns if "t_value_" in c]
    if t_v:
        df_print[t_v] = df_print[t_v].round(1)
    df_print = df_print.round(r2)
    df_print.rename(columns={"support": "sup"}, inplace=True)
    df_print.columns = df_print.columns.str.replace("d_*", f"{div_name}_", regex=False)
    df_print.columns = df_print.columns.str.replace("t_value", "t")
    for c in cols:
        df_print[c] = df_print[c].apply(lambda x: sortItemset(x, abbreviations))
    # TODO
    cols = list(df_print.columns)
    cols = [cols[1], cols[0]] + cols[2:]
    return df_print[cols]


def sortItemset(x, abbreviations={}):
    x = list(x)
    x.sort()
    x = ", ".join(x)
    for k, v in abbreviations.items():
        x = x.replace(k, v)
    return x


def printableCorrective(corr_df, metric_name, n_rows=5, abbreviations={}, colsOfI=None):
    colsOfI = (
        ["S", "item i", "v_S", "v_S+i", "corr_factor", "t_value_corr"]
        if colsOfI is None
        else colsOfI
    )
    corr_df = corr_df[[c for c in colsOfI if c in corr_df.columns]]
    corr_df_pr = printable(
        corr_df.head(n_rows), cols=["item i", "S"], abbreviations=abbreviations
    )
    corr_df_pr.rename(
        columns={
            "item i": f"corrective item {i_name}",
            "S": f"{p_name}",
            "v_i": f"{metric_name}({i_name})",
            "v_S": f"{metric_name}({p_name})",
            "v_S+i": f"{metric_name}({p_name} U {i_name})",
            "t_S+i": f"t_{p_name} U {i_name}",
        },
        inplace=True,
    )

    return corr_df_pr


def printableAll(dfs):
    import pandas as pd

    div_all = pd.DataFrame()
    for df in dfs:
        df_i = df.rename(columns={"d_fpr": "FPR", "d_fnr": "FNR", "d_accuracy": "ACC"})
        df_i = df_i.T.reset_index().T
        div_all = div_all.append(df_i, ignore_index=True)
    return div_all.rename(columns=div_all.iloc[0]).drop(div_all.index[0])
