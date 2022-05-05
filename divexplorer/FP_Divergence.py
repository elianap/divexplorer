def getItemsetMetric(freq_metrics, metric="d_fnr"):
    d = freq_metrics[["itemsets", metric]].set_index("itemsets").to_dict("index")
    return {
        k: {k1: v[metric] for k1, v in d.items() if k == len(k1)}
        for k in range(0, max(freq_metrics["length"] + 1))
    }


def getItemsetMetrics(freq_metrics, metrics=["d_fnr", "support"]):
    d = freq_metrics[["itemsets"] + metrics].set_index("itemsets").to_dict("index")
    return {
        k: {k1: v for k1, v in d.items() if k == len(k1)}
        for k in range(0, max(freq_metrics["length"] + 1))
    }


def getLenDictionaries(dictionary, sortKey=False):
    lenDict = {
        len(k): {x: dictionary[x] for x in dictionary if len(x) == len(k)}
        for k in dictionary
    }
    if sortKey:
        return {k: v for k, v in sorted(lenDict.items(), key=lambda item: item[0])}
    return lenDict


def sortItemset(x, abbreviations={}):
    x = list(x)
    x.sort()
    x = ", ".join(x)
    for k, v in abbreviations.items():
        x = x.replace(k, v)
    return x


def abbreviateDict(d, abbreviations):
    # Shapley values (dict) as input
    return {
        frozenset([sortItemset(k, abbreviations=abbreviations)]): v
        for k, v in d.items()
    }


from distutils.log import warn
from .shapley_value_FPx import (
    shapley_subset,
    computeShapleyItemset,
    computeDeltaDiffShap,
)

from .lattice_graph import (
    getLatticeItemsetMetric,
    plotLatticeGraph_colorGroups,
)


i_col = "item i"
delta_col = "delta_item"
v_si_col = "v_S+i"
v_s_col = "v_S"
corr_coef = "corr_coef"
corr_coef_sq = "corr_coef_sq"
s_col = "S"
corr_coef_mse = "corr_coef_mse"
MSE_col = "MSE"
SSE_col = "SSE"
SE_col = "SE"

map_metric = {"ACsf": "accuracy", "SPsf": "accuracy", "FPsf": "fp", "FNsf": "fn"}
map_beta_distribution = {
    "d_fpr": {"T": ["fp"], "F": ["tn"]},
    "d_fnr": {"T": ["fn"], "F": ["tp"]},
    "d_accuracy": {"T": ["tp", "tn"], "F": ["fp", "fn"]},
    "d_fpr_abs": {"T": ["fp"], "F": ["tn"]},
    "d_fnr_abs": {"T": ["fn"], "F": ["tp"]},
    "d_accuracy_abs": {"T": ["tp", "tn"], "F": ["fp", "fn"]},
    "d_posr": {"T": ["tp", "fn"], "F": ["tn", "fp"]},
    "d_negr": {"T": ["tn", "fp"], "F": ["tp", "fn"]},
    "d_error": {"T": ["fp", "fn"], "F": ["tp", "tn"]},
    "d_ppv": {"T": ["tp"], "F": ["fp"]},
    "d_tpr": {"T": ["tp"], "F": ["fn"]},
    "d_tnr": {"T": ["tn"], "F": ["fp"]},
    "d_npv": {"T": ["tn"], "F": ["fn"]},
    "d_fdr": {"T": ["fp"], "F": ["tp"]},
    "d_for": {"T": ["fn"], "F": ["tn"]},
    "d_precision": {"T": ["tp"], "F": ["fp"]},
    "d_recall": {"T": ["tp"], "F": ["fn"]},
    "d_f1": {"T": ["tp", "tp"], "F": ["fp", "fn"]}

}

VIZ_COL_NAME = "viz"

# TODO --> move
def _compute_t_test(df, col_mean, col_var, mean_d, var_d):
    return (abs(df[col_mean] - mean_d)) / ((df[col_var] + var_d) ** 0.5)


def _compute_std_beta_distribution(FPb):
    return ((FPb.a * FPb.b) / ((FPb.a + FPb.b) ** 2 * (FPb.a + FPb.b + 1))) ** (1 / 2)


def _compute_variance_beta_distribution(FPb):
    return (FPb.a * FPb.b) / ((FPb.a + FPb.b) ** 2 * (FPb.a + FPb.b + 1))


def _compute_mean_beta_distribution(FPb):
    return FPb.a / (FPb.a + FPb.b)


# Item name in the paper
i_name = "α"  # or i
# Pattern or itemset name in the paper
p_name = "I"
# Name for diverge in the paper
div_name = "Δ"

D_OUTCOME = "d_outcome"
AVG_OUTCOME = "outcome"

class FP_Divergence:
    def __init__(self, freq_metrics, metric):
        self.freq_metrics = freq_metrics
        self.metric = metric
        self.cl_metric = (
            self.metric.split("_")[1] if "_" in self.metric else map_metric[self.metric]
        )
        self.itemset_divergence = getItemsetMetric(freq_metrics, metric)
        self.df_delta = None
        self.global_shapley = None
        self.corr_df = None
        self.itemset_divergence_not_redundant = None
        self.itemset_divergence_not_redundant_df = None
        self.corr_statistics_df = None
        self.deltas_statistics_df = None
        self.metric_name = (
            "_".join(self.metric.split("_")[1:]).upper()
            if self.metric.startswith("d_")
            else self.metric.replace("_", "\\_")
        )
        self.t_value_col = (
            f"t_value_{'_'.join(map_beta_distribution[self.metric]['T'])}"
            if self.metric in map_beta_distribution
            else None
        )
        self.corrSignif = None

    def getItemsetDivergence(self, itemsetI):
        itemsetI = frozenset(itemsetI) if type(itemsetI) == list else itemsetI
        return self.itemset_divergence[len(itemsetI)][itemsetI]

    def getKVItemsetsDivergence(self):
        return (
            self.freq_metrics[["itemsets", self.metric]]
            .set_index("itemsets")[self.metric]
            .to_dict()
        )

    def getTvalues(self):
        if self.t_value_col not in self.freq_metrics.columns:
            self.t_test(ret=False)
        return (
            self.freq_metrics[["itemsets", self.t_value_col]]
            .set_index("itemsets")
            .to_dict()[self.t_value_col]
        )

    # TODO: getLower to showCorrective
    def plotLatticeItemset(
        self,
        itemset,
        Th_divergence=None,
        getLower=False,
        getAllGreaterTh=False,
        **kwargs,
    ):

        nameTitle = f"Metric: {self.metric}"
        info_lattice = getLatticeItemsetMetric(
            itemset, self.itemset_divergence, getLower=getLower
        )
        color_groups = {}
        nodes = info_lattice["itemset_metric"]
        # Save info node - parent source
        # node_sources={}
        if Th_divergence is not None:
            nameTitle = f"{nameTitle} - Threshold: {Th_divergence}"
            color_groups["greater"] = [
                k for k, v in nodes.items() if abs(v) >= Th_divergence
            ]
        if getLower:
            nameTitle = f"{nameTitle} - show lower"
            color_groups["lower"] = info_lattice["lower"]
        if getAllGreaterTh and Th_divergence is not None:
            color_groups["all_greater"] = []
            for node in [
                k for k, v in nodes.items() if abs(v) >= Th_divergence
            ]:  # color_groups["greater"]:
                if [p for p in color_groups["all_greater"] if p.issubset(node)] == []:
                    if [
                        k
                        for k, v in nodes.items()
                        if abs(v) < Th_divergence and node.issubset(k)
                    ] == []:
                        color_groups["all_greater"].append(node)
                # Save info node - parent source
                # else:
                # node_sources[node]=[p for p in color_groups["all_greater"] if p.issubset(node)]
        color_groups["normal"] = list(
            set(nodes) - set([v for v1 in color_groups.values() for v in v1])
        )
        color_map = {
            "normal": "#6175c1",
            "lower": "lightblue",
            "greater": "#ff6666",
            "all_greater": "#580023",
        }

        return plotLatticeGraph_colorGroups(
            info_lattice["lattice_graph"],
            info_lattice["itemset_metric"],
            color_groups,
            metric=nameTitle,
            color_map=color_map,
            **kwargs,
        )

    def getFItemsetsDivergence(self, redundant=True):
        if redundant:
            return self.itemset_divergence
        else:
            if self.itemset_divergence_not_redundant is not None:
                return self.itemset_divergence_not_redundant
            return self.getFItemsetsDivergenceNotRedundant(lenFormat=True)

    def getFItemsetsDivergenceNotRedundant(self, lenFormat=False):
        if self.itemset_divergence_not_redundant is not None:
            return self.itemset_divergence_not_redundant
        itemset_divergence_not_redundant_df = (
            self.getFItemsetsDivergenceDfNotRedundant()
        )
        itemset_divergence_not_redundant = (
            itemset_divergence_not_redundant_df.set_index("itemsets").T.to_dict("int")[
                self.metric
            ]
        )
        self.itemset_divergence_not_redundant = getLenDictionaries(
            itemset_divergence_not_redundant
        )
        return (
            self.itemset_divergence_not_redundant
            if lenFormat
            else itemset_divergence_not_redundant
        )

    def getFItemsetsDivergenceDfNotRedundant(self):
        def removeRedundant(df, a):
            import pandas as pd

            grouped_itemset = list(df.itemsets.values)
            d = pd.DataFrame(
                {
                    "itemsets": [
                        grouped_itemset[i]
                        for i in range(0, len(grouped_itemset))
                        if len(
                            [
                                k
                                for k in grouped_itemset[0:i]
                                if k.issubset(grouped_itemset[i])
                            ]
                        )
                        == 0
                    ]
                }
            )
            d[a] = df.name
            return d

        if self.itemset_divergence_not_redundant_df is not None:
            return self.itemset_divergence_not_redundant_df

        dfs = self.freq_metrics.sort_values(
            [self.metric, "length"], ascending=[False, True]
        )[["itemsets", self.metric]]
        dfs_g = dfs.copy()
        dfs_g.loc[dfs_g.loc[dfs_g[self.metric].isnull()].index, self.metric] = "NaN"
        grouped = dfs_g.groupby(self.metric, group_keys=False).apply(
            removeRedundant, self.metric
        )
        import math

        grouped = grouped.replace({self.metric: "NaN"}, float("NaN"))
        not_red = grouped.sort_values(self.metric, ascending=False).reset_index(
            drop=True
        )
        self.itemset_divergence_not_redundant_df = not_red
        return self.itemset_divergence_not_redundant_df

    def getDivergenceMetricNotRedundant(self, th_redundancy=0, sortV=True):
        if th_redundancy is None:
            return self.freq_metrics
        df_corr = self.getDfDeltaShapleyValue()
        redundant = df_corr.loc[abs(df_corr.delta_item) <= th_redundancy]
        redundant_itemsets = set(redundant["S+i"].values)
        # freq_metric_Red=self.freq_metrics.loc[self.freq_metrics.itemsets.isin(redundant_itemsets)]
        freq_metric_NotRed = self.freq_metrics.loc[
            self.freq_metrics.itemsets.isin(redundant_itemsets) == False
        ]
        if sortV:
            return freq_metric_NotRed.sort_values(
                [self.metric, self.cl_metric], ascending=False
            )
        else:
            return freq_metric_NotRed

    def getRedundantMarginalContribution(self, th_redundancy=0):
        df_corr = self.getDfDeltaShapleyValue()
        return df_corr.loc[abs(df_corr.delta_item) <= th_redundancy]

    def getInfoItemset(self, itemset):
        if type(itemset) == list:
            itemset = frozenset(itemset)
        return self.freq_metrics.loc[self.freq_metrics.itemsets == itemset]

    def getInfoItemsets(self, list_itemsets):
        if type(list_itemsets[0]) == list:
            list_itemsets = [frozenset(itemset) for itemset in list_itemsets]
        return self.freq_metrics.loc[
            self.freq_metrics.itemsets.apply(lambda x: x in list_itemsets)
        ]

    def getFMetricGreaterTh(
        self, T_thr=0.1, lenFormat=False, absValue=True, sortedV=False
    ):
        if absValue:
            greaterT = {
                k2: v2
                for k, v in self.itemset_divergence.items()
                for k2, v2 in v.items()
                if abs(v2) >= T_thr
            }
        else:
            greaterT = {
                k2: v2
                for k, v in self.itemset_divergence.items()
                for k2, v2 in v.items()
                if v2 > T_thr
            }
        if sortedV:
            greaterT = {
                k: v
                for k, v in sorted(
                    greaterT.items(), key=lambda item: item[1], reverse=True
                )
            }
        return getLenDictionaries(greaterT) if lenFormat else greaterT

    def getDivergenceTopK(self, K=10, lenFormat=False, th_redundancy=None, absF=False):
        # item_s_flat={k2: v for k in self.itemset_divergence for k2, v in  self.itemset_divergence[k].items()}
        # topK={k: v for k, v in sorted(item_s_flat.items(), key=lambda item: item[1], reverse=True)[:K]}
        scores = (
            self.freq_metrics[["itemsets", self.metric, self.cl_metric]]
            if th_redundancy is None
            else self.getDivergenceMetricNotRedundant(th_redundancy=th_redundancy)
        )

        if absF:
            topKDF = scores.iloc[scores[self.metric].abs().argsort()[::-1]][
                ["itemsets", self.metric]
            ].head(K)
        else:
            topKDF = scores.sort_values([self.metric, self.cl_metric], ascending=False)[
                ["itemsets", self.metric]
            ].head(K)
        topK = topKDF.set_index("itemsets").to_dict()[self.metric]

        return getLenDictionaries(topK) if lenFormat else topK

    def getDivergenceTopKDf(self, K=10, th_redundancy=None, absF=False):
        # item_s_flat={k2: v for k in self.itemset_divergence for k2, v in  self.itemset_divergence[k].items()}
        # topK={k: v for k, v in sorted(item_s_flat.items(), key=lambda item: item[1], reverse=True)[:K]}
        # OK:scores=self.freq_metrics if redundant else self.getFItemsetsDivergenceDfNotRedundant()
        scores = (
            self.freq_metrics[["itemsets", self.metric, self.cl_metric]]
            if th_redundancy is None
            else self.getDivergenceMetricNotRedundant(th_redundancy=th_redundancy)[
                ["itemsets", self.metric, self.cl_metric]
            ]
        )
        if absF:
            topKDF = scores.iloc[scores[self.metric].abs().argsort()[::-1]][
                ["itemsets", self.metric]
            ].head(K)
        else:
            topKDF = scores.sort_values([self.metric, self.cl_metric], ascending=False)[
                ["itemsets", self.metric]
            ].head(K)
        return topKDF

    def getDivergence(self, th_redundancy=None, absF=False):
        scores = (
            self.freq_metrics
            if th_redundancy is None
            else self.getDivergenceMetricNotRedundant(th_redundancy=th_redundancy)
        )
        if absF:
            sortedDF = scores.iloc[scores[self.metric].abs().argsort()[::-1]]
        else:
            sortedDF = scores.sort_values(
                [self.metric, self.cl_metric], ascending=False
            )
        return sortedDF

    def getFMetricSortedTopK(self, K, th_redundancy=None, absF=False):
        sortedDF = self.getDivergence(th_redundancy=th_redundancy, absF=absF)
        return sortedDF.head(K)

    def getFMetricSortedGreaterTh(self, thr_divergence, th_redundancy=None, absF=False):
        sortedDF = self.getDivergence(th_redundancy=th_redundancy, absF=absF)
        if absF:
            sortedDFGreaterTh = sortedDF.loc[
                abs(sortedDF[self.metric]) >= thr_divergence
            ]
        else:
            sortedDFGreaterTh = sortedDF.loc[sortedDF[self.metric] >= thr_divergence]
        return sortedDFGreaterTh

    def getFMetricSortedGreaterThTopK(
        self, K, thr_divergence, th_redundancy=None, absF=False
    ):
        sortedDFGreaterTh = self.getFMetricSortedGreaterTh(
            thr_divergence, th_redundancy=th_redundancy, absF=absF
        )
        return sortedDFGreaterTh.head(K)

    def getFMetricGreaterThTopK(self, T_thr=0.1, K=10, lenFormat=False):
        greaterT = self.getFMetricGreaterTh(T_thr=T_thr)
        topK = {
            k: v
            for k, v in sorted(
                greaterT.items(), key=lambda item: item[1], reverse=True
            )[:K]
        }
        return getLenDictionaries(topK) if lenFormat else topK

    def computeShapleyValue(self, itemset):
        return shapley_subset(itemset, self.itemset_divergence)

    def plotShapleyValue(
        self,
        itemset=None,
        shapley_values=None,
        sortedF=True,
        metric="",
        nameFig=None,
        saveFig=False,
        height=0.5,
        linewidth=0.8,
        sizeFig=(4, 3),
        labelsize=10,
        titlesize=10,
        title=None,
        abbreviations={},
        xlabel=False,
        show_figure=True
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=sizeFig, dpi=100)

        if shapley_values is None and itemset is None:
            # todo
            raise ValueError("Specify the itemset or the precomputed Shapley values (dict)")

        if shapley_values is None and itemset:
            shapley_values = self.computeShapleyValue(itemset)

        if abbreviations:
            shapley_values = abbreviateDict(shapley_values, abbreviations)
        sh_plt = {str(",".join(list(k))): v for k, v in shapley_values.items()}
        metric = f"{div_name}_{{{self.metric_name}}}" if metric is None else metric
        if sortedF:
            sh_plt = {k: v for k, v in sorted(sh_plt.items(), key=lambda item: item[1])}
        ax.barh(
            range(len(sh_plt)),
            sh_plt.values(),
            height=height,
            align="center",
            color="#7CBACB",
            linewidth=linewidth,
            edgecolor="#0C4A5B",
        )
        ax.set_yticks(range(len(sh_plt)), minor=False)
        ax.set_yticklabels(list(sh_plt.keys()), minor=False)
        ax.tick_params(axis="y", labelsize=labelsize)

        if xlabel:
            ax.set_xlabel(f"${div_name}({i_name}|{p_name})$", size=labelsize)
             # - Divergence contribution
        
        
        title = "" if title is None else title
        title = f"{title} ${metric}$" if metric != "" else title  # Divergence


        ax.set_title(title, fontsize=titlesize)

        if saveFig:
            nameFig = "./shap.pdf" if nameFig is None else nameFig
            # plt.savefig(f"{nameFig}", bbox_inches="tight", pad=0.05)
            # TMP - NEW
            # TODO
            plt.savefig(
                f"{nameFig}",
                bbox_inches="tight",
                pad=0.05,
                facecolor="white",
                transparent=False,
            )
        if show_figure:
            plt.show()
            plt.close()

    def computeGlobalShapleyValue(self):
        # TODO square
        scores_l = self.getFItemsetsDivergence()
        items = [list(i)[0] for i in self.itemset_divergence[1].keys()]
        attributes = list(set([k.split("=")[0] for k in items]))
        from collections import Counter

        card_map = dict(Counter([i.split("=")[0] for i in items]))
        global_shapley = {}
        for i in items:
            I = frozenset([i])
            global_shapley[I] = computeShapleyItemset(I, scores_l, attributes, card_map)

        self.global_shapley = global_shapley
        return self.global_shapley

    def getDfDeltaShapleyValue(self, v_i=True):
        if self.df_delta is not None:
            return self.df_delta
        self.df_delta = computeDeltaDiffShap(self.itemset_divergence, v_i=v_i)
        return self.df_delta

    # TODO significant
    def getCorrectiveItemsDf(self, verbose=True, v_i=True, squared=False):
        if self.corr_df is not None:
            return self.corr_df if verbose else self.corr_df[[i_col, s_col, corr_coef]]
        if self.df_delta is None:
            self.getDfDeltaShapleyValue(v_i=v_i)

        d = self.df_delta.copy()
        d[corr_coef] = (abs(d[v_si_col])) - (abs(d[v_s_col]))
        d[corr_coef_sq] = d[v_si_col] ** 2 - d[v_s_col] ** 2

        d = d.loc[d[corr_coef] < 0]
        # MOD
        if squared:
            d = d.sort_values(corr_coef_sq, ascending=True)
        else:
            d = d.sort_values(corr_coef, ascending=True)
        self.corr_df = d
        if verbose:
            return self.corr_df
        return (
            self.corr_df[[i_col, s_col, corr_coef, corr_coef_sq]]
            if squared
            else self.corr_df[[i_col, s_col, corr_coef]]
        )

    # TODO significant
    def getCorrectiveItemMaxCorrectiveCoef(self, verbose=True, v_i=True):
        if self.corr_df is None:
            self.getCorrectiveItemsDf(verbose=verbose, v_i=v_i)
        df1 = self.corr_df.copy()
        df1 = df1[df1.groupby([i_col])[corr_coef].transform(min) == df1[corr_coef]]
        return (
            df1.sort_values(corr_coef, ascending=True)
            if verbose
            else df1[[i_col, s_col, corr_coef]].sort_values(corr_coef, ascending=True)
        )

    # TODO significant Vebose: also v_S and v_S+i
    def getCorrectiveItemStatistics(self, MSE=False):
        if self.corr_statistics_df is not None:
            return (
                self.corr_statistics_df
                if MSE
                else self.corr_statistics_df.drop(columns=MSE_col, axis=1)
            )
        if self.corr_df is None:
            self.getCorrectiveItemsDf()
        c = self.corr_df
        statistics = c.groupby(i_col)[corr_coef].agg(["mean", "std", "count"])

        df_min = c[c.groupby([i_col])[corr_coef].transform(min) == c[corr_coef]][
            [i_col, s_col, corr_coef]
        ]
        df_min.rename(
            columns={s_col: f"{s_col}_min", corr_coef: f"{corr_coef}_min"}, inplace=True
        )
        df_max = c[c.groupby([i_col])[corr_coef].transform(max) == c[corr_coef]][
            [i_col, s_col, corr_coef]
        ]
        df_max.rename(
            columns={s_col: f"{s_col}_max", corr_coef: f"{corr_coef}_max"}, inplace=True
        )
        df_min.set_index(i_col, inplace=True)
        df_max.set_index(i_col, inplace=True)
        j = df_min.join(df_max)
        statistics = j.join(statistics)

        cnt = (
            self.getDfDeltaShapleyValue()
            .groupby(i_col)[delta_col]
            .agg(["count"])
            .rename(columns={"count": "tot"})
        )
        statistics = statistics.join(cnt)
        statistics["c%"] = statistics["count"] / statistics["tot"]
        # statistics.drop(columns="tot", inplace=True)

        df = c[[i_col, delta_col]].copy()
        df[SE_col] = df[delta_col] ** 2
        statistics[MSE_col] = df.groupby(i_col)[SE_col].mean()
        statistics[SSE_col] = df.groupby(i_col)[SE_col].sum()
        self.corr_statistics_df = statistics.sort_values(f"{corr_coef}_min")

        return (
            self.corr_statistics_df
            if MSE
            else self.corr_statistics_df.drop(columns=[SSE_col, MSE_col], axis=1)
        )

    def getDeltaItemStatisticsMSE(self, sortMSE=False):
        if self.deltas_statistics_df is not None:
            return self.deltas_statistics_df

        c = self.getDfDeltaShapleyValue().copy()

        delta_col = "delta_item"
        statistics = c.groupby(i_col)[delta_col].agg(["mean", "std", "count"])
        df_min = c.loc[c.groupby([i_col])[delta_col].idxmin()][
            [i_col, s_col, delta_col]
        ]
        df_min.rename(
            columns={s_col: f"{s_col}_min", delta_col: f"{delta_col}_min"}, inplace=True
        )
        df_min.set_index(i_col, inplace=True)
        df_min["count_min"] = (
            c.loc[c.groupby([i_col])[delta_col].transform(min) == c[delta_col]]
            .groupby([i_col])[delta_col]
            .count()
        )
        df_max = c.loc[c.groupby([i_col])[delta_col].idxmax()][
            [i_col, s_col, delta_col]
        ]
        df_max.rename(
            columns={s_col: f"{s_col}_max", delta_col: f"{delta_col}_max"}, inplace=True
        )
        df_max.set_index(i_col, inplace=True)
        df_max["count_max"] = (
            c.loc[c.groupby([i_col])[delta_col].transform(max) == c[delta_col]]
            .groupby([i_col])[delta_col]
            .count()
        )
        j = df_min.join(df_max)
        statistics = j.join(statistics)
        c[SE_col] = c[delta_col] ** 2
        statistics[MSE_col] = c.groupby(i_col)[SE_col].mean()
        statistics[SSE_col] = c.groupby(i_col)[SE_col].sum()
        delta_item_abs_max = f"{delta_col}_abs_max"
        statistics[delta_item_abs_max] = statistics.apply(
            lambda x: x[f"{delta_col}_min"]
            if abs(x[f"{delta_col}_min"]) > abs(x[f"{delta_col}_max"])
            else x[f"{delta_col}_max"],
            axis=1,
        )

        if sortMSE:
            self.deltas_statistics_df = statistics.sort_values(MSE_col, ascending=False)
        else:
            self.deltas_statistics_df = statistics.iloc[
                statistics[delta_item_abs_max].abs().argsort()[::-1]
            ]

        return self.deltas_statistics_df

    def getMaximumNegativeContribution(self):
        # TODO self.df_delta
        delta_col = "delta_item"
        i_col = "item i"
        if self.df_delta is None:
            self.df_delta = computeDeltaDiffShap(self.itemset_divergence)
        df1 = self.df_delta.copy()
        df1 = df1.loc[df1[delta_col] < 0]
        df1 = df1[df1.groupby([i_col])[delta_col].transform(min) == df1[delta_col]]
        return df1.sort_values(delta_col, ascending=True)

    def getMaximumPositiveContribution(self):
        # TODO self.df_delta
        delta_col = "delta_item"
        i_col = "item i"
        if self.df_delta is None:
            self.df_delta = computeDeltaDiffShap(self.itemset_divergence)
        df1 = self.df_delta.copy()
        df1 = df1.loc[df1[delta_col] > 0]
        df1 = df1[df1.groupby([i_col])[delta_col].transform(max) == df1[delta_col]]
        return df1.sort_values(delta_col, ascending=False)

    # def compute_std_beta_distribution(self, FPb):
    #    return ((FPb.a*FPb.b)/((FPb.a+FPb.b)**2*(FPb.a+FPb.b+1)))**(1/2)

    # def compute_variance_beta_distribution(self, FPb):
    #    return ((FPb.a*FPb.b)/((FPb.a+FPb.b)**2*(FPb.a+FPb.b+1)))

    # def compute_mean_beta_distribution(self, FPb):
    #    return FPb.a/(FPb.a+FPb.b)

    ### No
    def statistic_beta_distribution(self, statisticsOfI=["var", "mean"], FPb=None):
        statisticsOfI = (
            statisticsOfI if type(statisticsOfI) == list else [statisticsOfI]
        )
        if not set(statisticsOfI).issubset(["std", "var", "mean"]):
            raise ValueError("Accepted beta metrics: std, var, mean")
        if self.metric not in map_beta_distribution:
            raise ValueError(f"{self.metric} not in {map_beta_distribution.keys()}")
        if FPb is None:
            FPb = self.freq_metrics  # .copy()
        cols_beta = []
        cl_metric = map_beta_distribution[self.metric]
        FPb["a"] = 1 + FPb[cl_metric["T"]].sum(axis=1)
        FPb["b"] = 1 + FPb[cl_metric["F"]].sum(axis=1)
        cl_metric = "_".join(cl_metric["T"])
        for statisticOfI in statisticsOfI:
            col_beta = f"{statisticOfI}_beta_{cl_metric}"
            if statisticOfI == "std":
                FPb[col_beta] = _compute_std_beta_distribution(FPb[["a", "b"]])
            elif statisticOfI == "var":
                FPb[col_beta] = _compute_variance_beta_distribution(FPb[["a", "b"]])
            else:
                FPb[col_beta] = _compute_mean_beta_distribution(FPb[["a", "b"]])
            cols_beta.append(col_beta)
        FPb.drop(columns=["a", "b"], inplace=True)
        return FPb, cols_beta

    ###NO
    def t_test(self, verbose=False, ret=True):
        c_metric = "_".join(map_beta_distribution[self.metric]["T"])
        if f"t_value_{c_metric}" in self.freq_metrics.columns:
            if ret:
                return self.freq_metrics, f"t_value_{c_metric}"
            else:
                return
        FPb, cols_beta = self.statistic_beta_distribution(["mean", "var"])

        mean_col, var_col = f"mean_beta_{c_metric}", f"var_beta_{c_metric}"
        mean_d, var_d = FPb.loc[FPb.itemsets == frozenset()][
            [mean_col, var_col]
        ].values[0]
        FPb[f"t_value_{c_metric}"] = _compute_t_test(
            FPb[[mean_col, var_col]], mean_col, var_col, mean_d, var_d
        )
        self.t_value_col = f"t_value_{c_metric}"
        if ret == False:
            return
        if verbose:
            return FPb, cols_beta + [self.t_value_col]
        else:
            FPb.drop(columns=cols_beta, inplace=True)
            return FPb, self.t_value_col




    def correctiveTvalues(self):
        # Get-T-Values
        
        corrOfI = self.getCorrectiveItemsDf().copy()
        corrOfI["corr_factor"] = abs(corrOfI["v_S"]) - abs(corrOfI["v_S+i"])

        if self.metric == D_OUTCOME:
            return corrOfI       
        itemsetsOfI = corrOfI[["S", "S+i"]]
        itemsetsOfI = list(set(itemsetsOfI["S"].values)) + list(set(itemsetsOfI["S+i"]))
        df = self.freq_metrics.loc[self.freq_metrics.itemsets.isin(itemsetsOfI)].copy()
        FPb, cols = self.statistic_beta_distribution(FPb=df)
        dict_var = FPb[["itemsets"] + cols].set_index("itemsets").T.to_dict()
        for c in cols:
            corrOfI[f"{c}_S"] = corrOfI["S"].apply(lambda x: dict_var[x][c])
            corrOfI[f"{c}_S+i"] = corrOfI["S+i"].apply(lambda x: dict_var[x][c])
        m = "_".join(cols[0].split("_")[2:])
        corrOfI[f"t_value_corr"] = (
            abs(corrOfI[f"mean_beta_{m}_S"] - corrOfI[f"mean_beta_{m}_S+i"])
        ) / ((corrOfI[f"var_beta_{m}_S"] + corrOfI[f"var_beta_{m}_S+i"]) ** 0.5)
        #d_tt = self.getTvalues()
        #corrOfI[f"t_value_S+i"] = corrOfI["S+i"].apply(lambda x: d_tt[x])
        
        return corrOfI

    # Old getSignificant
    def getCorrectiveItems(self):
        #if self.corrSignif is not None:
        #    return self.corrSignif
        corrDf = self.correctiveTvalues()
        if self.metric == D_OUTCOME:
            import warnings
            warnings.warn("All corrective items are returned. Statistical significance to be computed.")
            return corrDf[[
            "item i",
            "S",
            "S+i",
            "v_i",
            "v_S",
            "v_S+i",
            "corr_factor",
            ]]
        colsOfI = [
            "item i",
            "S",
            "S+i",
            "v_i",
            "v_S",
            "v_S+i",
            "t_value_corr",
            # "t_value_S+i",
            "corr_factor",
        ]
        corrDf = corrDf[colsOfI]
        corrDfSignificant = corrDf.loc[corrDf["t_value_corr"] > 2]
        self.corrSignif = corrDfSignificant
        return corrDfSignificant

    # Old getSignificant
    def getCorrectiveItemsPos(self):
        corrDfSignificant = self.getCorrectiveItems()
        corrDfSignificant_pos = corrDfSignificant.loc[corrDfSignificant["v_S"] > 0]
        return corrDfSignificant_pos

    # TODO
    def getIndexesCorrectiveItemsets(self):
        corrSign = self.getCorrectiveItems()

        from copy import deepcopy

        # List of temsets for which a corrective behavior is observed
        corrItemsets = list(deepcopy(corrSign["S+i"]))

        df_itemsets = deepcopy(self.freq_metrics[["itemsets"]])

        # Sort the values according the order of the corrective item (i.e. sorted for corrective factor).
        # Use a column for keep the sorting.
        df_itemsets["corr_item_order"] = df_itemsets["itemsets"].apply(
            lambda x: corrItemsets.index(x) if x in corrItemsets else -1
        )
        # Get the indexes of the original itemsets, keeping only the one where a corrective behavior is observed, sorted by corrective factor
        indexes_corrective = list(
            df_itemsets[df_itemsets["corr_item_order"] >= 0]
            .sort_values("corr_item_order")
            .index
        )
        return indexes_corrective
