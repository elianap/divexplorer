# from utils_c import *
import numpy as np


def weight_delta_score(n, s):
    import math

    return (math.factorial(s) * math.factorial(n - s - 1)) / math.factorial(n)


def compute_sh_subset_item_i(item_i, subset_i, powerset_subset_i, item_score):
    subsets_item_i = [s for s in powerset_subset_i if item_i.issubset(s)]
    deltas_item_i = {}
    # tmp=[]
    for item in subsets_item_i:
        S = item - item_i
        deltas_item_i[item] = item_score[len(item)][item] - item_score[len(S)][S]

        # tmp.append([item, S, deltas_item_i[item], item_score[len(item)][item],item_score[len(S)][S] ])
    # tmp = pd.DataFrame(tmp, columns=["item", "S", "delta_item", "v_item", "v_S"])
    # display(tmp)
    return sum(
        [
            weight_delta_score(len(subset_i), len(k) - 1) * v
            for k, v in deltas_item_i.items()
        ]
    )


def shapley_subset(subset_i, item_score):
    powerset_subset_i = powerset(subset_i)
    item_sh_sub = {}
    for item_i in [frozenset([i]) for i in subset_i]:
        item_sh_sub[item_i] = compute_sh_subset_item_i(
            item_i, subset_i, powerset_subset_i, item_score
        )
    return item_sh_sub


from itertools import chain, combinations


def powerset(iterable):
    s = list(iterable)
    return [
        frozenset(i)
        for i in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    ]


"""def plotSh(shapley_dict, sortedF=True):
    import matplotlib.pyplot as plt

    sh_plt = {str(",".join(list(k))): v for k, v in shapley_dict.items()}
    if sortedF:
        sh_plt = {k: v for k, v in sorted(sh_plt.items(), key=lambda item: item[1])}
    plt.barh(range(len(sh_plt)), sh_plt.values(), align="center")
    plt.yticks(range(len(sh_plt)), list(sh_plt.keys()))
    plt.show()
"""


def computeGlobalMeanShapleyValue_AllSubsets(item_score):
    itemsets_l1 = list(item_score[1].keys())
    item_sh_sub = {}
    for item_i in itemsets_l1:
        item_i_superset = [
            v2 for i, v in item_score.items() for v2 in v if item_i.issubset(v2)
        ]
        deltas_item_i = {}
        for item in item_i_superset:
            S = item - item_i
            deltas_item_i[item] = item_score[len(item)][item] - item_score[len(S)][S]
        item_sh_sub[item_i] = sum(
            [
                weight_delta_score(len(itemsets_l1), len(k) - 1) * v
                for k, v in deltas_item_i.items()
            ]
        )
    return item_sh_sub


def getSubsetDict(dict_fp_l):
    subsets = {}
    # for i in list(dict_fp_l.keys())[:-1]:
    # tmp={k:[k2 for k2 in dict_fp_l[i+1] if k.issubset(k2)] for k in dict_fp_l[i]}
    # subsets.update({k:v for k,v in tmp.items() if v!=[]})  ##to remove empty subsets

    for l in list(dict_fp_l.keys())[:-1]:
        for item_i in dict_fp_l[l]:
            for item_ch in dict_fp_l[l + 1]:
                if item_i.issubset(item_ch):
                    if item_i not in subsets:
                        subsets[item_i] = []
                    subsets[item_i].append(item_ch)
    return subsets


"""def computeDeltaDiffShapOK(item_score, v_i=False):
    import pandas as pd

    diff_col = "delta_item"
    S_col = "S"
    S_item_col = "S+i"
    i_col = "item i"
    df_s = []
    s = getSubsetDict(item_score)
    for k in s:
        for item_i in s[k]:
            df_s.append(
                [
                    item_i - k,
                    k,
                    item_i,
                    item_score[len(k)][k],
                    item_score[len(item_i)][item_i],
                    item_score[len(item_i)][item_i] - item_score[len(k)][k],
                ]
            )
    df_s = pd.DataFrame(
        df_s,
        columns=[i_col, S_col, S_item_col, f"v_{S_col}", f"v_{S_item_col}", diff_col],
    )

    if v_i:
        df_s["v_i"] = df_s.apply(lambda x: item_score[1][x[i_col]], axis=1)
        df_s = df_s[
            [i_col, S_col, S_item_col, "v_i", f"v_{S_col}", f"v_{S_item_col}", diff_col]
        ]
    return df_s
"""


def computeDeltaDiffShap(item_score, v_i=False):
    import pandas as pd

    diff_col = "delta_item"
    S_col = "S"
    S_item_col = "S+i"
    i_col = "item i"
    df_s = []
    itemsets = [itemset for len_i, i_s in item_score.items() for itemset in i_s]
    for a in itemsets:
        if a != frozenset():
            df_s.extend(
                [
                    (
                        frozenset([i]),
                        frozenset(a) - frozenset([i]),
                        a,
                        item_score[len(frozenset(a) - frozenset([i]))][
                            frozenset(a) - frozenset([i])
                        ],
                        item_score[len(a)][a],
                        item_score[len(a)][a]
                        - item_score[len(frozenset(a) - frozenset([i]))][
                            frozenset(a) - frozenset([i])
                        ],
                    )
                    for i in list(a)
                ]
            )
    df_s = pd.DataFrame(
        df_s,
        columns=[i_col, S_col, S_item_col, f"v_{S_col}", f"v_{S_item_col}", diff_col],
    )

    if v_i:
        df_s["v_i"] = df_s.apply(lambda x: item_score[1][x[i_col]], axis=1)
        df_s = df_s[
            [i_col, S_col, S_item_col, "v_i", f"v_{S_col}", f"v_{S_item_col}", diff_col]
        ]
    return df_s


###### TODO
def normalizeShapley(shapleyValues):
    import numpy as np
    from sklearn.preprocessing import maxabs_scale  # normalize

    keys, values = list(shapleyValues.keys()), list(shapleyValues.values())
    # normalized=normalize([np.asarray(values)])[0]
    normalized = maxabs_scale(np.asarray(values))
    normalized = {keys[i]: normalized[i] for i in range(0, len(keys))}
    return normalized


def attr(item):
    return item.split("=")[0]


def attrs(itemset):
    return [item.split("=")[0] for item in itemset]


def plus(f1, f2):
    return frozenset(list(f1) + list(f2))


def weight_factor(lB, lA, lI, prod_mb):
    import math
    import numpy as np

    return (math.factorial(lB) * math.factorial(lA - lB - lI)) / (
        math.factorial(lA) * (prod_mb)
    )


def computeShapleyItemset_1(I, scores, attributes, card_map):
    u_I = 0
    # A\attr(I)
    Bs = set(attributes) - set([attr(i) for i in I])
    # For B \subseteq A\attr(I)
    for B in [list(i) for i in powerset(Bs)]:
        attrBI = B + [attr(i) for i in I]
        prod_mb = np.prod([card_map[i] for i in attrBI])
        # I_BI=[k  for k in  scores if len(k)==len(attrBI) and [i for i in k if attr(i) not in attrBI]==[]]
        I_B = [
            k
            for k in scores
            if len(k) == len(B) and [i for i in k if attr(i) not in B] == []
        ]
        for J in I_B:
            if plus(J, I) in scores:
                w = weight_factor(len(B), len(attributes), len(I), prod_mb)
                u_I = u_I + (scores[plus(J, I)] - scores[J]) * w
    return u_I


def computeShapleyItemset(I, scores_l, attributes, card_map):
    import numpy as np

    Bs = set(attributes) - set([attr(i) for i in I])
    u_I = 0
    I_Bs = [
        k2
        for k in scores_l
        for k2 in scores_l[k]
        if [i for i in k2 if attr(i) not in Bs] == []
    ]
    for J in I_Bs:
        JI = plus(J, I)
        if len(JI) in scores_l and JI in scores_l[len(JI)]:
            B = [attr(i) for i in J]
            attr_BI = B + [attr(i) for i in I]
            prod_mb = np.prod([card_map[i] for i in attr_BI])
            # I_BI=[k  for k in  scores_l[len(attr_BI)] if [i for i in k if attr(i) not in attr_BI]==[]]
            w = weight_factor(len(B), len(attributes), len(I), prod_mb)
            u_I = u_I + (scores_l[len(JI)][JI] - scores_l[len(J)][J]) * w
    return u_I


def normalizeMax(shap_values):
    shap_values_norm = shap_values.copy()
    maxv = abs(max(list(shap_values_norm.values()), key=abs))
    shap_values_norm = {k: v / maxv for k, v in shap_values_norm.items()}
    return shap_values_norm


###################################################################################################################
# SAME ITEMSET, DIFFERENT METHOD/METRIC


def compareShapleyValues(
    sh_score_1,
    sh_score_2,
    toOrder=0,
    title=[],
    sharedAxis=False,
    height=0.8,
    linewidth=0.8,
    sizeFig=(7, 7),
    saveFig=False,
    nameFig=None,
    labelsize=10,
    subcaption=False,
    pad=None,
    formatTicks=False,
):
    import matplotlib.pyplot as plt

    sh_score_1 = {str(",".join(list(k))): v for k, v in sh_score_1.items()}
    sh_score_2 = {str(",".join(list(k))): v for k, v in sh_score_2.items()}
    if toOrder == 0:
        sh_score_1 = {
            k: v for k, v in sorted(sh_score_1.items(), key=lambda item: item[1])
        }
        sh_score_2 = {k: sh_score_2[k] for k in sorted(sh_score_1, key=sh_score_1.get)}
    elif toOrder == 1:
        sh_score_2 = {
            k: v for k, v in sorted(sh_score_2.items(), key=lambda item: item[1])
        }
        sh_score_1 = {k: sh_score_1[k] for k in sorted(sh_score_2, key=sh_score_2.get)}
    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax1.barh(
        range(len(sh_score_1)),
        sh_score_1.values(),
        align="center",
        color="#7CBACB",
        height=height,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    plt.yticks(range(len(sh_score_1)), list(sh_score_1.keys()))
    if len(title) > 1:
        ax1.set_title(title[0])
    ax2 = fig.add_subplot(122)
    ax2.barh(
        range(len(sh_score_2)),
        sh_score_2.values(),
        align="center",
        color="#7CBACB",
        height=height,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    plt.yticks(range(len(sh_score_2)), [])
    if len(title) > 1:
        ax2.set_title(title[1])
    if sharedAxis:
        sh_scores = list(sh_score_1.values()) + list(sh_score_2.values())
        min_x, max_x = min(sh_scores) + min(0.01, min(sh_scores)), max(sh_scores) + min(
            0.01, max(sh_scores)
        )
        # print(min_x, max_x, min(sh_scores), max(sh_scores))
        ax1.set_xlim(min_x, max_x)
        ax2.set_xlim(min_x, max_x)
    ax1.tick_params(axis="y", labelsize=labelsize)
    if pad:
        fig.tight_layout(pad=pad)

    s1 = "(a)" if subcaption else ""  # r"$\bf{(a)}$"
    s2 = "(b)" if subcaption else ""  # r"$\bf{(b)}$"
    ax1.set_xlabel(f"{s1}", size=labelsize)
    ax2.set_xlabel(f"{s2}", size=labelsize)

    if formatTicks:
        major_formatter = FuncFormatter(my_formatter)
        ax1.xaxis.set_major_formatter(major_formatter)
        ax2.xaxis.set_major_formatter(major_formatter)

    plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
    # print(ax1.get_xlim())
    # print(ax2.get_xlim())
    if saveFig:
        nameFig = "./shap.pdf" if nameFig is None else f"{nameFig}.pdf"
        plt.savefig(nameFig, format="pdf", bbox_inches="tight")
    plt.show()


from matplotlib.ticker import FuncFormatter


def my_formatter(x, pos):
    val_str = "{:g}".format(x)
    return val_str


def plotComparisonShapleyValues(
    sh_score_1,
    sh_score_2,
    toOrder=0,
    title=[],
    sharedAxis=False,
    height=0.8,
    linewidth=0.8,
    sizeFig=(7, 7),
    saveFig=False,
    nameFig=None,
    labelsize=10,
    pad=0.5,
    subcaption=True,
    metrics_name=None,
    formatTicks=False,
    deltaLim=None,
):
    h1, h2 = (height[0], height[1]) if type(height) == list else (height, height)
    import matplotlib.pyplot as plt

    sh_score_1 = {str(",".join(list(k))): v for k, v in sh_score_1.items()}
    sh_score_2 = {str(",".join(list(k))): v for k, v in sh_score_2.items()}
    sh_score_1 = {k: v for k, v in sorted(sh_score_1.items(), key=lambda item: item[1])}
    sh_score_2 = {k: v for k, v in sorted(sh_score_2.items(), key=lambda item: item[1])}
    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax1.barh(
        range(len(sh_score_1)),
        sh_score_1.values(),
        align="center",
        color="#7CBACB",
        height=h1,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    plt.yticks(range(len(sh_score_1)), list(sh_score_1.keys()))
    if len(title) > 1:
        ax1.set_title(title[0])
    ax2 = fig.add_subplot(122)
    ax2.barh(
        range(len(sh_score_2)),
        sh_score_2.values(),
        align="center",
        color="#7CBACB",
        height=h2,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    plt.yticks(range(len(sh_score_2)), list(sh_score_2.keys()))
    if len(title) > 1:
        ax2.set_title(f"{title[1]}")
    fig.tight_layout(pad=pad)
    if len(title) > 1:
        plt.title(title[1])
    ax1.tick_params(axis="y", labelsize=labelsize)
    ax2.tick_params(axis="y", labelsize=labelsize)
    if sharedAxis:
        sh_scores = list(sh_score_1.values()) + list(sh_score_2.values())
        # min_x, max_x=min(sh_scores)+min(deltaLim, min(sh_scores)), max(sh_scores)+min(deltaLim, max(sh_scores))
        if deltaLim:
            min_x, max_x = min(sh_scores) - deltaLim, max(sh_scores) + deltaLim
        else:
            min_x, max_x = (
                min(sh_scores) + min(0.01, min(sh_scores)),
                max(sh_scores) + min(0.01, max(sh_scores)),
            )
        # print(min_x, max_x, min(sh_scores), max(sh_scores))
        ax1.set_xlim(min_x, max_x)
        ax2.set_xlim(min_x, max_x)
    s1 = "(a)" if subcaption else ""  # r"$\bf{(a)}$"
    s2 = "(b)" if subcaption else ""  # r"$\bf{(b)}$"
    # div_label=f"${div_name}({i_name}|{p_name})$"
    # x_labels=[f"${div_name}({i_name}|{p_name})$",f"${div_name}({i_name}|{p_name})$"] if metrics_name is None else [f"${div_name}_{{{metrics_name[0]}}}({i_name}|{p_name})$", f"${div_name}_{{{metrics_name[1]}}}({i_name}|{p_name})$" ]
    ax1.set_xlabel(f"{s1}", size=labelsize)
    ax2.set_xlabel(f"{s2}", size=labelsize)

    if formatTicks:
        major_formatter = FuncFormatter(my_formatter)
        ax1.xaxis.set_major_formatter(major_formatter)
        ax2.xaxis.set_major_formatter(major_formatter)

    plt.rcParams["figure.figsize"], plt.rcParams["figure.dpi"] = sizeFig, 100
    # print(ax1.get_xlim())
    # print(ax2.get_xlim())
    if saveFig:
        nameFig = "./shap.pdf" if nameFig is None else f"{nameFig}.pdf"
        plt.savefig(nameFig, format="pdf", bbox_inches="tight")
    plt.show()


#######################################################

"""
def computeInteractions(subset_i, item_score):
    powerset_subset_i=powerset(subset_i)
    interactions={}
    interactions.update({k:item_score[len(k)][k] for k in powerset_subset_i if len(k)==1})
    for s in powerset_subset_i:
        if s!=frozenset({}):
            interactions[s]=item_score[len(s)][s]
            for interaction in interactions:
                if interaction.issubset(s) and interaction!=s:
                        interactions[s]-=interactions[interaction]
    return interactions

def shapley_subset_from_interactions(subset_i, interactions):
    return {frozenset([item_i]):sum([v/len(k) for k, v in interactions.items() if frozenset([item_i]).issubset(k)]) for item_i in subset_i}


def computeGlobalMeanShapleyValue(freq_metrics, item_score):
    itemsets=freq_metrics.itemsets.values
    item_l1=[itemset for itemset in itemsets if len(itemset)==1]
    #Sum, Count
    sum_sh_dict={k:(0.0,0) for k in item_l1}

    for subset_i in itemsets:
        shapley_subset_i=shapley_subset(subset_i, item_score)
        for k, v in shapley_subset_i.items():
            sum_sh_dict[k]=(sum_sh_dict[k][0]+v, sum_sh_dict[k][1]+1)
    global_sh={k:v[0]/v[1] for k,v in sum_sh_dict.items()}

    return global_sh


    
    
def getItemsetMetricScore(item_score, itemset_i):
    return item_score[len(itemset_i)][itemset_i]


def getItemsetMetricScore(item_score, itemset_i):
    return item_score[len(itemset_i)][itemset_i]


def printInfoUnfairMetric(item_score_metric, metric, cl_type, itemset_i):
    print(f"{metric} - {cl_type}: {getItemsetMetricScore(item_score_metric[metric][cl_type], itemset_i)}")
    
    
def compareClassifierItemset_i(cl_type1, cl_type2, itemset_i, item_score_metric, metric, toOrder=0, sharedAxis=True, printInfo=True):
    if printInfo:
        printInfoUnfairMetric(item_score_metric, metric, cl_type1, itemset_i)
        printInfoUnfairMetric(item_score_metric, metric, cl_type2, itemset_i)
    sh_value_cl1=shapley_subset(itemset_i, item_score_metric[metric][cl_type1])
    sh_value_cl2=shapley_subset(itemset_i, item_score_metric[metric][cl_type2])

    compareShapleyValues(sh_value_cl1, sh_value_cl2,toOrder=toOrder, title=[cl_type1, cl_type2], sharedAxis=True)
    return sh_value_cl1, sh_value_cl2

"""
