import numpy as np
import os


def weight_delta_score(n, s):
    import math

    return (math.factorial(s) * math.factorial(n - s - 1)) / math.factorial(n)


def compute_sh_subset_item_i(item_i, subset_i, powerset_subset_i, item_score):
    subsets_item_i = [s for s in powerset_subset_i if item_i.issubset(s)]
    deltas_item_i = {}
    for item in subsets_item_i:
        S = item - item_i
        deltas_item_i[item] = item_score[len(item)][item] - item_score[len(S)][S]

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
    show_figure=True,
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=sizeFig, dpi=100)

    ax1.barh(
        range(len(sh_score_1)),
        sh_score_1.values(),
        align="center",
        color="#7CBACB",
        height=height,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    ax1.set_yticks(range(len(sh_score_1)), minor=False)
    ax1.set_yticklabels(list(sh_score_1.keys()), fontdict=None, minor=False)

    if len(title) > 1:
        ax1.set_title(title[0])

    ax2.barh(
        range(len(sh_score_2)),
        sh_score_2.values(),
        align="center",
        color="#7CBACB",
        height=height,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )

    ax2.set_yticks(range(len(sh_score_1)), minor=False)
    ax2.set_yticklabels([], minor=False)

    if len(title) > 1:
        ax2.set_title(title[1])
    if sharedAxis:
        sh_scores = list(sh_score_1.values()) + list(sh_score_2.values())
        min_x, max_x = min(sh_scores) + min(0.01, min(sh_scores)), max(sh_scores) + min(
            0.01, max(sh_scores)
        )
        ax1.set_xlim(min_x, max_x)
        ax2.set_xlim(min_x, max_x)
    ax1.tick_params(axis="y", labelsize=labelsize)

    if pad:
        fig.tight_layout(pad=pad)
    else:
        fig.tight_layout()

    s1 = "(a)" if subcaption else ""  # r"$\bf{(a)}$"
    s2 = "(b)" if subcaption else ""  # r"$\bf{(b)}$"
    ax1.set_xlabel(f"{s1}", size=labelsize)
    ax2.set_xlabel(f"{s2}", size=labelsize)

    if saveFig:
        nameFig = (
            os.path.join(os.getcwd(), "shap.pdf")
            if nameFig is None
            else f"{nameFig}.pdf"
        )
        plt.savefig(nameFig, format="pdf", bbox_inches="tight")
    
    if show_figure:
        plt.show()
        plt.close()



def plotComparisonShapleyValues(
    sh_score_1,
    sh_score_2,
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
    deltaLim=None,
    show_figure=True,
    metrics_name=None,  # TODO remove - rmvold
):
    h1, h2 = (height[0], height[1]) if type(height) == list else (height, height)
    import matplotlib.pyplot as plt

    sh_score_1 = {str(",".join(list(k))): v for k, v in sh_score_1.items()}
    sh_score_2 = {str(",".join(list(k))): v for k, v in sh_score_2.items()}
    sh_score_1 = {k: v for k, v in sorted(sh_score_1.items(), key=lambda item: item[1])}
    sh_score_2 = {k: v for k, v in sorted(sh_score_2.items(), key=lambda item: item[1])}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=sizeFig, dpi=100)

    ax1.barh(
        range(len(sh_score_1)),
        sh_score_1.values(),
        align="center",
        color="#7CBACB",
        height=h1,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    ax1.set_yticks(range(len(sh_score_1)))
    ax1.set_yticklabels(list(sh_score_1.keys()))

    if len(title) > 1:
        ax1.set_title(title[0])

    ax2.barh(
        range(len(sh_score_2)),
        sh_score_2.values(),
        align="center",
        color="#7CBACB",
        height=h2,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    ax2.set_yticks(range(len(sh_score_2)))
    ax2.set_yticklabels(list(sh_score_2.keys()))

    if len(title) > 1:
        ax2.set_title(f"{title[1]}")
    if pad:
        fig.tight_layout(pad=pad)
    else:
        fig.tight_layout()
    if len(title) > 1:
        plt.title(title[1])
    ax1.tick_params(axis="y", labelsize=labelsize)
    ax2.tick_params(axis="y", labelsize=labelsize)
    if sharedAxis:
        sh_scores = list(sh_score_1.values()) + list(sh_score_2.values())

        if deltaLim:
            min_x, max_x = min(sh_scores) - deltaLim, max(sh_scores) + deltaLim
        else:
            min_x, max_x = (
                min(sh_scores) + min(0.01, min(sh_scores)),
                max(sh_scores) + min(0.01, max(sh_scores)),
            )

        ax1.set_xlim(min_x, max_x)
        ax2.set_xlim(min_x, max_x)
    s1 = "(a)" if subcaption else ""  # r"$\bf{(a)}$"
    s2 = "(b)" if subcaption else ""  # r"$\bf{(b)}$"

    ax1.set_xlabel(f"{s1}", size=labelsize)
    ax2.set_xlabel(f"{s2}", size=labelsize)

    if saveFig:
        nameFig = (
            os.path.join(os.getcwd(), "shap.pdf")
            if nameFig is None
            else f"{nameFig}.pdf"
        )
        plt.savefig(nameFig, format="pdf", bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close()


#######################################################


"""
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

"""
"""
###### TODO
def normalizeShapley(shapleyValues):
    import numpy as np
    from sklearn.preprocessing import maxabs_scale  # normalize

    keys, values = list(shapleyValues.keys()), list(shapleyValues.values())
    # normalized=normalize([np.asarray(values)])[0]
    normalized = maxabs_scale(np.asarray(values))
    normalized = {keys[i]: normalized[i] for i in range(0, len(keys))}
    return normalized
"""
"""
def getSubsetDict(dict_fp_l):
    subsets = {}

    for l in list(dict_fp_l.keys())[:-1]:
        for item_i in dict_fp_l[l]:
            for item_ch in dict_fp_l[l + 1]:
                if item_i.issubset(item_ch):
                    if item_i not in subsets:
                        subsets[item_i] = []
                    subsets[item_i].append(item_ch)
    return subsets
"""