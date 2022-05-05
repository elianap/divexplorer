import pandas as pd
import numpy as np
from mlxtend.frequent_patterns.apriori import (
    generate_new_combinations_low_memory,
    generate_new_combinations,
)
from mlxtend.frequent_patterns import fpcommon as fpc

from .utils_FPgrowth import fpgrowth_cm


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


OUTCOME = "outcome"
CLASSIFICATION = "classification"
D_OUTCOME = "d_outcome"
AVG_OUTCOME = "outcome"


def oneHotEncoding(dfI):
    attributes = dfI.columns
    X_one_hot = dfI.copy()
    X_one_hot = pd.get_dummies(X_one_hot, prefix_sep="=", columns=attributes)
    X_one_hot.reset_index(drop=True, inplace=True)
    return X_one_hot


def _compute_t_test(df, col_mean, col_var, mean_d, var_d):
    return (abs(df[col_mean] - mean_d)) / ((df[col_var] + var_d) ** 0.5)


def _compute_t_test_welch(df, col_mean, col_var, col_size, mean_d, var_d, size_d):
    return (abs(df[col_mean] - mean_d)) / (
        (df[col_var] / df[col_size] + var_d / size_d) ** 0.5
    )


def _compute_variance_beta_distribution(FPb):
    return (FPb.a * FPb.b) / ((FPb.a + FPb.b) ** 2 * (FPb.a + FPb.b + 1))


def _compute_mean_beta_distribution(FPb):
    return FPb.a / (FPb.a + FPb.b)


def check_target_inputs(class_name, pred_name, target_name):
    if target_name is None and (class_name is None and pred_name is None):
        raise ValueError("Specify the target column(s)")
    if target_name is not None and (class_name is not None or pred_name is not None):
        raise ValueError(
            "Specify only a type of target: target_name if outcome target or class_name and/or pred_name for classification targets"
        )


def define_target(true_class_name, predicted_class_name, target_name):
    if (true_class_name is not None) or (predicted_class_name is not None):
        return CLASSIFICATION
    elif target_name is not None:
        return OUTCOME
    else:
        # Remove, never raised if we check before the input
        raise ValueError("None specified")

def check_single_classification_target(true_class_name, predicted_class_name):
    if (true_class_name is None) or (predicted_class_name is None):
        return True

class FP_DivergenceExplorer:
    def __init__(
        self,
        X_discrete,
        target_name=None,
        true_class_name=None,
        predicted_class_name=None,
        class_map={},
        ignore_cols=[],
        is_one_hot_encoding=False,
        dataset_name="",  # TODO remove
    ):
        # TODO now function in import dataset
        check_target_inputs(true_class_name, predicted_class_name, target_name)

        self.target_type = define_target(
            true_class_name, predicted_class_name, target_name
        )

        if self.target_type == CLASSIFICATION:

            if check_single_classification_target(true_class_name, predicted_class_name):
                if true_class_name is not None:
                    single_target_class = true_class_name
                else:
                    single_target_class = predicted_class_name
                cols = [single_target_class]  + ignore_cols
            else:
                cols = [true_class_name, predicted_class_name] + ignore_cols
                single_target_class = None
                
            if single_target_class is None:
                self.y = X_discrete[[true_class_name]].copy()
                self.y_predicted =  X_discrete[predicted_class_name].copy().values
                
            else:
                self.y = X_discrete[[single_target_class]].copy()
                self.y_predicted = X_discrete[[single_target_class]].copy().values

            self.y_true_pred = None
            self.y_true_pred = self.y.copy()
            self.y_true_pred.columns = ["true_class"]  # TODO class1?
            self.y_true_pred = self.y_true_pred.assign(predicted=self.y_predicted)

            self.class_map = class_map
            if self.class_map == {}:
                from sklearn.utils.multiclass import unique_labels

                labels = np.sort(unique_labels(self.y, self.y_predicted))
                if len(labels) > 2:
                    # todo error
                    print("Binary class")
                    raise ValueError(f"Not binary problem:{len(labels)}")
                self.class_map = {"N": labels[0], "P": labels[1]}
        else:
            # TODO - ADD MULTIPLE TARGETS --> TARGET COLUMNS
            cols = [target_name] + ignore_cols
            from copy import deepcopy

            self.target_col_name = target_name
            self.target_squared_col_name = f"{target_name}_squared"
            self.target_scores_df = deepcopy(X_discrete[[target_name]])
            self.target_scores_df[self.target_squared_col_name] = (
                X_discrete[target_name] ** 2
            )

            self.support_count_col = "support_count"

        if is_one_hot_encoding:
            self.X = X_discrete.drop(columns=cols)
        else:
            self.X = oneHotEncoding(X_discrete.drop(columns=cols))

    def instanceConfusionMatrix(self, df):
        # TODO
        df["tn"] = (
            (df.true_class == df.predicted) & (df.true_class == self.class_map["N"])
        ).astype(int)
        df["fp"] = (
            (df.true_class != df.predicted) & (df.true_class == self.class_map["N"])
        ).astype(int)
        df["tp"] = (
            (df.true_class == df.predicted) & (df.true_class == self.class_map["P"])
        ).astype(int)
        df["fn"] = (
            (df.true_class != df.predicted) & (df.true_class == self.class_map["P"])
        ).astype(int)
        return df

    def apriori_divergence(
        self,
        df,
        df_true_pred,
        min_support=0.5,
        use_colnames=False,
        max_len=None,
        verbose=0,
        low_memory=False,
        cols_orderTP=["tn", "fp", "fn", "tp"],
        sortedV="support",
    ):
        """

        Returns
        -----------
        pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
          that are >= `min_support` and < than `max_len`
          (if `max_len` is not None).
          Each itemset in the 'itemsets' column is of type `frozenset`,
          which is a Python built-in type that behaves similarly to
          sets except that it is immutable
          (For more info, see
          https://docs.python.org/3.6/library/stdtypes.html#frozenset).
        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
        """

        def filterColumns(df_filter, cols):
            return df_filter[(df_filter[df_filter.columns[list(cols)]] > 0).all(1)]

        def sum_values(_x):
            out = np.sum(_x, axis=0)
            return np.array(out).reshape(-1)

        def _support(_x, _n_rows, _is_sparse):
            """DRY private method to calculate support as the
            row-wise sum of values / number of rows
            Parameters
            -----------
            _x : matrix of bools or binary
            _n_rows : numeric, number of rows in _x
            _is_sparse : bool True if _x is sparse
            Returns
            -----------
            np.array, shape = (n_rows, )
            Examples
            -----------
            For usage examples, please see
            http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
            """
            out = np.sum(_x, axis=0) / _n_rows
            return np.array(out).reshape(-1)

        if min_support <= 0.0:
            raise ValueError(
                "`min_support` must be a positive "
                "number within the interval `(0, 1]`. "
                "Got %s." % min_support
            )

        fpc.valid_input_check(df)

        if hasattr(df, "sparse"):
            # DataFrame with SparseArray (pandas >= 0.24)
            if df.size == 0:
                X = df.values
            else:
                X = df.sparse.to_coo().tocsc()
            is_sparse = True
        else:
            # dense DataFrame
            X = df.values
            is_sparse = False
        support = _support(X, X.shape[0], is_sparse)
        ary_col_idx = np.arange(X.shape[1])
        support_dict = {0: 1, 1: support[support >= min_support]}
        itemset_dict = {0: [()], 1: ary_col_idx[support >= min_support].reshape(-1, 1)}
        conf_metrics = {
            0: np.asarray([sum_values(df_true_pred[cols_orderTP])]),
            1: np.asarray(
                [
                    sum_values(filterColumns(df_true_pred, item)[cols_orderTP])
                    for item in itemset_dict[1]
                ]
            ),
        }
        max_itemset = 1
        rows_count = float(X.shape[0])

        all_ones = np.ones((int(rows_count), 1))

        while max_itemset and max_itemset < (max_len or float("inf")):
            next_max_itemset = max_itemset + 1

            # With exceptionally large datasets, the matrix operations can use a
            # substantial amount of memory. For low memory applications or large
            # datasets, set `low_memory=True` to use a slower but more memory-
            # efficient implementation.
            if low_memory:
                combin = generate_new_combinations_low_memory(
                    itemset_dict[max_itemset], X, min_support, is_sparse
                )
                # slightly faster than creating an array from a list of tuples
                combin = np.fromiter(combin, dtype=int)
                combin = combin.reshape(-1, next_max_itemset + 1)

                if combin.size == 0:
                    break
                if verbose:
                    print(
                        "\rProcessing %d combinations | Sampling itemset size %d"
                        % (combin.size, next_max_itemset),
                        end="",
                    )

                itemset_dict[next_max_itemset] = combin[:, 1:]
                support_dict[next_max_itemset] = combin[:, 0].astype(float) / rows_count
                max_itemset = next_max_itemset
                # TODO
            else:
                combin = generate_new_combinations(itemset_dict[max_itemset])
                combin = np.fromiter(combin, dtype=int)
                combin = combin.reshape(-1, next_max_itemset)

                if combin.size == 0:
                    break
                if verbose:
                    print(
                        "\rProcessing %d combinations | Sampling itemset size %d"
                        % (combin.size, next_max_itemset),
                        end="",
                    )

                if is_sparse:
                    _bools = X[:, combin[:, 0]] == all_ones
                    for n in range(1, combin.shape[1]):
                        _bools = _bools & (X[:, combin[:, n]] == all_ones)
                else:
                    _bools = np.all(X[:, combin], axis=2)
                support = _support(np.array(_bools), rows_count, is_sparse)
                _mask = (support >= min_support).reshape(-1)
                if any(_mask):
                    itemset_dict[next_max_itemset] = np.array(combin[_mask])
                    support_dict[next_max_itemset] = np.array(support[_mask])
                    conf_metrics[next_max_itemset] = np.asarray(
                        [
                            sum_values(
                                filterColumns(df_true_pred, itemset)[cols_orderTP]
                            )
                            for itemset in itemset_dict[next_max_itemset]
                        ]
                    )
                    max_itemset = next_max_itemset
                else:
                    # Exit condition
                    break

        all_res = []
        for k in sorted(itemset_dict):
            support = pd.Series(support_dict[k])
            itemsets = pd.Series(
                [frozenset(i) for i in itemset_dict[k]], dtype="object"
            )
            # conf_matrix_col=pd.Series(list(conf_metrics[k]))
            conf_metrics_cols = pd.DataFrame(
                list(conf_metrics[k]), columns=cols_orderTP
            )

            res = pd.concat((support, itemsets, conf_metrics_cols), axis=1)
            all_res.append(res)

        res_df = pd.concat(all_res)
        res_df.columns = ["support", "itemsets"] + cols_orderTP

        if use_colnames:
            mapping = {idx: item for idx, item in enumerate(df.columns)}
            res_df["itemsets"] = res_df["itemsets"].apply(
                lambda x: frozenset([mapping[i] for i in x])
            )

        res_df["length"] = res_df["itemsets"].str.len()
        res_df["support_count"] = np.sum(res_df[cols_orderTP], axis=1)

        res_df.sort_values(sortedV, ascending=False, inplace=True)
        res_df = res_df.reset_index(drop=True)

        if verbose:
            print()  # adds newline if verbose counter was used

        return res_df

    def fpgrowth_divergence_metrics(
        self,
        df,
        df_confusion_matrix,
        min_support=0.5,
        use_colnames=False,
        verbose=0,
        cols_orderTP=["tn", "fp", "fn", "tp"],
        sortedV="support",
    ):

        fp = fpgrowth_cm(
            df,
            df_confusion_matrix,
            min_support=min_support,
            use_colnames=use_colnames,
            cols_orderTP=cols_orderTP,
        )
        row_root = dict(df_confusion_matrix.sum())
        row_root.update({"support": 1, "itemsets": frozenset()})
        fp.loc[len(fp), row_root.keys()] = row_root.values()
        #fp = fp.append(row_root, ignore_index=True)
        fp["length"] = fp["itemsets"].str.len()

        fp["support_count"] = (fp["support"] * len(df)).round()

        fp.sort_values(sortedV, ascending=False, inplace=True)
        fp = fp.reset_index(drop=True)
        return fp

    def getFrequentPatternDivergence(
        self,
        min_support,
        sortedV="support",
        metrics=["d_fpr", "d_fnr", "d_accuracy"],
        FPM_type="fpgrowth",
    ):

        if FPM_type not in ["fpgrowth", "apriori"]:
            raise ValueError(
                f'{FPM_type} algorithm is not handled. For now, we integrate the DivExplorer computation in "fpgrowth" and "apriori" algorithms.'
            )

        # TODO Anticipate it?
        if self.target_type == CLASSIFICATION:
            y_conf_matrix = self.instanceConfusionMatrix(self.y_true_pred)
            conf_matrix_cols = ["tn", "fp", "fn", "tp"]

        if FPM_type == "fpgrowth":

            if self.target_type == CLASSIFICATION:
                input_data_targets = y_conf_matrix[conf_matrix_cols]
                cols_orderTP = conf_matrix_cols
            else:
                input_data_targets = self.target_scores_df
                cols_orderTP = [self.target_col_name, self.target_squared_col_name]

            df_FP_metrics = self.fpgrowth_divergence_metrics(
                self.X.copy(),
                input_data_targets,
                min_support=min_support,
                use_colnames=True,
                sortedV=sortedV,
                cols_orderTP=cols_orderTP,
            )
        else:
            if self.target_type == CLASSIFICATION:

                attributes_one_hot = self.X.columns
                df_with_conf_matrix = pd.concat(
                    [self.X, y_conf_matrix[conf_matrix_cols]], axis=1
                )
                input_data_X = df_with_conf_matrix[attributes_one_hot]
                input_data_targets = df_with_conf_matrix
            else:
                raise ValueError(
                    "The apriori implementation is available only for classification purposes."
                )
                input_data_X = self.X.copy()
                input_data_targets = self.target_scores_df

            df_FP_metrics = self.apriori_divergence(
                input_data_X,
                input_data_targets,
                min_support=min_support,
                use_colnames=True,
                sortedV=sortedV,
            )

        df_FP_divergence = self.computeDivergenceItemsets(
            df_FP_metrics, metrics=metrics
        )

        # T_test values
        if self.target_type == CLASSIFICATION:
            df_FP_divergence = self.t_test_FP(df_FP_divergence, metrics=metrics)
        else:
            df_FP_divergence = self.statistical_significance_outcome(
                df_FP_divergence, AVG_OUTCOME, self.target_squared_col_name
            )
            # Drop the sum (avg = sum / support_count)
            df_FP_divergence.drop(
                columns=[self.target_col_name],
                inplace=True,
            )
        return df_FP_divergence

    def computeDivergenceItemsets(
        self,
        fm_df,
        metrics=["d_fpr", "d_fnr", "d_accuracy"],
        cols_orderTP=["tn", "fp", "fn", "tp"],
    ):
        from .utils_metrics_FPx import (
            fpr_df,
            fnr_df,
            accuracy_df,
            classification_error_df,
            true_positive_rate_df,
            true_negative_rate_df,
            precision_df,
            recall_df,
            f1_score_df
        )

        name_funct = {
            "fpr": fpr_df,
            "fnr": fnr_df,
            "accuracy": accuracy_df,
            "error": classification_error_df,
            "tpr": true_positive_rate_df,
            "tnr": true_negative_rate_df,
            "precision": precision_df,
            "recall": recall_df,
            "f1": f1_score_df,
        }

        # "ppv": positive_predicted_value_df
        # # "npv" : negative_predicted_value_df
        # "fdr" : false_discovery_rate_df
        # "for" : false_omission_rate_df

        # TODO - REFACTOR CODE
        if D_OUTCOME in metrics:
            cols_orderTP = [self.target_col_name, self.target_squared_col_name]
            from .utils_metrics_FPx import averageScore

            fm_df[AVG_OUTCOME] = averageScore(
                fm_df[cols_orderTP + [self.support_count_col]],
                self.target_col_name,
                self.support_count_col,
            )
            from .utils_metrics_FPx import getInfoRoot

            infoRoot = getInfoRoot(fm_df)

            fm_df[D_OUTCOME] = fm_df[AVG_OUTCOME] - infoRoot[AVG_OUTCOME].values[0]

        else:

            from .utils_metrics_FPx import getInfoRoot

            rootIndex = getInfoRoot(fm_df).index

            for d_metric in metrics:
                metric = d_metric[2:]
                if d_metric == "d_posr":
                    # TODO
                    from .utils_metrics_FPx import get_pos, posr_df

                    fm_df["P"] = get_pos(fm_df[cols_orderTP])
                    fm_df[metric] = posr_df(fm_df[cols_orderTP])

                elif d_metric == "d_negr":
                    # TODO
                    from .utils_metrics_FPx import get_neg, negr_df

                    fm_df["N"] = get_neg(fm_df[cols_orderTP])
                    fm_df[metric] = negr_df(fm_df[cols_orderTP])

                else:
                    fm_df[metric] = name_funct[metric](fm_df[cols_orderTP])

                fm_df[d_metric] = fm_df[metric] - fm_df.loc[rootIndex][metric].values[0]

        return fm_df

    def mean_var_beta_distribution(self, FP_df, metric):

        cl_metric = map_beta_distribution[metric]
        FP_df["a"] = 1 + FP_df[cl_metric["T"]].sum(axis=1)
        FP_df["b"] = 1 + FP_df[cl_metric["F"]].sum(axis=1)
        cl_metric = "_".join(cl_metric["T"])
        FP_df[f"mean_beta_{cl_metric}"] = _compute_mean_beta_distribution(
            FP_df[["a", "b"]]
        )
        FP_df[f"var_beta_{cl_metric}"] = _compute_variance_beta_distribution(
            FP_df[["a", "b"]]
        )
        FP_df.drop(columns=["a", "b"], inplace=True)
        return FP_df

    def t_test_FP(self, FP_df, metrics=["d_fpr", "d_fnr", "d_accuracy"]):
        for metric in metrics:
            if metric not in map_beta_distribution:
                raise ValueError(f"{metric} not in {map_beta_distribution.keys()}")

            c_metric = "_".join(map_beta_distribution[metric]["T"])
            FPb = self.mean_var_beta_distribution(FP_df, metric)

            mean_col, var_col = f"mean_beta_{c_metric}", f"var_beta_{c_metric}"
            mean_d, var_d = FPb.loc[FPb.itemsets == frozenset()][
                [mean_col, var_col]
            ].values[0]
            FPb[f"t_value_{c_metric}"] = _compute_t_test(
                FPb[[mean_col, var_col]], mean_col, var_col, mean_d, var_d
            )
            FPb.drop(
                columns=[f"mean_beta_{c_metric}", f"var_beta_{c_metric}"], inplace=True
            )
        return FPb

    def statistical_significance_outcome(
        self, fm_df, mean_col, squared_col, type_test="welch"
    ):

        fm_df["var"] = (
            fm_df[squared_col] / fm_df[self.support_count_col] - fm_df[mean_col] ** 2
        )

        from .utils_metrics_FPx import getInfoRoot

        _infoRoot_dict = getInfoRoot(fm_df).T.to_dict()
        if len(_infoRoot_dict) != 1:
            raise ValueError("Multiple roots")
        infoRoot_dict = _infoRoot_dict[list(_infoRoot_dict.keys())[0]]
        if type_test == "welch":
            fm_df[f"t_value_{mean_col}"] = _compute_t_test_welch(
                fm_df,
                mean_col,
                "var",
                self.support_count_col,
                infoRoot_dict[mean_col],
                infoRoot_dict["var"],
                infoRoot_dict[self.support_count_col],
            )
        else:
            fm_df[f"t_value_{mean_col}"] = _compute_t_test(
                fm_df, mean_col, "var", infoRoot_dict[mean_col], infoRoot_dict["var"]
            )
        fm_df.drop(
            columns=["var", squared_col],
            inplace=True,
        )
        return fm_df