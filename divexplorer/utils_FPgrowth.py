import numpy as np
import pandas as pd
import collections
from distutils.version import LooseVersion as Version
from pandas import __version__ as pandas_version


def filterColumns(df_filter, cols):
    return df_filter[(df_filter[df_filter.columns[list(cols)]] > 0).all(1)]


def sum_values(_x):
    out = np.sum(_x, axis=0)
    return np.array(out).reshape(-1)


def setup_fptree(df, min_support, df_targets):
    num_itemsets = len(df.index)  # number of itemsets in the database

    is_sparse = False
    if hasattr(df, "sparse"):
        # DataFrame with SparseArray (pandas >= 0.24)
        if df.size == 0:
            itemsets = df.values
        else:
            itemsets = df.sparse.to_coo().tocsr()
            is_sparse = True
    else:
        # dense DataFrame
        itemsets = df.values

    # support of each individual item
    # if itemsets is sparse, np.sum returns an np.matrix of shape (1, N)
    item_support = np.array(np.sum(itemsets, axis=0) / float(num_itemsets))
    item_support = item_support.reshape(-1)

    items = np.nonzero(item_support >= min_support)[0]

    # Define ordering on items for inserting into FPTree
    indices = item_support[items].argsort()
    rank = {item: i for i, item in enumerate(items[indices])}

    if is_sparse:
        # Ensure that there are no zeros in sparse DataFrame
        itemsets.eliminate_zeros()

    # Building tree by inserting itemsets in sorted order
    # Heuristic for reducing tree size is inserting in order
    #   of most frequent to least frequent
    tree = FPTree(rank, trgs_root=np.zeros_like(df_targets[0]))
    for i in range(num_itemsets):
        if is_sparse:
            # itemsets has been converted to CSR format to speed-up the line
            # below.  It has 3 attributes:
            #  - itemsets.data contains non null values, shape(#nnz,)
            #  - itemsets.indices contains the column number of non null
            #    elements, shape(#nnz,)
            #  - itemsets.indptr[i] contains the offset in itemset.indices of
            #    the first non null element in row i, shape(1+#nrows,)
            nonnull = itemsets.indices[itemsets.indptr[i] : itemsets.indptr[i + 1]]
        else:
            nonnull = np.where(itemsets[i, :])[0]
        itemset = [item for item in nonnull if item in rank]
        itemset.sort(key=rank.get, reverse=True)

        tree.insert_itemset(itemset, trgs_i=df_targets[i].copy())

    return tree, rank


def generate_itemsets(
    generator, num_itemsets, colname_map, cols_orderTP=["tn", "fp", "fn", "tp"]
):
    itemsets = []
    supports = []
    c_dict = {}
    for i_c_dict in range(0, len(cols_orderTP)):
        c_dict[i_c_dict] = []
    for sup, iset, cf_final in generator:
        itemsets.append(frozenset(iset))
        supports.append(sup / num_itemsets)
        for i_c_dict in range(0, len(cols_orderTP)):
            c_dict[i_c_dict].append(cf_final[i_c_dict])

    res_dic = {"support": supports, "itemsets": itemsets}

    res_dic.update({cols_orderTP[i]: c_dict[i] for i in c_dict})
    res_df = pd.DataFrame(res_dic)

    if colname_map is not None:
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([colname_map[i] for i in x])
        )

    return res_df


def valid_input_check(df):

    if f"{type(df)}" == "<class 'pandas.core.frame.SparseDataFrame'>":
        msg = (
            "SparseDataFrame support has been deprecated in pandas 1.0,"
            " and is no longer supported in mlxtend. "
            " Please"
            " see the pandas migration guide at"
            " https://pandas.pydata.org/pandas-docs/"
            "stable/user_guide/sparse.html#sparse-data-structures"
            " for supporting sparse data in DataFrames."
        )
        raise TypeError(msg)

    if df.size == 0:
        return
    if hasattr(df, "sparse"):
        if not isinstance(df.columns[0], str) and df.columns[0] != 0:
            raise ValueError(
                "Due to current limitations in Pandas, "
                "if the sparse format has integer column names,"
                "names, please make sure they either start "
                "with `0` or cast them as string column names: "
                "`df.columns = [str(i) for i in df.columns`]."
            )

    # Fast path: if all columns are boolean, there is nothing to checks
    all_bools = df.dtypes.apply(pd.api.types.is_bool_dtype).all()
    if not all_bools:
        # Pandas is much slower than numpy, so use np.where on Numpy arrays
        if hasattr(df, "sparse"):
            if df.size == 0:
                values = df.values
            else:
                values = df.sparse.to_coo().tocoo().data
        else:
            values = df.values
        idxs = np.where((values != 1) & (values != 0))
        if len(idxs[0]) > 0:
            # idxs has 1 dimension with sparse data and 2 with dense data
            val = values[tuple(loc[0] for loc in idxs)]
            s = (
                "The allowed values for a DataFrame"
                " are True, False, 0, 1. Found value %s" % (val)
            )
            raise ValueError(s)


class FPTree(object):
    def __init__(self, rank=None, trgs_root=None):
        self.root = FPNode_CM(None, target_matrix=trgs_root)
        self.nodes = collections.defaultdict(list)
        self.cond_items = []
        self.rank = rank
        self.root_trgs_root = trgs_root

    def conditional_tree(self, cond_item, minsup):
        """
        Creates and returns the subtree of self conditioned on cond_item.
        Parameters
        ----------
        cond_item : int | str
            Item that the tree (self) will be conditioned on.
        minsup : int
            Minimum support threshold.
        Returns
        -------
        cond_tree : FPtree
        """
        # Find all path from root node to nodes for item
        branches = []
        count = collections.defaultdict(int)

        for node in self.nodes[cond_item]:
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count

        # Define new ordering or deep trees may have combinatorially explosion
        items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        rank = {item: i for i, item in enumerate(items)}

        # Create conditional tree
        cond_tree = FPTree(rank, trgs_root=np.zeros_like(self.root_trgs_root))
        for idx, branch in enumerate(branches):
            branch = sorted(
                [i for i in branch if i in rank], key=rank.get, reverse=True
            )

            cond_tree.insert_itemset(
                branch,
                self.nodes[cond_item][idx].count,
                trgs_i=self.nodes[cond_item][idx].target_matrix.copy(),
            )
        cond_tree.cond_items = self.cond_items + [cond_item]

        return cond_tree

    def insert_itemset(self, itemset, count=1, trgs_i=None):
        """
        Inserts a list of items into the tree.
        Parameters
        ----------
        itemset : list
            Items that will be inserted into the tree.
        count : int
            The number of occurrences of the itemset.
        """

        self.root.count += count
        self.root.target_matrix += trgs_i

        if len(itemset) == 0:
            return

        # Follow existing path in tree as long as possible
        index = 0
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count += count
                child.target_matrix += trgs_i.copy()
                node = child
                index += 1
            else:
                break

        # Insert any remaining items
        for item in itemset[index:]:
            child_node = FPNode_CM(item, count, node, target_matrix=trgs_i.copy())
            self.nodes[item].append(child_node)
            node = child_node

    def is_path(self):
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or len(self.nodes[i][0].children) > 1:
                return False
        return True

    def print_status(self, count, colnames):
        cond_items = [str(i) for i in self.cond_items]
        if colnames:
            cond_items = [str(colnames[i]) for i in self.cond_items]
        cond_items = ", ".join(cond_items)
        print(
            "\r%d itemset(s) from tree conditioned on items (%s)" % (count, cond_items),
            end="\n",
        )


class FPNode_CM(object):
    def __init__(self, item, count=0, parent=None, target_matrix=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = collections.defaultdict(FPNode_CM)
        self.target_matrix = np.array(target_matrix)  # target_matrix.copy()

        if parent is not None:
            parent.children[item] = self

    def itempath_from_root(self):
        """Returns the top-down sequence of items from self to
        (but not including) the root node."""
        path = []
        if self.item is None:
            return path

        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path


# mlxtend Machine Learning Library Extensions
# Author: Steve Harenberg <harenbergsd@gmail.com>
#
# License: BSD 3 clause

import math
import itertools


def fpgrowth_cm(
    df,
    df_targets,
    min_support=0.5,
    use_colnames=False,
    max_len=None,
    verbose=0,
    cols_orderTP=["tn", "fp", "fn", "tp"],
):
    """Get frequent itemsets from a one-hot DataFrame
    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame the encoded format. Also supports
      DataFrames with sparse data; for more info, please
      see (https://pandas.pydata.org/pandas-docs/stable/
           user_guide/sparse.html#sparse-data-structures)
      Please note that the old pandas SparseDataFrame format
      is no longer supported in mlxtend >= 0.17.2.
      The allowed values are either 0/1 or True/False.
      For example,
    ```
           Apple  Bananas   Beer  Chicken   Milk   Rice
        0   True    False   True     True  False   True
        1   True    False   True    False  False   True
        2   True    False   True    False  False  False
        3   True     True  False    False  False  False
        4  False    False   True     True   True   True
        5  False    False   True    False   True   True
        6  False    False   True    False   True  False
        7   True     True  False    False  False  False
    ```
    min_support : float (default: 0.5)
      A float between 0 and 1 for minimum support of the itemsets returned.
      The support is computed as the fraction
      transactions_where_item(s)_occur / total_transactions.
    use_colnames : bool (default: False)
      If true, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.
    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default) all
      possible itemsets lengths are evaluated.
    verbose : int (default: 0)
      Shows the stages of conditional tree generation.
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
    ----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
    """
    valid_input_check(df)

    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )
    df_targets = df_targets.values.copy()
    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    tree, _ = setup_fptree(df, min_support, df_targets)
    minsup = math.ceil(min_support * len(df.index))  # min support as count
    generator = fpg_step(tree, minsup, colname_map, max_len, verbose)

    return generate_itemsets(
        generator, len(df.index), colname_map, cols_orderTP=cols_orderTP
    )


def fpg_step(tree, minsup, colnames, max_len, verbose):
    """
    Performs a recursive step of the fpgrowth algorithm.
    Parameters
    ----------
    tree : FPTree
    minsup : int
    Yields
    ------
    lists of strings
        Set of items that has occurred in minsup itemsets.
    """
    count = 0
    items = tree.nodes.keys()

    if tree.is_path():
        # If the tree is a path, we can combinatorally generate all
        # remaining itemsets without generating additional conditional trees
        size_remain = len(items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                supports_t = {i: tree.nodes[i][0].count for i in itemset}
                from operator import itemgetter

                id_min, support = min(supports_t.items(), key=itemgetter(1))
                cf_y = tree.nodes[id_min][0].target_matrix
                yield support, tree.cond_items + list(itemset), cf_y
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            support = sum([node.count for node in tree.nodes[item]])
            cf_y = sum([node.target_matrix for node in tree.nodes[item]])
            yield support, tree.cond_items + [item], cf_y

    if verbose:
        tree.print_status(count, colnames)

    # Generate conditional trees to generate frequent itemsets one item larger
    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)

            for sup, iset, cf_y in fpg_step(
                cond_tree, minsup, colnames, max_len, verbose
            ):
                yield sup, iset, cf_y