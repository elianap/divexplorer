ITEMSETS = ["support", "itemsets", "tn", "tp", "fn", "fp"]
FAIRNESS_METRICS = [
    "d_fpr",
    "d_fpr_abs",
    "d_fnr",
    "d_fnr_abs",
    "d_accuracy",
    "SPsf",
    "FPsf",
    "FNsf",
    "ACsf",
]
DIVERGENCE_METRICS = ["d_fpr", "d_fnr", "d_accuracy"]
CLASSIFICATION_METRICS = ["fpr", "fnr", "accuracy"]
EFF_LOSS = ["effect_size", "log_loss"]


def selectItemsInteractive(fpis):
    import ipywidgets as widgets

    items = [list(i)[0] for i in fpis.itemset_divergence[1].keys()]
    attributes = list(set([k.split("=")[0] for k in items]))

    map_a_i = {k: [] for k in attributes}
    for item in items:
        map_a_i[item.split("=")[0]].append(item)

    def getSelectedItems(b):
        selected = []
        for i in w_items:
            if w_items[i].value != "":
                selected.append(w_items[i].value)
        display(
            fpis.getInfoItemset(selected)[
                ITEMSETS + CLASSIFICATION_METRICS + DIVERGENCE_METRICS
            ]
        )

    style = {"description_width": "initial"}

    w_items = {}
    w_items_list = []
    for i in map_a_i:
        w_items[i] = widgets.Dropdown(
            options=[""] + list(map_a_i[i]), value="", description=i, style=style
        )
        w_items_list.append(w_items[i])

    from ipywidgets import VBox

    h = VBox(w_items_list)
    display(h)
    btn2 = widgets.Button(description="Select items")
    a = btn2.on_click(getSelectedItems)
    display(btn2)