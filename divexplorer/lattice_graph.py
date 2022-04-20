from itertools import chain, combinations


def powerset(iterable):
    s = list(iterable)
    return [
        frozenset(i)
        for i in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    ]


def getItemsetMetricScore(item_score, itemset_i):
    return item_score[len(itemset_i)][itemset_i]


def getLatticeItemsetMetric(itemset, item_score, rounded=4, getLower=False):
    powerset_itemOfI = powerset(itemset)
    info_new = (
        {"lattice_graph": {}, "itemset_metric": {}, "lower": []}
        if getLower
        else {"lattice_graph": {}, "itemset_metric": {}}
    )
    for i in powerset_itemOfI:
        info_new["itemset_metric"][i] = round(
            getItemsetMetricScore(item_score, i), rounded
        )
        if i not in info_new["lattice_graph"]:
            info_new["lattice_graph"][i] = []
        for k in info_new["lattice_graph"]:
            if k != i and k.issubset(i) and (len(i) - 1 == len(k)):
                info_new["lattice_graph"][k].append(i)
                if getLower:
                    if getItemsetMetricScore(item_score, i) < getItemsetMetricScore(
                        item_score, k
                    ):
                        info_new["lower"].append(i)
    return info_new


def orderedNameMapping(vertices, todo):
    return [", ".join(sorted(list(v))) for v in vertices]

def plotLatticeGraph_colorGroups(
    inputTuples,
    name_mapping,
    different_colors_group,
    metric="",
    color_map={},
    annotation_F=True,
    sizeDot="",
    useMarker=True,
    show=False,
    font_size_div=10,
    font_size_hover_labels=10,
    showTitle=False,
    round_v=3,
    width=None,
    height=None,
    showGrid=True,
    plot_bgcolor="rgb(248,248,248)",
    displayItemsetLabels=False,
    font_size_ItemsetLabels=10,
):

    from igraph import Graph, EdgeSeq

    G = Graph.TupleList([(k, v) for k, vs in inputTuples.items() for v in vs])

    lay = G.layout("rt", root=[0])

    nr_vertices = G.vcount()
    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)

    E = [e.tuple for e in G.es()]  # list of edges

    labels = G.vs()["name"]

    groups = {}
    groups_labels = {}
    X_group = {}
    Y_group = {}
    if useMarker:
        markers_type = {
            "normal": "circle-dot",
            "lower": "diamond",
            "greater": "square",
            "all_greater": "hexagon",
        }
    else:
        markers_type = {k: "circle-dot" for k in different_colors_group}
    colors = ["#6175c1", "#ff6666", "#008000", "#FFC0CB"]  # todo
    setColorMap = False if color_map != {} else True
    counter_c = 0
    for group_i in different_colors_group:
        different_color = different_colors_group[group_i]
        groups[group_i] = [
            i for i in range(0, len(labels)) if labels[i] in different_color
        ]
        groups_labels[group_i] = [
            labels[i] for i in range(0, len(labels)) if labels[i] in different_color
        ]
        X_group[group_i] = [position[k][0] for k in groups[group_i]]
        Y_group[group_i] = [2 * M - position[k][1] for k in groups[group_i]]
        if setColorMap:
            color_map[group_i] = colors[counter_c]
            counter_c = counter_c + 1

    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

    sizeDot = 10 if sizeDot == "small" else 18

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=Xe,
            y=Ye,
            mode="lines",
            line=dict(color="rgb(210,210,210)", width=1),
            hoverinfo="none",
        )
    )
    for group_i in different_colors_group:
        fig.add_trace(
            go.Scatter(
                x=X_group[group_i],
                y=Y_group[group_i],
                mode="markers",
                name=metric,
                marker=dict(
                    symbol=markers_type[group_i],
                    size=sizeDot,
                    color=color_map[group_i],  #'#DB4551',
                    line=dict(color="rgb(50,50,50)", width=1),
                ),
                text=orderedNameMapping(groups_labels[group_i], name_mapping)
                if annotation_F
                else groups_labels[group_i],
                hoverinfo="text",
                opacity=0.8,
                hoverlabel=dict(font_size=font_size_hover_labels),
            )
        )

    if annotation_F:
        labels_text = [str(round(name_mapping[l], round_v)) for l in labels]

        axis = dict(
            showline=False,  # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=showGrid,
            showticklabels=False,
        )

        def make_annotations(pos, labels_text, font_size=10, font_color="rgb(0,0,0)"):
            L = len(pos)
            if len(labels_text) != L:
                raise ValueError("The lists pos and text must have the same len")
            annotations = []
            for k in range(L):
                annotations.append(
                    dict(
                        text=labels_text[
                            k
                        ],  # or replace labels with a different list for the text within the circle
                        x=pos[k][0],
                        # y=2 * M - position[k][1] + 0.05 * (2 * M - position[k][1]),
                        y=2 * M - position[k][1] + 0.03 * (2 * M),
                        xref="x1",
                        yref="y1",
                        font=dict(color=font_color, size=font_size),
                        showarrow=False,
                    )
                )
            return annotations

        fig.update_layout(
            title=metric if showTitle else None,  # TODO - TMP
            annotations=make_annotations(
                position, labels_text, font_size=font_size_div
            ),
            font_size=10,
            showlegend=False,
            xaxis=axis,
            yaxis=axis,
            margin=dict(l=0, r=0, b=0, t=20) if showTitle else dict(l=0, r=0, b=0, t=0),
            hovermode="closest",
            plot_bgcolor=plot_bgcolor,
            width=width,
            height=height,
        )

    if displayItemsetLabels:
        max_len = max([len(i) for i in name_mapping.keys()])
        X_range = [abs(lay[k][0]) for k in range(nr_vertices)]
        X_range = max(X_range) - (min(X_range))
        order_mapping = {
            v: id_v for id_v, v in enumerate(name_mapping) if len(v) in [1, max_len]
        }
        for group_i, a in groups_labels.items():
            for i, itemset in enumerate(a):
                if len(itemset) not in [1, max_len]:
                    continue
                p = (X_group[group_i][i], Y_group[group_i][i])
                get_x_pos = lambda pos_x, pad: pos_x - pad * X_range
                get_y_pos = lambda pos_y, pad: pos_y + pad * pos_y

                p_ref_x = 0.2 if order_mapping[itemset] % 2 == 0 else 0.25
                p_ref_y = -0.045
                get_name = lambda v: ", ".join(sorted(list(v)))

                fig.add_annotation(
                    x=p[0],
                    y=p[1],
                    xref="x",
                    yref="y",
                    text=get_name(itemset),
                    align="left",
                    axref="x",
                    ayref="y",
                    ax=get_x_pos(p[0], p_ref_x + 0.01 * (font_size_ItemsetLabels - 10))
                    if len(itemset) == 1
                    else get_x_pos(p[0], -0.7),
                    ay=get_y_pos(p[1], p_ref_y if len(itemset) == 1 else -0.03),
                    showarrow=True,
                    font=dict(
                        # family="Courier New, monospace",
                        size=font_size_ItemsetLabels,
                        color="black",
                    ),
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor=color_map[group_i],
                    opacity=0.8,
                    # yanchor="middle"
                )

    if show:
        fig.show()
    # TMP
    return fig
