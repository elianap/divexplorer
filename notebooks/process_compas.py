from .import_datasets import import_process_compas, discretize

dataset_name = "compas"
dfI, class_map = import_process_compas(risk_class=True)
dfI.reset_index(drop=True, inplace=True)
dfI["predicted"] = dfI["predicted"].replace({"Medium-Low": 0, "High": 1})
attributes = dfI.columns.drop(["class", "predicted"])
X_discretized = discretize(dfI, attributes=attributes, dataset_name=dataset_name)
X_discretized["class"] = dfI["class"]
X_discretized["predicted"] = dfI["predicted"]
