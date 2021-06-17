import pandas as pd
import numpy as np
import os

DATASET_DIR = os.path.join(".", "datasets")
import warnings

warnings.filterwarnings("ignore")
# COMPAS
# https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
# Quantize priors count between 0, 1-3, and >3
def quantizePrior(x):
    if x <= 0:
        return "0"
    elif 1 <= x <= 3:
        return "[1,3]"
    else:
        return ">3"


# Quantize length of stay
def quantizeLOS(x):
    if x <= 7:
        return "<week"
    if 8 < x <= 93:
        return "1w-3M"
    else:
        return ">3Months"


# Class label
# df_raw[["score_text","decile_score"]].sort_values(["score_text", "decile_score"])


def get_decile_score_class(x):
    if x >= 8:
        return "High"
    else:
        return "Medium-Low"


def get_decile_score_class2(x):
    if x >= 5:
        return "Medium-High"
    else:
        return "Low"


# https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model
def import_process_heart(discretize=False, bins=3, inputDir=DATASET_DIR):

    dt = pd.read_csv(os.path.join(inputDir, "heart.csv"))
    dt.columns = [
        "age",
        "sex",
        "chest_pain_type",
        "resting_blood_pressure",
        "cholesterol",
        "fasting_blood_sugar",
        "rest_ecg",
        "max_heart_rate_achieved",
        "exercise_induced_angina",
        "st_depression",
        "st_slope",
        "num_major_vessels",
        "thalassemia",
        "class",
    ]
    dt["sex"][dt["sex"] == 0] = "female"
    dt["sex"][dt["sex"] == 1] = "male"

    dt["chest_pain_type"][dt["chest_pain_type"] == 0] = "typical angina"
    dt["chest_pain_type"][dt["chest_pain_type"] == 1] = "atypical angina"
    dt["chest_pain_type"][dt["chest_pain_type"] == 2] = "non-anginal pain"
    dt["chest_pain_type"][dt["chest_pain_type"] == 3] = "asymptomatic"

    dt["fasting_blood_sugar"][dt["fasting_blood_sugar"] == 0] = "<120mg/ml"
    dt["fasting_blood_sugar"][dt["fasting_blood_sugar"] == 1] = ">120mg/ml"

    dt["rest_ecg"][dt["rest_ecg"] == 0] = "normal"
    dt["rest_ecg"][dt["rest_ecg"] == 1] = "ST-T wave abnormality"
    dt["rest_ecg"][dt["rest_ecg"] == 2] = "left ventricular hypertrophy"

    dt["exercise_induced_angina"][dt["exercise_induced_angina"] == 0] = "no"
    dt["exercise_induced_angina"][dt["exercise_induced_angina"] == 1] = "yes"

    dt["st_slope"][dt["st_slope"] == 0] = "upsloping"
    dt["st_slope"][dt["st_slope"] == 1] = "flat"
    dt["st_slope"][dt["st_slope"] == 2] = "downsloping"

    dt["thalassemia"][dt["thalassemia"] == 1] = "normal"
    dt["thalassemia"][dt["thalassemia"] == 2] = "fixed defect"
    dt["thalassemia"][dt["thalassemia"] == 3] = "reversable defect"
    dt = dt[dt.thalassemia != 0]

    dt = dt[dt.num_major_vessels != 4]
    dt["sex"] = dt["sex"].astype("object")
    dt["chest_pain_type"] = dt["chest_pain_type"].astype("object")
    dt["fasting_blood_sugar"] = dt["fasting_blood_sugar"].astype("object")
    dt["rest_ecg"] = dt["rest_ecg"].astype("object")
    dt["exercise_induced_angina"] = dt["exercise_induced_angina"].astype("object")
    dt["st_slope"] = dt["st_slope"].astype("object")
    dt["thalassemia"] = dt["thalassemia"].astype("object")
    dt["class"] = dt["class"].astype("int")
    dt["num_major_vessels"] = dt["num_major_vessels"].astype("object")

    if discretize:
        dt = KBinsDiscretizer_continuos(dt, bins=bins)
    return dt, {"N": 1, "P": 0}


def import_process_adult(discretize=False, bins=3, inputDir=DATASET_DIR):
    education_map = {
        "10th": "Dropout",
        "11th": "Dropout",
        "12th": "Dropout",
        "1st-4th": "Dropout",
        "5th-6th": "Dropout",
        "7th-8th": "Dropout",
        "9th": "Dropout",
        "Preschool": "Dropout",
        "HS-grad": "High School grad",
        "Some-college": "High School grad",
        "Masters": "Masters",
        "Prof-school": "Prof-School",
        "Assoc-acdm": "Associates",
        "Assoc-voc": "Associates",
    }
    occupation_map = {
        "Adm-clerical": "Admin",
        "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar",
        "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar",
        "Handlers-cleaners": "Blue-Collar",
        "Machine-op-inspct": "Blue-Collar",
        "Other-service": "Service",
        "Priv-house-serv": "Service",
        "Prof-specialty": "Professional",
        "Protective-serv": "Other",
        "Sales": "Sales",
        "Tech-support": "Other",
        "Transport-moving": "Blue-Collar",
    }
    married_map = {
        "Never-married": "Never-Married",
        "Married-AF-spouse": "Married",
        "Married-civ-spouse": "Married",
        "Married-spouse-absent": "Separated",
        "Separated": "Separated",
        "Divorced": "Separated",
        "Widowed": "Widowed",
    }

    country_map = {
        "Cambodia": "SE-Asia",
        "Canada": "British-Commonwealth",
        "China": "China",
        "Columbia": "South-America",
        "Cuba": "Other",
        "Dominican-Republic": "Latin-America",
        "Ecuador": "South-America",
        "El-Salvador": "South-America",
        "England": "British-Commonwealth",
        "France": "Euro_1",
        "Germany": "Euro_1",
        "Greece": "Euro_2",
        "Guatemala": "Latin-America",
        "Haiti": "Latin-America",
        "Holand-Netherlands": "Euro_1",
        "Honduras": "Latin-America",
        "Hong": "China",
        "Hungary": "Euro_2",
        "India": "British-Commonwealth",
        "Iran": "Other",
        "Ireland": "British-Commonwealth",
        "Italy": "Euro_1",
        "Jamaica": "Latin-America",
        "Japan": "Other",
        "Laos": "SE-Asia",
        "Mexico": "Latin-America",
        "Nicaragua": "Latin-America",
        "Outlying-US(Guam-USVI-etc)": "Latin-America",
        "Peru": "South-America",
        "Philippines": "SE-Asia",
        "Poland": "Euro_2",
        "Portugal": "Euro_2",
        "Puerto-Rico": "Latin-America",
        "Scotland": "British-Commonwealth",
        "South": "Euro_2",
        "Taiwan": "China",
        "Thailand": "SE-Asia",
        "Trinadad&Tobago": "Latin-America",
        "United-States": "United-States",
        "Vietnam": "SE-Asia",
    }
    # as given by adult.names
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income-per-year",
    ]
    train = pd.read_csv(
        os.path.join(inputDir, "adult.data"),
        header=None,
        names=column_names,
        skipinitialspace=True,
        na_values="?",
    )
    test = pd.read_csv(
        os.path.join(inputDir, "adult.test"),
        header=0,
        names=column_names,
        skipinitialspace=True,
        na_values="?",
    )
    dt = pd.concat([test, train], ignore_index=True)
    dt["education"] = dt["education"].replace(education_map)
    dt.drop(columns=["education-num", "fnlwgt"], inplace=True)
    dt["occupation"] = dt["occupation"].replace(occupation_map)
    dt["marital-status"] = dt["marital-status"].replace(married_map)
    dt["native-country"] = dt["native-country"].replace(country_map)

    dt.rename(columns={"income-per-year": "class"}, inplace=True)
    dt["class"] = (
        dt["class"].astype("str").replace({">50K.": ">50K", "<=50K.": "<=50K"})
    )
    dt.dropna(inplace=True)
    dt.reset_index(drop=True, inplace=True)
    if discretize:
        dt = KBinsDiscretizer_continuos(dt, bins=bins)
    dt.drop(columns=["native-country"], inplace=True)
    return dt, {"N": "<=50K", "P": ">50K"}


def import_process_compas2(discretize=False, risk_class=False, inputDir=DATASET_DIR):
    import pandas as pd

    df_raw = pd.read_csv(os.path.join(inputDir, "compas-scores-two-years.csv"))
    cols_propb = [
        "c_charge_degree",
        "race",
        "age_cat",
        "sex",
        "priors_count",
        "days_b_screening_arrest",
        "two_year_recid",
    ]  # , "is_recid"]
    cols_propb.sort()
    # df_raw[["days_b_screening_arrest"]].describe()
    df = df_raw[cols_propb]
    # Warning
    df["length_of_stay"] = (
        pd.to_datetime(df_raw["c_jail_out"]).dt.date
        - pd.to_datetime(df_raw["c_jail_in"]).dt.date
    ).dt.days

    df = df.loc[
        abs(df["days_b_screening_arrest"]) <= 30
    ]  # .sort_values("days_b_screening_arrest")
    # df=df.loc[df["is_recid"]!=-1]
    df = df.loc[df["c_charge_degree"] != "O"]  # F: felony, M: misconduct
    discrete = [
        "age_cat",
        "c_charge_degree",
        "race",
        "sex",
        "two_year_recid",
    ]  # , "is_recid"]
    continuous = ["days_b_screening_arrest", "priors_count", "length_of_stay"]

    toDiscretized = ["priors_count", "length_of_stay"]
    if discretize:
        df["priors_count_d"] = df["priors_count"].apply(lambda x: quantizePrior(x))
        df["length_of_stay_d"] = df["length_of_stay"].apply(lambda x: quantizeLOS(x))
        discretized = ["priors_count_d", "length_of_stay_d"]
        df = df[discrete + discretized]
    else:
        df = df[discrete + toDiscretized]
    df.rename(columns={"two_year_recid": "class"}, inplace=True)
    if risk_class:
        df["predicted"] = df_raw["decile_score"].apply(get_decile_score_class2)
    return df, {"N": 0, "P": 1}


def import_process_compas(discretize=False, risk_class=False, inputDir=DATASET_DIR):
    import warnings

    warnings.filterwarnings("ignore")
    import pandas as pd

    df_raw = pd.read_csv(os.path.join(inputDir, "compas-scores-two-years.csv"))
    cols_propb = [
        "c_charge_degree",
        "race",
        "age_cat",
        "sex",
        "priors_count",
        "days_b_screening_arrest",
        "two_year_recid",
    ]  # , "is_recid"]
    cols_propb.sort()
    # df_raw[["days_b_screening_arrest"]].describe()
    # Warning
    print("test")
    df = df_raw[cols_propb].copy()
    df["length_of_stay"] = (
        pd.to_datetime(df_raw["c_jail_out"]).dt.date
        - pd.to_datetime(df_raw["c_jail_in"]).dt.date
    ).dt.days

    df = df.loc[
        abs(df["days_b_screening_arrest"]) <= 30
    ]  # .sort_values("days_b_screening_arrest")
    # df=df.loc[df["is_recid"]!=-1]
    df = df.loc[df["c_charge_degree"] != "O"]  # F: felony, M: misconduct
    discrete = [
        "age_cat",
        "c_charge_degree",
        "race",
        "sex",
        "two_year_recid",
    ]  # , "is_recid"]
    continuous = ["days_b_screening_arrest", "priors_count", "length_of_stay"]

    toDiscretized = ["priors_count", "length_of_stay"]
    if discretize:
        df["priors_count_d"] = df["priors_count"].apply(lambda x: quantizePrior(x))
        df["length_of_stay_d"] = df["length_of_stay"].apply(lambda x: quantizeLOS(x))
        discretized = ["priors_count_d", "length_of_stay_d"]
        df = df[discrete + discretized]
    else:
        df = df[discrete + toDiscretized]
    df.rename(columns={"two_year_recid": "class"}, inplace=True)
    if risk_class:
        df["predicted"] = df_raw["decile_score"].apply(get_decile_score_class)
    return df, {"N": 0, "P": 1}


# REMOVE
def quantizeEdges(x, edges):
    if len(edges) == 1:
        if x <= edges[0]:
            return f"<={edges[0]}"
        else:
            return f">{edges[0]}"
    elif len(edges) == 2:
        if x <= edges[0]:
            return f"<={edges[0]}"
        elif edges[0] < x < edges[1]:
            return f"({edges[0]}-{edges[1]})"
        else:
            return f">={edges[1]}"
    elif len(edges) == 3:
        if x <= edges[0]:
            return f"<={edges[0]}"
        elif edges[0] < x < edges[1]:
            return f"({edges[0]}-{edges[1]})"
        elif edges[1] <= x < edges[2]:
            return f"[{edges[1]}-{edges[2]})"
        else:
            return f">={edges[2]}"

    else:
        return "------"  # TODO


def import_process_bank(bins=3, discretize=False, inputDir=DATASET_DIR):
    dt = pd.read_csv(os.path.join(inputDir, "datasets_4471_6849_bank.csv"), sep=",")
    dt.drop(columns=["duration"], inplace=True)
    dt.rename(columns={"deposit": "class"}, inplace=True)
    if discretize:
        dt = KBinsDiscretizer_continuos(dt, bins=bins)
    return dt, {"N": "no", "P": "yes"}


def import_process_german(inputDir=DATASET_DIR):
    df = pd.read_csv(os.path.join(inputDir, "credit-g.csv"))
    # print(df['personal_status'].value_counts())
    gender_map = {
        "'male single'": "male",
        "'female div/dep/mar'": "female",
        "'male mar/wid'": "male",
        "'male div/sep'": "male",
    }
    status_map = {
        "'male single'": "single",
        "'female div/dep/mar'": "married/wid/sep",
        "'male mar/wid'": "married/wid/sep",
        "'male div/sep'": "married/wid/sep",
    }
    df["sex"] = df["personal_status"].replace(gender_map)
    df["civil_status"] = df["personal_status"].replace(status_map)
    df.drop(columns=["personal_status"], inplace=True)
    df.rename(columns={"credit": "class"}, inplace=True)
    return df, {"P": "good", "N": "bad"}


def KBinsDiscretizer_continuos(dt, attributes=None, bins=3):
    attributes = dt.columns if attributes is None else attributes
    continuous_attributes = [a for a in attributes if dt.dtypes[a] != np.object]
    X_discretize = dt[attributes].copy()

    for col in continuous_attributes:
        if len(dt[col].value_counts()) > 10:
            from sklearn.preprocessing import KBinsDiscretizer

            est = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
            est.fit(dt[[col]])
            edges = [i.round() for i in est.bin_edges_][0]
            edges = [int(i) for i in edges][1:-1]
            if len(set(edges)) != len(edges):
                edges = [
                    edges[i]
                    for i in range(0, len(edges))
                    if len(edges) - 1 == i or edges[i] != edges[i + 1]
                ]
            for i in range(0, len(edges)):
                if i == 0:
                    data_idx = dt.loc[dt[col] <= edges[i]].index
                    X_discretize.loc[data_idx, col] = f"<={edges[i]}"
                if i == len(edges) - 1:
                    data_idx = dt.loc[dt[col] > edges[i]].index
                    X_discretize.loc[data_idx, col] = f">{edges[i]}"

                data_idx = dt.loc[
                    (dt[col] > edges[i - 1]) & (dt[col] <= edges[i])
                ].index
                X_discretize.loc[data_idx, col] = f"({edges[i-1]}-{edges[i]}]"
        else:
            X_discretize[col] = X_discretize[col].astype("object")
    return X_discretize


def getClassifier(type_cl, args={}):
    if type_cl == "RF":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(random_state=42, **args)
    elif type_cl == "NN":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(random_state=42)
    elif type_cl == "tree":
        from sklearn import tree

        clf = tree.DecisionTreeClassifier(random_state=42, **args)
    elif type_cl == "NB":
        from sklearn.naive_bayes import MultinomialNB

        clf = MultinomialNB()
    else:
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(random_state=42)
    return clf


def train_predict(
    dfI,
    type_cl="RF",
    labelEncoding=True,
    validation="train",
    k_cv=10,
    args={},
    retClf=False,
    fold="stratified",
):
    # Attributes with one hot encoding (attribute=value)
    attributes = dfI.columns.drop("class")
    X = dfI[attributes].copy()
    y = dfI[["class"]].copy()
    encoders = {}
    if labelEncoding:
        from sklearn.preprocessing import LabelEncoder

        encoders = {}
        for column in attributes:
            if dfI.dtypes[column] == np.object:
                le = LabelEncoder()
                X[column] = le.fit_transform(dfI[column])
                encoders[column] = le

    if type_cl == "RF":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(random_state=42, **args)
    elif type_cl == "NN":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(random_state=42, **args)
    elif type_cl == "tree":
        from sklearn import tree

        clf = tree.DecisionTreeClassifier(random_state=42, **args)
    elif type_cl == "NB":
        from sklearn.naive_bayes import MultinomialNB

        clf = MultinomialNB()
    else:
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(random_state=42)

    if validation != "cv":
        # Data for training and testing
        if validation == "all":
            X_train, y_train = X, y
        else:
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )  # stratify=y
        clf.fit(X_train, y_train.values.ravel())
        X_FP, y_FP = (X_test, y_test) if validation == "test" else (X_train, y_train)

        y_predicted = clf.predict(X_FP)
    else:
        from sklearn.model_selection import cross_val_predict

        if fold == "stratified":
            from sklearn.model_selection import StratifiedKFold

            cv = StratifiedKFold(
                n_splits=k_cv, random_state=42
            )  # Aggiunto per fissare il random state
        else:
            from sklearn.model_selection import KFold

            cv = KFold(
                n_splits=k_cv, random_state=42
            )  # Aggiunto per fissare il random state
        y_predicted = cross_val_predict(clf, X, y, cv=cv)
        X_FP, y_FP = X, y

    y_predict_prob = None
    if validation != "cv":
        y_predict_prob = clf.predict_proba(X_FP)
    else:
        y_predict_prob = cross_val_predict(
            clf, X, y["class"], cv=cv, method="predict_proba"
        )

    indexes_FP = X_FP.index
    X_FP.reset_index(drop=True, inplace=True)
    y_FP.reset_index(drop=True, inplace=True)

    if retClf and validation != "cv":
        return X_FP, y_FP, y_predicted, y_predict_prob, encoders, indexes_FP, clf
    return X_FP, y_FP, y_predicted, y_predict_prob, encoders, indexes_FP


def cap_gains_fn(x):
    x = x.astype(float)
    d = np.digitize(
        x, [0, np.median(x[x > 0]), float("inf")], right=True
    )  # .astype('|S128')
    return d.copy()


def discretize(dfI, bins=4, dataset_name=None, attributes=None, indexes_FP=None):
    indexes_validation = dfI.index if indexes_FP is None else indexes_FP
    attributes = dfI.columns if attributes is None else attributes
    if dataset_name == "compas":
        X_discretized = dfI[attributes].copy()
        X_discretized["priors_count"] = X_discretized["priors_count"].apply(
            lambda x: quantizePrior(x)
        )
        X_discretized["length_of_stay"] = X_discretized["length_of_stay"].apply(
            lambda x: quantizeLOS(x)
        )
    elif dataset_name == "adult":
        X_discretized = dfI[attributes].copy()
        X_discretized["capital-gain"] = cap_gains_fn(
            X_discretized["capital-gain"].values
        )
        X_discretized["capital-gain"] = X_discretized["capital-gain"].replace(
            {0: "0", 1: "Low", 2: "High"}
        )
        X_discretized["capital-loss"] = cap_gains_fn(
            X_discretized["capital-loss"].values
        )
        X_discretized["capital-loss"] = X_discretized["capital-loss"].replace(
            {0: "0", 1: "Low", 2: "High"}
        )
        X_discretized = KBinsDiscretizer_continuos(X_discretized, attributes, bins=bins)
    else:
        X_discretized = KBinsDiscretizer_continuos(dfI, attributes, bins=bins)
    return X_discretized.loc[indexes_validation].reset_index(drop=True)
