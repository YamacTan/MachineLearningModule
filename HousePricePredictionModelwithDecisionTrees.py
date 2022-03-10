#######################
# Yamac TAN - Data Science Bootcamp - Week 7 - Project 1
#######################

# %%

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, DecisionTreeRegressor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve, cross_val_score
from skompiler import skompile
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Metrikler
from sklearn.model_selection import train_test_split, cross_val_score
import graphviz

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 90)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

warnings.simplefilter(action='ignore', category=Warning)  # Olası bazı warningleri ignore ediyoruz.

# %%
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

# %%

train_ = pd.read_csv("Odevler/HAFTA_08 YAPAY OGRENME PART 2/PROJE_I/HousePrice/train.csv")
test_ = pd.read_csv("Odevler/HAFTA_08 YAPAY OGRENME PART 2/PROJE_I/HousePrice/test.csv")

train = train_.copy()
test = test_.copy()

train.index = train["Id"]
test.index = test["Id"]

train.drop('Id', inplace=True, axis=1)
test.drop('Id', inplace=True, axis=1)

# Our data is now ready for EDA and other Data Preprocessing
# %%

train = train.loc[:, train.isnull().sum() < 0.8*train.shape[0]]
# Remove the columns whose sums of null values are bigger than %80 of total entries.

cat_cols, num_cols, cat_but_car = grab_col_names(train)

for col in num_cols:
    print(col, check_outlier(train, col))  # Result: We have outlier values in some variables.
for col in num_cols:
    replace_with_thresholds(train, col)

missing_values_table(train)

na_columns = missing_values_table(train, na_name = True)
train = train.drop(na_columns, axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(train)
rare_analyser(train, "SalePrice", cat_cols)
train = rare_encoder(train, 0.01)

binary_cols = [col for col in train.columns if train[col].dtype not in [int, float] and train[col].nunique() == 2]
for col in binary_cols:
    label_encoder(train, col)

ohe_cols = [col for col in train.columns if 11 >= train[col].nunique() > 2]
train = one_hot_encoder(train, ohe_cols)

train.drop('Neighborhood', inplace= True, axis=1 )

# %%
# Model

y = train[["SalePrice"]]
X = train.drop('SalePrice', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
cart_model = DecisionTreeRegressor(random_state=1).fit(X_train, y_train)

# Train RMSE :
y_pred = cart_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# Test RMSE
y_pred = cart_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# 10-Fold Cross Validate
np.mean(np.sqrt(-cross_val_score(cart_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# %%
#
cart_model.get_params()


cart_params = {'max_depth': [1,3,5,7,10, None],
               "min_samples_split": range(2, 20),
               'max_features': [None, 1,3,4,5,6,7]}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)

cart_best_grid.best_params_

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=1).fit(X, y)
cart_final.get_params()

np.mean(np.sqrt(-cross_val_score(cart_final,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))