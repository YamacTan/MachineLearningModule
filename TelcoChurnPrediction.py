#######################
# Yamac TAN - Data Science Bootcamp - Week 8 - Project 2
#######################

# %%

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split

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

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

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


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
# %%

df = pd.read_csv("Odevler/HAFTA_08 YAPAY OGRENME PART 2/PROJE_II/Telco-Customer-Churn.csv")
df.shape
df.isnull().sum()
df.describe().T
df.head(15)

# %%
# Görev 1

#Adım 1
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Adım 2
df.dtypes

for x in range (len(df)):
    if (df.loc[x, "TotalCharges"]) == " ":
        df.loc[x, "TotalCharges"] = np.nan
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df.dtypes  #Checked

#Adım 3 -
for col in num_cols:
    num_summary(df, col)
for col in cat_cols:
    cat_summary(df, col)

#Adım 4

# The reason for the error in the target variable analysis in this section is that the Churn variable has
# "Yes" and "No" values. This will be fixed after binary encoding.
for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

df.groupby("Churn").mean()

#Adım 5

for col in num_cols:
    print(col, check_outlier(df, col))
# There are not any outlier values.

#Adım 6:

missing_values_table(df)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())



# %%
# Görev 2

#Adım 1:

for col in num_cols:
    print(col, check_outlier(df, col))

missing_values_table(df)

# Datasette eksik ya da aykırı gözlem bulunmadığından bir işlem yapmaya gerek yoktur.

#Adım 2:
df.loc[(df['tenure'] > 33), "customer_class"] = "loyal"
df.loc[(df['tenure'] <= 33), "customer_class"] = "standard"

#Adım 3:

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

#Adım 4:

cat_cols, num_cols, cat_but_car = grab_col_names(df)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# %%
# Görev 3

df.head()
df.drop("customerID", axis = 1, inplace= True)

y = df["Churn"]
X = df.drop(["Churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


# 1 - DecisionTreeClassifier
cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Cross validate, yukarıda yazdıgımız fit methodunu görmezden gelir.

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7299427788523534
cv_results['test_f1'].mean()
# 0.49946105770892774
cv_results['test_roc_auc'].mean()
# 0.6604916232362472


# 2 - RandomForestClassifier

rf_model = RandomForestClassifier(random_state=1)
rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7931262088974854
cv_results['test_f1'].mean()
# 0.556658842144538
cv_results['test_roc_auc'].mean()
# 0.8269607481133804

# 3 - GBM
gbm_model = GradientBoostingClassifier(random_state=1)
gbm_model.get_params()

cv_results = cross_validate(gbm_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8050531914893616
cv_results['test_f1'].mean()
# 0.5883868707258069
cv_results['test_roc_auc'].mean()
# 0.847430122875098

# 4 - XGBoost

xgboost_model = XGBClassifier(random_state=1, use_label_encoder=False)
xgboost_model.get_params()

cv_results = cross_validate(xgboost_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7881570357833656
cv_results['test_f1'].mean()
# 0.5630368305374369
cv_results['test_roc_auc'].mean()
# 0.824520556433248


# 5 - LightGBM

lgbm_model = LGBMClassifier(random_state=1)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7983808833010961
cv_results['test_f1'].mean()
# 0.5791640459704356
cv_results['test_roc_auc'].mean()
# 0.8361987945059159


# 6 - Catboost Model

catboost_model = CatBoostClassifier(random_state=1, verbose=False)
catboost_model.get_params()

cv_results = cross_validate(catboost_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.799940965506125
cv_results['test_f1'].mean()
# 0.5785616313995834
cv_results['test_roc_auc'].mean()
# 0.8413579068151866

# Best 4 Models for Accuracy:
#  1 - GBM
#  2 - CatBoost
#  3 - LightGBM
#  4 - RandomTreeClassifier

# %%

# GBM Parameter Optimization
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 5, 7],
              "n_estimators": [100, 500, 600],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=1 ).fit(X, y)


# CatBoost Parameter Optimization

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid.best_params_
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=1).fit(X, y)


# LightGBM Parameter Optimization
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500],
               "colsample_bytree": [0.5, 1]}


lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=1).fit(X, y)




# RandomForest Parameter Optimization
rf_params = {"max_depth": [5, None],
             "max_features": [3, 5, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=1).fit(X, y)

# %%

plot_importance(gbm_final, X)
gbm_most_importants = X[["tenure", "InternetService_Fiber optic", "MonthlyCharges", "TotalCharges"]]
plot_importance(catboost_final, X)
catboost_most_importants = X[["tenure", "Contract_Two year", "MonthlyCharges", "TotalCharges"]]
plot_importance(lgbm_final, X)
lgbm_most_importants = X[["tenure", "PaperlessBilling", "MonthlyCharges", "TotalCharges"]]
plot_importance(rf_final, X)
rf_most_importants = X[["tenure", "customer_class", "MonthlyCharges", "TotalCharges"]]

# %%

#GBM Final with Best Params and Most Importants
cv_results = cross_validate(gbm_final, gbm_most_importants, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7934098968407479
cv_results['test_f1'].mean()
# 0.5546490779277797
cv_results['test_roc_auc'].mean()
# 0.8293471840199971

#Catboost Final with Best Params and Most Importants
cv_results = cross_validate(catboost_final, catboost_most_importants, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7939780786589298
cv_results['test_f1'].mean()
# 0.5360153730453043
cv_results['test_roc_auc'].mean()
# 0.8306218646612062

#LGBM Final with Best Params and Most Importants
cv_results = cross_validate(lgbm_final, lgbm_most_importants, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7908558994197292
cv_results['test_f1'].mean()
# 0.5243213309766566
cv_results['test_roc_auc'].mean()
# 0.821946174861384

#LGBM Final with Best Params and Most Importants
cv_results = cross_validate(rf_final, rf_most_importants, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7706935041908445
cv_results['test_f1'].mean()
# 0.5153137668221844
cv_results['test_roc_auc'].mean()
# 0.7811300304750696

# When the models whose parameters are optimized are run again with the "main" variables,which are the most weighted in
# terms of importance, they give almost similar results with a training set that contains too many variables.
# What can be understood here is that the contribution of the variables in the dataset that are not of high importance to
# the model success is quite low.
