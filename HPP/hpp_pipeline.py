# Library Imports and Settings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#%pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from scipy.stats import norm
from sklearn.impute import KNNImputer
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import scipy
import scipy.stats as stats
from scipy.stats import skew,boxcox_normmax, zscore
from scipy.special import boxcox1p
from scipy.stats import boxcox
from scipy.stats import boxcox_normmax

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Method Definitions 
#######################################################
# For Dataset Loading
######################################################
def load_train():
    data = pd.read_csv("../HPP/train.csv")
    return data

def load_test():
    data = pd.read_csv("../HPP/test.csv")
    return data

def concat_df_on_y_axis(df_1, df_2):
    return pd.concat([df_1, df_2])

def concat_df_on_x_axis(df_1, df_2):
    return pd.concat([df_1, df_2], axis = 1)

def grab_col_names(dataframe, cat_th=10, car_th=20):

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


######################################################
# For Outliers
######################################################
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

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

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

######################################################
# For Missing Values
######################################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return missing_df, na_columns
    
    return missing_df

def quick_missing_imp_groupped(data, cat_cols, num_cols, missing_columns_df, num_method = 'median', groupby = 'Neighborhood', target = 'SalePrice'):
    missing_columns = missing_columns_df.T.columns
    missing_columns = [col for col in missing_columns if col not in target]
    missing_cat_cols = [col for col in missing_columns if col not in num_cols]
    missing_num_cols = [col for col in missing_columns if col not in cat_cols]

    print("# BEFORE")
    print(missing_columns_df)  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    data[missing_cat_cols] = data.groupby(groupby)[missing_cat_cols].transform(lambda x:x.fillna(x.mode()[0]))

    if num_method == "mean":
        data[missing_num_cols] = data.groupby(groupby)[missing_num_cols].transform(lambda x: x.fillna(x.mean()))
        
    elif num_method == "median":
        data[missing_num_cols] = data.groupby(groupby)[missing_num_cols].transform(lambda x: x.fillna(x.median()))
    
    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[missing_columns].isnull().sum(), "\n\n")

    return data


######################################################
# For Encoders
######################################################
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def one_hot_encoder_na_dummy(dataframe, categorical_cols, drop_first=True, dummy_na=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dummy_na = dummy_na)
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_analyser(dataframe, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe)}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

######################################################
# For EDA
######################################################
def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        #import seaborn as sns
        #import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    #print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


# PREPROCESSING
def hpp_data_prep():

    ## Dataset Reading
    df_train = load_train()
    df_test = load_test()
    df = concat_df_on_y_axis(df_train, df_test)
    df.shape


    df_copy = df.copy()
    check_df(df_copy)


    df_copy.drop(['Id'], axis = 1, inplace = True)
    cat_cols_eda, num_cols_eda, cat_but_car_eda = grab_col_names(df)


    nans = df.isna().sum().sort_values(ascending=False)
    nans = nans[nans > 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    ax.bar(nans.index, nans.values, zorder=2, color="#3f72af")
    ax.set_ylabel("No. of missing values", labelpad=10)
    ax.set_xlim(-0.6, len(nans) - 0.4)
    ax.xaxis.set_tick_params(rotation=90)
    plt.show()

    '''
    for feature in cat_cols_eda:
        cat_summary(df_train, feature)
        df_train.groupby(feature)['SalePrice'].mean().plot.bar()
        plt.title(feature + ' vs Sale Price')
        plt.show()


    for col in num_cols_eda:
        num_summary(df_train, col, True)
    '''

    y1 = df_train['SalePrice']
    plt.figure(2); plt.title('Normal')
    sns.distplot(y1, kde=False, fit=stats.norm)
    plt.figure(3); plt.title('Log Normal')
    sns.distplot(y1, kde=False, fit=stats.lognorm)


    #Log Transform
    #y = np.log(df_copy["SalePrice"])

    sns.set(font_scale=1.1)
    corr_train = df_train[num_cols_eda].corr()
    mask = np.triu(corr_train.corr())
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_train, annot=True, fmt='.2f', cmap='coolwarm', square=True, mask=mask, linewidth=1, cbar=True)
    plt.show()


    ## Grabbing Columns (NUM - CAT)
    cols_with_na_meaning = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                                'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']


    for col in cols_with_na_meaning:
        df_copy[col].fillna("None",inplace=True)


    encoding = {
        'None': 0,
        'Po': 1, 'No': 1, 'Unf': 1, 'Sal': 1, 'MnWw': 1,
        'Fa': 2, 'Mn': 2, 'LwQ': 2, 'Sev': 2, 'RFn': 2, 'GdWo': 2,
        'TA': 3, 'Av': 3, 'Rec': 3, 'Maj2': 3, 'Fin': 3, 'MnPrv': 3,
        'Gd': 4, 'BLQ': 4, 'Maj1': 4, 'GdPrv': 4,
        'Ex': 5, 'ALQ': 5, 'Mod': 5,
        'GLQ': 6, 'Min2': 6,
        'Min1': 7,
        'Typ': 8,
    }

    # Kodlamayı uygulayacağımız sütunlar listesi
    columns_to_encode = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 
                        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']

    # Kodlamayı uygulama
    for column in columns_to_encode:
        df_copy[column] = df_copy[column].map(encoding)


    cat_cols, num_cols, cat_but_car = grab_col_names(df_copy)


    print(cat_cols)
    print('-----------------------')

    print(num_cols)
    print('-----------------------')

    print(cat_but_car)


    known_num_cols = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'OverallCond', 'YrSold']


    for col in known_num_cols:
        if num_cols.__contains__(col) == False:
            num_cols.append(col)
            cat_cols.remove(col)

    for col in columns_to_encode:
        if num_cols.__contains__(col) == False:
            num_cols.append(col)
            cat_cols.remove(col)


    check_df(df_copy)


    num_cols_without_target = [col for col in num_cols if col not in 'SalePrice']


    df_copy[cat_cols] = df_copy[cat_cols].applymap(lambda x: x.replace(' ', '') if isinstance(x, str) else x)


    ## Suppressing Outliers of Numeric Columns
    print(len(num_cols))

    print(len(cat_cols))

    print('*------------------------*')

    for col in num_cols_without_target:
        print(col, check_outlier(df_copy, col))

    for col in num_cols_without_target:
        replace_with_thresholds(df_copy, col)

    print('------------------------')

    for col in num_cols_without_target:
        print(col, check_outlier(df_copy, col))

    print('*------------------------*')


    ## Dealing with Missing Values and Encoding
    missing_df, missing_columns = missing_values_table(df_copy, True)


    df_copy = quick_missing_imp_groupped(df_copy, cat_cols = cat_cols, num_cols = num_cols_without_target, missing_columns_df = missing_df)


    rare_analyser(df_copy, cat_cols)


    df_copy = rare_encoder(df_copy, 0.01)


    df_copy["NEW_1st*GrLiv"] = df_copy["1stFlrSF"] * df_copy["GrLivArea"]

    df_copy["NEW_Garage*GrLiv"] = (df_copy["GarageArea"] * df_copy["GrLivArea"])

    df_copy["TotalQual"] = df_copy[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                        "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence", 'BsmtQual', 'BsmtExposure', 
                        'GarageFinish','PoolQC']].sum(axis = 1) # 42


    # Total Floor
    df_copy["NEW_TotalFlrSF"] = df_copy["1stFlrSF"] + df_copy["2ndFlrSF"] # 32

    # Total Finished Basement Area
    df_copy["NEW_TotalBsmtFin"] = df_copy.BsmtFinSF1 + df_copy.BsmtFinSF2 # 56

    # Porch Area
    df_copy["NEW_PorchArea"] = df_copy.OpenPorchSF + df_copy.EnclosedPorch + df_copy.ScreenPorch + df_copy["3SsnPorch"] + df_copy.WoodDeckSF # 93

    # Total House Area
    df_copy["NEW_TotalHouseArea"] = df_copy.NEW_TotalFlrSF + df_copy.TotalBsmtSF # 156

    df_copy["NEW_TotalSqFeet"] = df_copy.GrLivArea + df_copy.TotalBsmtSF # 35


    # Lot Ratio
    df_copy["NEW_LotRatio"] = df_copy.GrLivArea / df_copy.LotArea # 64

    df_copy["NEW_RatioArea"] = df_copy.NEW_TotalHouseArea / df_copy.LotArea # 57

    df_copy["NEW_GarageLotRatio"] = df_copy.GarageArea / df_copy.LotArea # 69

    # MasVnrArea
    df_copy["NEW_MasVnrRatio"] = df_copy.MasVnrArea / df_copy.NEW_TotalHouseArea # 36

    # Dif Area
    df_copy["NEW_DifArea"] = (df_copy.LotArea - df_copy["1stFlrSF"] - df_copy.GarageArea - df_copy.NEW_PorchArea - df_copy.WoodDeckSF) # 73


    df_copy["NEW_OverallGrade"] = df_copy["OverallQual"] * df_copy["OverallCond"] # 61


    df_copy["NEW_Restoration"] = np.where(df_copy["YearRemodAdd"] < df_copy["YearBuilt"], 0, df_copy["YearRemodAdd"] - df_copy["YearBuilt"])

    df_copy["NEW_HouseAge"] = np.where(df_copy["YrSold"] < df_copy["YearBuilt"], 0, df_copy["YrSold"] - df_copy["YearBuilt"])

    df_copy["NEW_RestorationAge"] = np.where(df_copy["YrSold"] < df_copy["YearRemodAdd"], 0, df_copy["YrSold"] - df_copy["YearRemodAdd"])

    df_copy["NEW_GarageAge"] = np.abs(df_copy.GarageYrBlt - df_copy.YearBuilt) # 17

    df_copy["NEW_GarageRestorationAge"] = np.abs(df_copy.GarageYrBlt - df_copy.YearRemodAdd) # 30

    df_copy["NEW_GarageSold"] = np.where(df_copy["YrSold"] < df_copy["GarageYrBlt"], 0, df_copy["YrSold"] - df_copy["GarageYrBlt"])


    df_copy["NEW_TotalBaths"] = df_copy["FullBath"] + df_copy["BsmtFullBath"] + 0.5*(df_copy["HalfBath"]+df_copy["BsmtHalfBath"])


    df_copy['NEW_HasPool'] = df_copy['PoolArea'].apply(lambda x: 1 if x > 0 else 0)


    df_copy['NEW_Has2ndFloor'] = df_copy['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)


    df_copy['NEW_HasGarage'] = df_copy['GarageCars'].apply(lambda x: 1 if x > 0 else 0)


    df_copy['NEW_HasBsmt'] = df_copy['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)


    df_copy['NEW_HasFireplace'] = df_copy['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


    df_copy['NEW_HasPorch'] = df_copy['NEW_PorchArea'].apply(lambda x: 1 if x > 0 else 0)

    new_num_features = ['NEW_1st*GrLiv', 'NEW_Garage*GrLiv', 'TotalQual', 'NEW_TotalFlrSF', 'NEW_TotalBsmtFin', 'NEW_PorchArea', 'NEW_TotalHouseArea', 'NEW_TotalSqFeet',
                    'NEW_LotRatio', 'NEW_RatioArea', 'NEW_GarageLotRatio', 'NEW_MasVnrRatio', 'NEW_DifArea', 'NEW_OverallGrade', 'NEW_Restoration', 'NEW_HouseAge', 
                    'NEW_RestorationAge',  'NEW_GarageAge',  'NEW_GarageRestorationAge',  'NEW_GarageSold', 'NEW_TotalBaths']

    new_cat_features = ['NEW_HasPool', 'NEW_Has2ndFloor', 'NEW_HasGarage',
                    'NEW_HasBsmt', 'NEW_HasFireplace', 'NEW_HasPorch']


    num_cols.extend(new_num_features)
    cat_cols.extend(new_cat_features)


    ## Data Transform & Feature Scaling
    ### Data Transformation
    #By looking at the data we can say that "MSSubClass" and "YrSold" are Catagorical Variables, so we transform them into dtype : object
    df_copy[["MSSubClass", "YrSold"]] = df_copy[["MSSubClass", "YrSold"]].astype("category") #converting into catagorical value


    num_to_cat = ["MSSubClass", "YrSold"]


    for col in num_to_cat:
        if cat_cols.__contains__(col) == False:
            cat_cols.append(col)
            num_cols.remove(col)


    #"MoSold" is a Cyclic Value. We handle this type of data by mapping each cyclical variable onto a circle such that the lowest value for that variable appears right next to the largest value. We compute the x- and y- component of that point using sine and cosin trigonometric functions.
    df_copy["MoSoldsin"] = np.sin(2 * np.pi * df_copy["MoSold"] / 12) #Sine Function
    df_copy["MoSoldcos"] = np.cos(2 * np.pi * df_copy["MoSold"] / 12) #Cosine Function
    df_copy = df_copy.drop("MoSold", axis=1)


    num_cols.append('MoSoldsin')
    num_cols.append('MoSoldcos')
    num_cols.remove('MoSold')


    def log_plots(df, col):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(df[col], bins=30, edgecolor='black', color='blue')
        plt.title('Original Data Distribution')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # Step 3: Apply Log Transformation
        #df_copy[col] = np.log(df_copy[col] + 1e-6)

        # Step 4: Visualize Transformed Data Distribution
        plt.subplot(1, 2, 2)
        plt.hist(np.log1p(df[col]), bins=30, edgecolor='black', color='green')
        plt.title('Log-Transformed Data Distribution')
        plt.xlabel('Log('+ col +')')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    '''
    for col in num_cols:
        if col != 'MoSoldsin' and col != 'MoSoldcos':
            log_plots(df_copy, col)
    '''
    '''
    #Since our data is skewed we apply boxcox normalization and transform the skwed data
    skewed = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
        'ScreenPorch', 'PoolArea', 'LowQualFinSF', 'MiscVal', 'TotRmsAbvGrd', 
        'NEW_1st*GrLiv', 'NEW_Garage*GrLiv', 'TotalQual', 'NEW_TotalFlrSF', 
        'NEW_TotalBsmtFin', 'NEW_PorchArea', 'NEW_TotalHouseArea', 'NEW_TotalSqFeet', 
        'NEW_LotRatio', 'NEW_RatioArea'
    ]
    
    for col in skewed:
    
        # Step 3: Apply Log Transformation
        df_copy[col] = np.log1p(df_copy[col])
    '''

    num_cols_without_target = [col for col in num_cols if col not in 'SalePrice']

    scaler = RobustScaler()

    df_copy[num_cols_without_target] = pd.DataFrame(scaler.fit_transform(df_copy[num_cols_without_target]))

    ## Encoding
    df_copy = one_hot_encoder(df_copy, cat_cols)
    df_copy = one_hot_encoder(df_copy, cat_but_car)


    ## Final
    check_df(df_copy)


    split_index = 1460

    # İlk parça: 0'dan split_index'e kadar
    preprocessed_train = df_copy.iloc[:split_index]

    # İkinci parça: split_index'ten sona kadar
    preprocessed_test = df_copy.iloc[split_index:]


    check_df(preprocessed_train)

    preprocessed_test = preprocessed_test.drop(["SalePrice"], axis=1)

    #LOCAL OUTLIER FACTOR
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(preprocessed_train)

    df_scores = clf.negative_outlier_factor_
    print(df_scores[0:5])
    # df_scores = -df_scores
    print(np.sort(df_scores)[0:5])

    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked=True, xlim=[0, 10], style='.-')
    plt.show()
    th = np.sort(df_scores)[2]

    print(preprocessed_train[df_scores < th])

    print(preprocessed_train[df_scores < th].shape)


    print(preprocessed_train.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T)

    print(preprocessed_train[df_scores < th].index)


    preprocessed_train[df_scores < th].drop(axis=0, labels=preprocessed_train[df_scores < th].index)


    y = np.log(preprocessed_train["SalePrice"])
    X = preprocessed_train.drop(["SalePrice"], axis=1)

    rf_model = RandomForestRegressor(random_state=46).fit(X, y)


    ######################################################
    # Feature Importance
    ######################################################

    def plot_importance(model, features, num=len(X), save=False):
        feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
        plt.figure(figsize=(25, 25))
        sns.set(font_scale=1)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                        ascending=False)[0:num])
        plt.title('Features')
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig('importances.png')


    plot_importance(rf_model, X)


    ## TO CSV
    preprocessed_train.to_csv('train_preprocessed.csv', index = False)
    preprocessed_test.to_csv('test_preprocessed.csv', index = False)

    return X, y, preprocessed_test

## Imports
#%pip install lightgbm
#%pip install catboost
#%pip install xgboost
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

## Methods
def load_train_prep():
    data = pd.read_csv("../HPP/train_preprocessed.csv")
    return data

def load_test_prep():
    data = pd.read_csv("../HPP/test_preprocessed.csv")
    return data

######################################################
# Base Models
######################################################


def base_models(X, y, scoring = 'neg_root_mean_squared_error'):
    print("Base Models....")
    models = [('LR', LinearRegression()),
                   ("Ridge", Ridge()),
                   ("Lasso", Lasso()),
                   ("ElasticNet", ElasticNet()),
                   ('KNN', KNeighborsRegressor()),
                   ('CART', DecisionTreeRegressor()),
                   ('RF', RandomForestRegressor()),
                   ('SVR', SVR()),
                   ('GBM', GradientBoostingRegressor()),
                   ("XGBoost", XGBRegressor(objective = 'reg:squarederror')),
                   ("LightGBM", LGBMRegressor()),
                   ("CatBoost", CatBoostRegressor(verbose=False))
          ]

    for name, regressor in models:
        cv_results = cross_validate(regressor, X, y, cv = 5, scoring = scoring)
        print(f"{scoring}: {round(-cv_results['test_score'].mean(), 4)} ({name}) ")

    #for name, regressor in models:
        #rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring=scoring)))
        #print(f"RMSE: {round(rmse, 4)} ({name}) ")



######################################################
# Automated Hyperparameter Optimization
######################################################

# Ridge Regression parameters
ridge_params = {"alpha": [0.1, 1.0, 10.0, 100.0]}

# Lasso Regression parameters
lasso_params = {"alpha": [0.01, 0.1, 1.0, 10.0]}

# ElasticNet parameters
elasticnet_params = {"alpha": [0.1, 1.0, 10.0],
                     "l1_ratio": [0.1, 0.5, 0.9]}

# K-Nearest Neighbors parameters
knn_params = {"n_neighbors": range(2, 50)}

# Decision Tree Regressor parameters
cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

# Random Forest Regressor parameters
rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

# Support Vector Regressor parameters
svr_params = {'kernel': ['linear', 'rbf'],
              'C': [0.1, 1, 10],
              'gamma': ['scale', 'auto']}

# Gradient Boosting Regressor parameters
gbm_params = {"learning_rate": [0.01, 0.1],
              "n_estimators": [100, 200],
              "max_depth": [3, 5, 7]}

# XGBoost parameters
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

# LightGBM parameters
lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}

# CatBoost parameters
catboost_params = {"depth": [4, 6, 10],
                   "learning_rate": [0.01, 0.1],
                   "iterations": [100, 200]}

# Creating a list of models and their respective parameter grids
regressors = [
    ("Ridge", Ridge(), ridge_params),
    ("Lasso", Lasso(), lasso_params),
    ("ElasticNet", ElasticNet(), elasticnet_params),
    ('KNN', KNeighborsRegressor(), knn_params),
    ('CART', DecisionTreeRegressor(), cart_params),
    ('RF', RandomForestRegressor(), rf_params),
    #('SVR', SVR(), svr_params),
    ('GBM', GradientBoostingRegressor(), gbm_params),
    ("XGBoost", XGBRegressor(objective='reg:squarederror'), xgboost_params),
    ("LightGBM", LGBMRegressor(), lightgbm_params),
    ("CatBoost", CatBoostRegressor(verbose=False), catboost_params)
]



def hyperparameter_optimization(X, y, cv = 5, scoring = "neg_root_mean_squared_error"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv = cv, scoring = scoring)
        print(f"{scoring} (Before): {round(-cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv = cv, scoring = scoring)
        print(f"{scoring} (After): {round(-cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end = "\n\n")
        best_models[name] = final_model
    return best_models


######################################################
# Stacking & Ensemble Learning
######################################################

def voting_regressor(best_models, X, y):
    print("Voting Regressor...")

    voting_reg = VotingRegressor(estimators=[('CatBoost', best_models["CatBoost"]),
                                              ('GBM', best_models["GBM"]),
                                              ('XGBoost', best_models["XGBoost"]),
                                              ('Lasso', best_models["Lasso"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"]),
                                              ('ElasticNet', best_models["ElasticNet"]),
                                              #('Ridge', best_models["Ridge"]),
                                              ], 
                                ).fit(X, y)

    cv_results = cross_validate(voting_reg, X, y, cv = 5, scoring = 'neg_root_mean_squared_error')
    print(f"RMSE: {-cv_results['test_score'].mean()}")
    return voting_reg



def main():
    X, y, test_prep = hpp_data_prep()

    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_reg = voting_regressor(best_models, X, y)
    predictions = voting_reg.predict(test_prep)

    dictionary = {"Id":test_prep.index + 1461, "SalePrice":predictions}
    dfSubmission = pd.DataFrame(dictionary)

    #dfSubmission['SalePrice'] = pd.DataFrame(scaler.inverse_transform(dfSubmission['SalePrice']))
    dfSubmission['SalePrice'] = np.exp(dfSubmission['SalePrice'])
    
    dfSubmission.to_csv("housePricePredictions.csv", index=False)

    joblib.dump(voting_reg, "voting_reg.pkl")

    return voting_reg


if __name__ == "__main__":
    print("Process Started...")
    main()
