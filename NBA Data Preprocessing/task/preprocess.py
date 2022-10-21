import os

import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Check for ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"


# write your code here

def clean_data(_data_path):
    # setting option to see all rows and columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = pd.read_csv(_data_path)

    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df["draft_year"] = pd.to_datetime(df['draft_year'], format='%Y')

    # df.height = df.height.apply(lambda col: col.strip().split()[-1])
    df["height"] = df["height"].str.split("/", expand=True)[1].str.strip().astype(float)
    # df[["height", "weight", "salary"]] = df[["height", "weight", "salary"]].astype(np.float64)
    df["weight"] = df["weight"].str.split("/", expand=True)[1].str.split(" ", expand=True)[1].str.strip().astype(float)

    df.loc[df["team"].isnull(), "team"] = "No Team"
    df.loc[df["country"] != "USA", "country"] = "Not-USA"
    # df.salary = df.salary.apply(lambda value: value.strip().replace("$", ""))
    df["salary"] = df["salary"].str[1:].astype(float)

    df.loc[df["draft_round"] == "Undrafted", "draft_round"] = "0"
    return df


data_ = clean_data(data_path)


# print(data[['b_day', 'team', 'height', 'weight', 'country', 'draft_round', 'draft_year', 'salary']].head())


def feature_data(data):
    data['version'] = pd.to_datetime(data['version'], format='NBA2k%y')
    # data['diff'] = (data['version'] - data['b_day']) / np.timedelta64(1, "Y")
    data['age'] = data['version'].dt.year - data['b_day'].dt.year
    data['experience'] = data['version'].dt.year - data['draft_year'].dt.year
    data['bmi'] = (data['weight'] / data['height'] ** 2).round(6)
    data.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'], inplace=True)

    r = data[['age', 'experience', 'bmi']]

    n = data.nunique()

    columns = []
    for item in data:
        # finding the columns which have more than 50 unique data
        if data[item].nunique() > 50 and data[item].dtype == object:
            columns.append(item)
    data.drop(columns=columns, inplace=True)

    # print(data.dtypes)
    # print(r.head(5))
    # print(n)
    return data


def multicol_data(df_):
    # leave only the columns we need
    # df_c = df_[['rating', 'age', 'experience', 'bmi']]
    # calculating the correlation
    corr = df_.corr(numeric_only=True)
    # print("Correlation matrix:", corr, sep="\n")
    multicol_set = set()
    for ind in corr.index:
        for col in corr.columns:
            # finding multicollinearity features
            if abs(corr.at[ind, col]) > 0.5 and ind != col and "salary" not in (ind, col):
                if corr.at["salary", ind] <= corr.at["salary", col]:
                    # comparing target feature with multicollinearity feature selecting the lower to drop
                    multicol_set.add(ind)
                else:
                    multicol_set.add(col)
    # print("multicollinearity set:", multicol_set)
    # target is salary
    # age, and experience have multicollinearity (age, experience: 0.92)
    # correlation between (salary, age: 0.44), (salary, experience: 0.53)
    # age has lower correlation, we drop age which has lower correlation with salary
    df_.drop(columns=multicol_set, inplace=True)
    return df_


def transform_data(df):
    num_feat_df = df.select_dtypes('number')  # numerical features
    # print(num_feat_df.describe())
    features = num_feat_df[["rating", "experience", "bmi"]]  # removing target (salary) to scale the rest
    scale = StandardScaler()
    num_feat_scaled_df = pd.DataFrame(scale.fit_transform(features), columns=features.columns)  # scaling the dataframe
    # print(num_feat_scaled_df)

    cat_feat_df = df.select_dtypes('object')  # categorical features
    # print(cat_feat_df.describe())

    """ method_1
    # create numeric labels
    label_encoder = LabelEncoder()
    for col in cat_feat_df:
        cat_feat_df[col] = label_encoder.fit_transform(cat_feat_df[col])
    # binary encode (OneHotEncoder)
    onehot_encoder = OneHotEncoder(sparse=False)
    enc_data = pd.DataFrame(onehot_encoder.fit_transform(cat_feat_df))
    """

    # method 2
    enc_data = pd.get_dummies(cat_feat_df, prefix="", prefix_sep="")
    final_df = pd.concat([num_feat_scaled_df, enc_data], axis=1)
    # print(final_df.head(3))
    return final_df, num_feat_df["salary"]


df_cleaned = clean_data(data_path)
df_featured = feature_data(df_cleaned)
df = multicol_data(df_featured)
X, y = transform_data(df)

answer = {
    'shape': [X.shape, y.shape],
    'features': list(X.columns),
}
print(answer)
