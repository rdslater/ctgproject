import pandas as pd


def replace_rare(dataframe, column, num_values=10):
    tmp = dataframe[column].value_counts()
    top_vals = tmp[0:num_values].index.to_list()
    dataframe.loc[~dataframe[column].isin(top_vals), column] = "OTHER"
    return 0


def data_prep():
    """
    Reads in a hard coded data frame and does cleanup.
    First it selects only customers that have booked a destination.
    All dates are converted to timestamps
    3 columns have rare data, so I created an "other" category for each
    All Categorical variables are then OHE with get_dummies()
    Target Variauble is encoded.
    Final numeric data frame and map of class to country are returned.
    """
    train_data = pd.read_csv("./data/train_users.csv")
    first_booked = train_data
    first_booked["timestamp_first_active"] = pd.to_datetime(
        first_booked["timestamp_first_active"], format="%Y%m%d%H%M%S"
    ).astype("int64")

    first_booked["date_account_created"] = pd.to_datetime(
        first_booked["date_account_created"], format="%Y-%m-%d"
    ).astype("int64")

    first_booked["date_first_booking"] = pd.to_datetime(
        first_booked["date_first_booking"], format="%Y-%m-%d"
    ).astype("int64")

    # List of columns containing rare categorical data
    rare_data_cols = [
        "first_device_type",
        "affiliate_provider",
        "language",
        "first_browser",
    ]
    encoded = first_booked.columns.to_list()
    for col in rare_data_cols:
        replace_rare(first_booked, col)
    indexes = [4, 6, 8, 9, 10, 11, 12, 13, 14]
    encoded = [encoded[i] for i in indexes]
    data = pd.get_dummies(first_booked, columns=encoded)
    country_data = pd.factorize(data['country_destination'])
    data['country_destination'] = country_data[0]
    return data, country_data[1]


if __name__ == "__main__":
    pass
