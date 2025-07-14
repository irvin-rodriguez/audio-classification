from sklearn.model_selection import train_test_split

# Given a dataframe and number of mfcc to use, this function
# extracts the need X and y values to begin training
def get_feature_target(df, num_mfcc):
    mfcc_cols = [f"MFCC_{i + 1}" for i in range(num_mfcc)]
    X = df[mfcc_cols].values
    y = df['digit'].values
    return X, y

def create_stratify_label(df):
    return df['digit'].astype(str) + "_" + df['gender'].astype(str) + "_" + df['speaker'].astype(str)

def split_data(df, num_mfcc, test_size=0.2, random_state=42):
    X, y = get_feature_target(df, num_mfcc)
    stratify_labels = create_stratify_label(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_labels, random_state=random_state
    )
    return X_train, X_test, y_train, y_test