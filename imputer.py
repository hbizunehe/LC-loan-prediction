from sklearn.impute import KNNImputer
import pandas as pd
import joblib
from pathlib import Path

class Imputer:
  """Performs imputation for Nan values using KNN.
  It also performs caching as the imputation takes hours to complete"""
  knn_imputer = KNNImputer()

  def fit(self, df):
    """Fitting KNN imputer"""
    self.knn_imputer.fit(df)

  def transform(self, df):
    """Replace Nan values"""
    # hash will be used as a unique file name for the given dataset
    hash = joblib.hash(df["loan_amnt"])
    file_name = f"./data/imputed/{hash}.csv"
    # KNNImputer takes close to 4 hours for training data and an hour to dev data
    # So caching the result once it is comupted
    if Path(file_name).is_file():
      df_imputed = pd.read_csv(file_name, engine='c')
    else:
      df_imputed = self.knn_imputer.transform(df)
      # Caching the imputer result on hard drive for future use
      df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
      df_imputed.to_csv(file_name)
    # This `Unnamed: 0` column is added by pandas from the index of ndarray. It is not required.
    df_imputed.drop(columns=['Unnamed: 0'], inplace=True)
    return df_imputed

  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)