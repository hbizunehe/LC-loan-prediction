from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np

class Encoders:
  """Builds and maintains one-hot feature encoding"""
  one_hot = {} # It stores list of transformers that map feature-value to vectors
  droped_column = {} # Dropped column for each feature encoding

  def fit(self, df, features):
    """Learn a map from a feature value to indices"""
    for feature in features:
      #if len(df[feature].unique()) <= 1002: # TODO We will need all columns to make it easy for decision tree and other algorithms
      self.droped_column[feature] = np.sort(df[feature].unique())[0]
      records = df[[feature]].to_dict('records')
      self.one_hot[feature] = DictVectorizer(sparse=False)
      self.one_hot[feature].fit(records)

  def transform(self, df):
    """Transforms feature values to array and merge it to original dataframe"""
    for feature, encoder in self.one_hot.items():
      records = df[[feature]].to_dict('records')
      one_hot = encoder.transform(records)
      names = self.names(feature)
      new_cols = pd.DataFrame(one_hot, columns=names)
      df = df.drop([feature], axis=1)
      df = pd.concat([df, new_cols], axis=1)
    # Dropping one column for each feature as it will be repetitive info
    for feature, value in self.droped_column.items():
      df = df.drop([feature+"="+value], axis=1)
    return df

  def fit_transform(self, df, features):
    """It performes fit first and then transform on the data"""
    self.fit(df, features)
    return self.transform(df)
  
  def vocabulary(self, feature):
    return self.one_hot[feature].vocabulary_
  
  def names(self, feature):
    return self.one_hot[feature].get_feature_names()