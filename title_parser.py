from string_grouper import group_similar_strings, compute_pairwise_similarities
from sklearn.feature_extraction.text import *
from spellchecker import SpellChecker
from wordsegment import load, segment
import pandas as pd
import numpy as np
import nltk

# Installing NLTK Data that will be used for preprocessing
nltk.download('stopwords', quiet=True) # Used to exclude stopwords from features
nltk.download('wordnet', quiet=True) # Used for part of speech (pos) tagging
nltk.download('averaged_perceptron_tagger', quiet=True)
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
wordnet = nltk.corpus.wordnet
load()

class TitleParser:
  """Reduces a title to a common one. It is aimed to address mistyped titles"""
  emp_title_map = {}
  emp_title_pd = None
  title_correction_map = {} # Caching wrong and correct titles to save time
  word_correction_map = {} # Caching wrong and correct word to save time
  title_similarity_map = {}
  spell = SpellChecker(distance=1)

  def preprocess(self, df):
    df['emp_title'].replace(to_replace = r'\d{1,}', value = ' ', regex = True, inplace=True) # remove numbers
    df['emp_title'].replace(to_replace = r'[^a-zA-Z0-9_]', value = ' ', regex = True, inplace=True) # remove all special characters
    df['emp_title'].replace(to_replace = r'\s+', value = ' ', regex = True, inplace=True) # remove all extra spaces
    df['emp_title'].replace(to_replace = r'[0-9]', value = ' ', regex=True, inplace=True) # remove numbers
    df['emp_title'].replace(to_replace = r'\.', value = ' ', regex = True, inplace=True) # remove periods
    df['emp_title'].replace(to_replace = r'\'', value = ' ', regex = True, inplace=True) # remove apostrophes 
    df['emp_title'].replace(to_replace = r'-', value = ' ', regex = True, inplace=True) # remove hyphens
    df['emp_title'].replace(to_replace = r'\(', value = ' ', regex = True, inplace=True) # remove (
    df['emp_title'].replace(to_replace = r'\)', value = ' ', regex = True, inplace=True) # remove )
    df['emp_title'].replace(to_replace = ',', value = ' ', regex = True, inplace=True) # remove commas
    df['emp_title'].replace(to_replace = r'\\', value= ' ', regex=True, inplace=True) # remove backslash    
    df['emp_title'].replace(to_replace = r'/', value= ' ', regex=True, inplace=True) # remove forward slash
    df['emp_title'].replace(to_replace = r"^ +| +$", value= r"", regex=True, inplace=True) # remove leading and trailing spaces
    df['emp_title'] = df['emp_title'].str.lower() # convert to lowercase
    df['emp_title'] = df['emp_title'].str.strip() # remove leading spaces
    df['emp_title'] = df['emp_title'].map(self.correct_spell)
    
    return df

  def pos(self, word):
    """ Generates a pos to be used by lemmatizer
    """
    # lemmatize function part of speech (pos) attribute only recongnizes
    # nouns, verbs, adjectives, adverbs and satellite adjectives
    pos = nltk.pos_tag([word])
    tag = pos[0][1][0]
    if tag == 'N':
        return wordnet.NOUN
    elif tag == 'V':
        return wordnet.VERB
    elif tag == 'J':
        return wordnet.ADJ
    elif tag == 'R':
        return wordnet.ADV
    elif tag == 'S':
        return wordnet.ADJ_SAT
    else:
        return wordnet.NOUN
  
  def correct_spell(self, title):
    if (isinstance(title, float) and np.isnan(title)) or title == "":
      return ""
    
    if title in self.title_correction_map:
      return self.title_correction_map[title]

    title_split = self.spell.split_words(title)
    misspelled = self.spell.unknown(title_split)
    # If there is no spell issue then return the title as it is
    new_title = ""
    if len(misspelled) == 0:
      new_title = title
    else:
      for word in misspelled:
        if word in self.word_correction_map:
          correct_word = self.word_correction_map[word]
        else:
          # Get the one `most likely` answer
          correct_word = self.spell.correction(word)
          # Try to split it if it is unable to correct it
          if correct_word == word:
              correct_word = ' '.join(segment(word))
          # This word must be junk so remove it
          if word == correct_word:
            del title_split[title_split.index(word)]
            continue
          else:
            self.word_correction_map[word] = correct_word
        # Replacing with the correct one
        title_split[title_split.index(word)] = correct_word
      new_title = ' '.join(title_split)
    # Remove common words
    title_split = self.spell.split_words(new_title)
    clean_title = []
    for w in title_split:
      if w not in stopwords:
        clean_title.append(lemmatizer.lemmatize(w, self.pos(w)))
    new_title = ' '.join(clean_title)
    self.title_correction_map[title] = new_title

    return new_title

  def fit(self, df):
    # Caching the result in a csv file once it is comupted
    df = self.preprocess(df.copy())
    emp_title = df.loc[:,'emp_title'].unique()
    emp_title = pd.DataFrame(data=emp_title, columns=['title'])
    emp_title.dropna(inplace=True)
    # You can play around with the threshold value for min_similarity, a value lower than 0.7 is not recommended
    emp_title[["group_id", "similar_title"]] = group_similar_strings(emp_title['title'], min_similarity=0.85, n_blocks = 'auto')
    self.emp_title_pd = emp_title
    # Putting the data into dictionery for fast access as panda filtering is very very slow
    for i in self.emp_title_pd.index:
      self.emp_title_map[self.emp_title_pd['title'][i]] = self.emp_title_pd['similar_title'][i]

  def transform(self, df):
    df = self.preprocess(df.copy())
    df['new_title'] = df['emp_title'].map(self.map_title)
    return df

  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)

  def get_title_dataframe(self):
    return self.emp_title_pd

  def get_title_dict(self):
    return self.emp_title_map

  def map_title(self, title):
    if (isinstance(title, float) and np.isnan(title)) or title == "":
      return ""
    # Check if there is exact title matching
    if title in self.emp_title_map:
      return self.emp_title_map[title]
    # Check if similar title is there
    elif title in self.title_similarity_map:
      return self.title_similarity_map[title]
    # Further matching can be done using compute_pairwise_similarities
    # but it is very very slow. it takes 11 minutes for 1000 records with cache
    #else:
    #  n_titles = pd.Series([title]*self.emp_title_pd.shape[0])
    #  similarity = compute_pairwise_similarities(n_titles, self.emp_title_pd.title)
    #  max = similarity.idxmax()
    #  # Picking a similar title
    #  if similarity.get(max) > 0.4:
    #    print(str(similarity.get(max))+" -> "+n_titles.get(max)+" -> "+self.emp_title_pd['title'].get(max))
    #    similar_title = self.emp_title_pd['similar_title'].get(max)
    #    self.title_similarity_map[title] = similar_title
    #    return similar_title
    return title