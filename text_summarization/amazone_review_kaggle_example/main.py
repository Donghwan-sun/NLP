import numpy as np
import pandas as pd
import re
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


# 데이터 로드
data = pd.read_csv('./data/archive/Reviews.csv')
data.head()

