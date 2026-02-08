import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
def preprocessing_text(x):
    x = x.lower()
    x = x.translate(str.maketrans('', '', string.punctuation))
    words = stopwords.words('english')
    x = ' '.join([word for word in x.split() if word not in words])
    ps = PorterStemmer()
    ps.stem(x)
    x = word_tokenize(x)
    return ' '.join(x)