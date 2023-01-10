import constants
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


class TextPreprocessing:
    def __init__(self, df):
        self.df = df

    def preprocess(self,
                   to_lower=True,
                   remove_punctuations=True,
                   lemmatize=True,
                   remove_stopwords=False
                   ):

        # converting to lower string
        self.df['raw_text'] = self.df[constants.TEXT].copy()

        if to_lower:
            print('Converting to lower case..')
            self.df[constants.TEXT] = self.df[constants.TEXT].str.lower()

        # removing punctuations
        if remove_punctuations:
            print('Removing Punctuations..')
            self.df[constants.TEXT] = self.df[constants.TEXT].str.replace(r'[^\w\s]', '', regex=True)

        # tokenization
        print('Tokenization..')
        self.df[constants.TEXT] = self.df[constants.TEXT].apply(word_tokenize)

        # lemmatization
        if lemmatize:
            print('Lemmatization..')
            lm = WordNetLemmatizer()
            self.df[constants.TEXT] = self.df[constants.TEXT].apply(
                lambda words: [lm.lemmatize(word) for word in words])

        # treating stop words
        # not removing stop words by default because past research suggests that social media content is short
        # and performance goes down on removing stop words
        if remove_stopwords:
            print('Removing stopwords..')
            self.df[constants.TEXT] = self.df[constants.TEXT].apply(
                lambda words: [w for w in words if w not in stopwords.words("english")])

        # joining text back
        print("Joining words to text..")
        self.df[constants.TEXT] = self.df[constants.TEXT].apply(lambda words: ' '.join(words))

        print("\n Shape of Data: ", self.df.shape)
        print("\nSnapshot of Data: \n", self.df.head())

        return
