# clean data
import constants


class CleanData:
    def __init__(self, df):
        self.df = df

    def clean_data(self, drop_duplicates=True, drop_missing=True):
        print("Data shape before cleaning: ", self.df.shape)

        # dropping text containing spaces only
        self.df = self.df[self.df[constants.TEXT].str.isspace() == False].copy()

        # dropping duplicate rows
        if drop_duplicates:
            self.df.drop_duplicates(inplace=True)

        # dropping rows with missing text
        if drop_missing:
            self.df.dropna(inplace=True)

        self.df.sort_values(by=constants.ID, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print("Data shape after cleaning: ", self.df.shape)

        return
