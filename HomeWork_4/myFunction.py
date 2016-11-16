from sklearn.preprocessing import LabelEncoder
import numpy as np


from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier


 
def MeanDecreaseAccuracy(CrowdstormingDataJuly1st_Array,TargetVAriableRater,ShuffleSplitNumber,
                         RandomState,CrowdstormingDataJuly1st):
    '''
    This gives the impact of each feature on the accuracy of the model. We permute the values of each 
    feature and measure how much the permutation decreases the accuracy of the model. Permuting important 
    variables should affect significantly the accuracy of model.
    
    Input: 
    TargetVAriableRater as target 
    CrowdstormingDataJuly1st_Array as Feature set 
    
    Output:
    Features Sorted by their score with regard model accuracy
    '''
    for train_idx, test_idx in ShuffleSplit(ShuffleSplitNumber,RandomState):
        X_train, X_test = CrowdstormingDataJuly1st_Array[train_idx], CrowdstormingDataJuly1st_Array[test_idx]
        Y_train, Y_test = TargetVAriableRater[train_idx], TargetVAriableRater[test_idx]
        X_train[np.isnan(X_train)]=0.0
        X_test[np.isnan(X_test)]=0.0
        Feature_name = CrowdstormingDataJuly1st.columns # Get Column Headings of our Dataframe
        scores = defaultdict(list)

        Y_train = Y_train[:,0]
        Y_train = np.asarray(Y_train, dtype="|S6")
        Y_test = Y_test[:,0]
        type(Y_train)
        type(Y_test)
        Y_test=Y_test.astype(float)
        ForestClassifier = RandomForestClassifier(n_estimators=100)
        r = ForestClassifier.fit(X_train, Y_train)

        tester = ForestClassifier.predict(X_test)
        tester=tester.astype(np.float)

        acc = r2_score(Y_test, tester)

        for i in range(CrowdstormingDataJuly1st_Array.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            tester2 = ForestClassifier.predict(X_t)
            tester2=tester2.astype(np.float)
            shuff_acc = r2_score(Y_test,tester2)
            scores[Feature_name[i]].append((acc-shuff_acc)/acc)
    ModelFeatureAccuracyScore = (sorted([(round(np.mean(score), 4), Feature_name) for
                  Feature_name, score in scores.items()], reverse=True))
    return ModelFeatureAccuracyScore

class MultiColumnLabelEncoder(LabelEncoder):
    """
    Wraps sklearn LabelEncoder functionality for use on multiple columns of a
    pandas dataframe.

    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, dframe):
        """
        Fit label encoder to pandas columns.

        Access individual column classes via indexig `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            for idx, column in enumerate(self.columns):
                # fit LabelEncoder to get `classes_` for the column
                le = LabelEncoder()
                le.fit(dframe.loc[:, column].values)
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                # append this column's encoder
                self.all_encoders_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                le.fit(dframe.loc[:, column].values)
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
        return self

    def fit_transform(self, dframe):
        """
        Fit label encoder and return encoded labels.

        Access individual column classes via indexing
        `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`

        Access individual column encoded labels via indexing
        `self.all_labels_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            self.all_labels_ = np.ndarray(shape=self.columns.shape,
                                          dtype=object)
            for idx, column in enumerate(self.columns):
                # instantiate LabelEncoder
                le = LabelEncoder()
                # fit and transform labels in the column
                dframe.loc[:, column] =\
                    le.fit_transform(dframe.loc[:, column].values)
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
                self.all_labels_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                dframe.loc[:, column] = le.fit_transform(
                        dframe.loc[:, column].values)
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
        return dframe

    def transform(self, dframe):
        """
        Transform labels to normalized encoding.
        """
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[
                    idx].transform(dframe.loc[:, column].values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]\
                    .transform(dframe.loc[:, column].values)
        return dframe.loc[:, self.columns].values

    def inverse_transform(self, dframe):
        """
        Transform labels back to original encoding.
        """
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]\
                    .inverse_transform(dframe.loc[:, column].values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]\
                    .inverse_transform(dframe.loc[:, column].values)
        return dframe    
