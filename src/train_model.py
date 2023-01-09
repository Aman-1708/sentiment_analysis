from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


class TrainModel:
    def __init__(self):
        self.clf = None

    def train_model(self, X_train, y_train, X_test, y_test):

        self.clf = SVC(kernel='linear', probability=True)
        print('Fitting SVM Classifier')
        self.clf.fit(X_train, y_train)
        print('Complete!..')

        print('Predicting Train')
        y_train_pred = self.clf.predict(X_train)
        y_train_pred_proba = self.clf.predict_proba(X_train)

        print('Predicting Test')
        y_test_pred = self.clf.predict(X_test)
        y_test_pred_proba = self.clf.predict_proba(X_test)

        print("Evaluation on Train..")
        conf_m_train = confusion_matrix(y_train, y_train_pred)
        report_train = classification_report(y_train, y_train_pred)
        auc_train = roc_auc_score(y_train, y_train_pred_proba[:,1], multi_class='ovr', average='macro')

        print("Confusion Matrix - \n", conf_m_train)
        print("Classification report - \n", report_train)
        print("AUC score - \n", auc_train)

        print("Evaluation on Test..")
        conf_m_test = confusion_matrix(y_test, y_test_pred)
        report_test = classification_report(y_test, y_test_pred)
        auc_test = roc_auc_score(y_test, y_test_pred_proba[:,1], multi_class='ovr', average='macro')

        print("Confusion Matrix - \n", conf_m_test)
        print("Classification report - \n", report_test)
        print("AUC score - \n", auc_test)
        return
