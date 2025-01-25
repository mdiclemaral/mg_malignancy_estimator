import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


class XG:
    def test_xgboost_model(self, model, df):
        df.dropna(inplace=True)

        X_test = df.drop(columns=['sample_id'])  # ??
        # print(X_test.columns)
        # print(len(X_test.columns))

        y_pred = model.predict(X_test.to_numpy())
        y_proba = model.predict_proba(X_test.to_numpy())
        
        return X_test, y_pred, y_proba[:, 1]

    def save_predictions(self, df, X_test, y_pred, y_proba):
        # Map the test indices back to the external_ids
        test_indices = X_test.index
        results = df.loc[test_indices, ['sample_id']].copy()
        results['prediction'] = y_pred
        results['probability'] = y_proba

        results = results.set_index('sample_id').T.to_dict('list')
        return results

    def test_model(self,idd, model, dataset):
        X_test, y_pred, y_proba = self.test_xgboost_model(model, dataset)
        results = self.save_predictions(dataset, X_test, y_pred, y_proba)
        # print("AAAA:", y_proba)
        # print("")
        return results, y_proba
        