from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class Transformador(BaseEstimator, TransformerMixin):
    def __init__( self, colunas_quantitativas, colunas_categoricas):
        self.colunas_quantitativas = colunas_quantitativas
        self.colunas_categoricas = colunas_categoricas
        self.enc = OneHotEncoder()
        self.scaler = MinMaxScaler()

    def fit(self, X, y = None ):
        self.enc.fit(X[self.colunas_categoricas])
        self.scaler.fit(X[self.colunas_quantitativas])
        return self 

    def transform(self, X, y = None):
      
      X_categoricas = pd.DataFrame(data=self.enc.transform(X[self.colunas_categoricas]).toarray(),
                                  columns= self.enc.get_feature_names(self.colunas_categoricas))
      
      X_continuas = pd.DataFrame(data=self.scaler.transform(X[self.colunas_quantitativas]),
                                  columns= self.colunas_quantitativas)
      
      X = pd.concat([X_continuas, X_categoricas], axis=1)

      return X