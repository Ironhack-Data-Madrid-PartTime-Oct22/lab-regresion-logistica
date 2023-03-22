
import pandas as pd
import numpy as np
import sidetable
import pickle
#librerias limpieza de nulos
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

# librerías de visualización
import seaborn as sns
import matplotlib.pyplot as plt

#libreria para el balanceo
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# librerías para crear el modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder  

from sklearn import tree

# para calcular las métricas
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import cohen_kappa_score


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")






##gestion de nulos:

def limpiar_nulos_iterative_imputer(df):
    #este metodo solo funciona con variables numericas.
    numericas = df.select_dtypes(include = np.number)
    # creamos una instancia del método Iterative Imputer con las características que queremos 
    imputer = IterativeImputer(estimator=None, missing_values=np.nan,  max_iter=10, tol=0.001, n_nearest_features=None, initial_strategy= np.mean, verbose=0, random_state=None)
    # lo aplicamos sobre nuestras variables numéricas. 
    imputer.fit(numericas)
    # transformamos nuestros datos, para que se reemplacen los valores nulos usando "transform". 
    ## ⚠️ Esto nos va a devolver un array!
    imputer.transform(numericas)
    # convertimos el array que nos devuelve en un dataframe
    #ademas se le cambia el nombre a la columna, ya que el nuvo df viene sin los nombres
    #es importante hacer un reset index, porque al trabajar con los vecinos, si no estan ordenados generara mas nulos.
    numericas_trans = pd.DataFrame(imputer.transform(numericas), columns = numericas.columns)
    # creamos la lista de columnas nueva
    columnas = numericas_trans.columns
    #creamos un df nuevo para no alterar el original
    df2 = df.copy()
    # le quitamos las columnas con los valores originales
    df2.drop(columnas, axis = 1, inplace = True)
    #asignamos las columnas nuevas
    df2[columnas] = numericas_trans
    return df2
#%%
def tipo_a_float(columna,df):
    df[columna] = df[columna].astype(np.number)
    return print(df[columna].dtypes)

# %%
def ordinal_encoder(orden, df, columna):
    ordinal = OrdinalEncoder(categories = [orden], dtype = int)
    transformados_oe = ordinal.fit_transform(df[[columna]])
    df[columna] = transformados_oe
    
    with open(f'datos/encoding{columna}.pkl', 'wb') as s:
        pickle.dump(ordinal, s)
    return df

#%%
def one_hot_encoder(dff, columnas):
    
    oh = OneHotEncoder()
    
    transformados = oh.fit_transform(dff[columnas])
    
    oh_df = pd.DataFrame(transformados.toarray(), columns = oh.get_feature_names_out(), dtype = int)
    
    dff[oh_df.columns] = oh_df
    
    dff.drop(columnas, axis = 1, inplace = True)
    
    with open(f'datos/encoding{columnas[0]}.pkl', 'wb') as s:
        pickle.dump(oh, s)
    
    return dff

#%%
def balanceo(dataframe, variable_respuesta, input):
    X = dataframe.drop([variable_respuesta], axis = 1)
    y = dataframe[[variable_respuesta]]
    if input == 'downsampling':
        modelo = RandomUnderSampler()
    elif input == 'upsampling':
        modelo = RandomOverSampler()
    else:
        print("aprende a escribir")
    X_mod, y_mod = modelo.fit_resample(X,y)
    return X_mod, y_mod