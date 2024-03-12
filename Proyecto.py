# LIBRERIAS 

import pandas as pd
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import plotly.express as px
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import warnings

from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# CARGUE DE DATOS
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# carpeta='C:/Users/Intevo/Desktop/UD/'
carpeta='C:/Users/santi/OneDrive/Documentos/INTEVO/U-Distrital/'
df=pd.read_csv(carpeta+'1_Base_de_datos_ORIGINAL estudiantes_2008_2019.txt',sep="|")
pd.options.display.max_columns=None
print(df)
columnas=['estrato','genero','biologia','quimica','fisica','sociales','aptitud_verbal',
          'espanol_literatura','aptitud_matematica','condicion_matematica',
          'filosofia','historia','geografia','idioma','puntos_icfes','puntos_homologados',
          'anno_nota','semestre_nota','nota','promedio']
df2=df[columnas] 
df2
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# PREPROCESAMIENTO DE DATOS
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

columns_to_dummy = df2.select_dtypes(include=['object']).columns
columns_to_dummy
df_w_dummy = pd.get_dummies(df2, columns=columns_to_dummy, prefix=columns_to_dummy,dtype=int)
df_w_dummy.head()
df2=df_w_dummy.dropna()
tamaño=50000
df2=df2.sample(n=tamaño,random_state=42)
df2.shape
df2 = df2.rename(columns={'genero_NO REGISTRA': 'genero_NO_REGISTRA'})
#df2['aprobacion']=np.where((df2['nota']>30),'1','0')
X = df2.drop("nota",axis=1)
y = df2.nota
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=103)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# MODELO DE ÁRBOL DE DECISIÓN
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
tree = DecisionTreeRegressor()
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [10, 20, 30]
}
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, scoring='r2', cv=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Mejores hiperparámetros:", best_params)
best_tree = DecisionTreeRegressor(max_depth=best_params['max_depth'],
                                  min_samples_leaf=best_params['min_samples_leaf'])
best_tree.fit(X_train, y_train)
y_train_pred = best_tree.predict(X_train)
y_test_pred = best_tree.predict(X_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print("R^2 en entrenamiento:", r2_train)
print("R^2 en prueba:", r2_test)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# MODELO GRADIENTBOOSTING
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
gbr = GradientBoostingRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, scoring='r2', cv=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Mejores hiperparámetros:", best_params)
best_gbr = GradientBoostingRegressor(n_estimators=best_params['n_estimators'],
                                     learning_rate=best_params['learning_rate'],
                                     max_depth=best_params['max_depth'])
best_gbr.fit(X_train, y_train)
y_train_pred = best_gbr.predict(X_train)
y_test_pred = best_gbr.predict(X_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print("R^2 en entrenamiento:", r2_train)
print("R^2 en prueba:", r2_test)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# MODELO XGBR 
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
xgb = XGBRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='r2', cv=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Mejores hiperparámetros:", best_params)
best_xgb = XGBRegressor(n_estimators=best_params['n_estimators'],
                        learning_rate=best_params['learning_rate'],
                        max_depth=best_params['max_depth'])
best_xgb.fit(X_train, y_train)
y_train_pred = best_xgb.predict(X_train)
y_test_pred = best_xgb.predict(X_test)
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# MODELO VOTING 
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
regressors = [('Gradient', best_gbr), ('XGB', best_xgb), ('Tree', best_tree)]
vr = VotingRegressor(estimators = regressors, n_jobs = -1, verbose = 1, weights = ( 0.2, 0.5, 0.3))
vr.fit(X_train, y_train.ravel())
#Prediciendo valores de entrenamiento
y_train_hat = vr.predict(X_train)
#Prediciendo valores de validación
y_test_hat = vr.predict(X_test)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# SERIALIZACIÓN DE MODELOS 
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

with open("DecisionTreeRegressor.pkl", "wb") as file:
    pickle.dump(best_tree, file)

    
with open("GradientRegressor.pkl", "wb") as file:
    pickle.dump(best_gbr, file)
    
with open("XgbootsRegressor.pkl", "wb") as file:
    pickle.dump(best_xgb, file)
    
with open("VotingRegressor.pkl", "wb") as file:
    pickle.dump(vr, file)
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# BASEMODEL PARA DESPLIEGUE
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from pydantic import BaseModel
class df2_model(BaseModel):
    estrato: float
    biologia: float
    quimica: float
    fisica: float
    sociales : float 
    aptitud_verbal : float 
    espanol_literatura : float 
    aptitud_matematica : float 
    condicion_matematica : float 
    filosofia : float 
    historia : float 
    geografia : float 
    idioma : float 
    puntos_icfes : float 
    puntos_homologados : float 
    anno_nota : float 
    semestre_nota : float 
    promedio : float 
    genero_FEMENINO : float 
    genero_MASCULINO : float 
    genero_NO_REGISTRA : float


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# ENDPOINTS MODELOS
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


# Endpoint para el modelo de tree
@app.post("/predict_tree")
def predict_tree(data: df2_model):
    try:
        input_features = [data.estrato, data.biologia, data.quimica, data.fisica, data.sociales, data.aptitud_verbal,
                          data.espanol_literatura, data.aptitud_matematica, data.condicion_matematica, data.filosofia,
                          data.historia, data.geografia, data.idioma, data.puntos_icfes,data.puntos_homologados,
                          data.anno_nota, data.semestre_nota, data.promedio, data.genero_MASCULINO, data.genero_FEMENINO,
                          data.genero_NO_REGISTRA]
        prediction = int(best_tree.predict([input_features])[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para el modelo de gradient
@app.post("/predict_gradient")
def predict_gradient(data: df2_model):
    try:
        input_features = [data.estrato, data.biologia, data.quimica, data.fisica, data.sociales, data.aptitud_verbal,
                          data.espanol_literatura, data.aptitud_matematica, data.condicion_matematica, data.filosofia,
                          data.historia, data.geografia, data.idioma, data.puntos_icfes,data.puntos_homologados,
                          data.anno_nota, data.semestre_nota, data.promedio, data.genero_MASCULINO, data.genero_FEMENINO,
                          data.genero_NO_REGISTRA]
        prediction = int(best_gbr.predict([input_features])[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint para el modelo de xgb
@app.post("/predict_xgboost")
def predict_xboost(data: df2_model):
    try:
        input_features = [data.estrato, data.biologia, data.quimica, data.fisica, data.sociales, data.aptitud_verbal,
                          data.espanol_literatura, data.aptitud_matematica, data.condicion_matematica, data.filosofia,
                          data.historia, data.geografia, data.idioma, data.puntos_icfes,data.puntos_homologados,
                          data.anno_nota, data.semestre_nota, data.promedio, data.genero_MASCULINO, data.genero_FEMENINO,
                          data.genero_NO_REGISTRA]
        prediction = int(best_xgb.predict([input_features])[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
# Endpoint para el modelo de voting
@app.post("/predict_voting")
def predict_voting(data: df2_model):
    try:
        input_features = [data.estrato, data.biologia, data.quimica, data.fisica, data.sociales, data.aptitud_verbal,
                          data.espanol_literatura, data.aptitud_matematica, data.condicion_matematica, data.filosofia,
                          data.historia, data.geografia, data.idioma, data.puntos_icfes,data.puntos_homologados,
                          data.anno_nota, data.semestre_nota, data.promedio, data.genero_MASCULINO, data.genero_FEMENINO,
                          data.genero_NO_REGISTRA]
        prediction = int(vr.predict([input_features])[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   
    
    
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# POST MODELOS
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "API para predecir calificación de estudiantes"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
