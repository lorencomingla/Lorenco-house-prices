# House Price Prediction Model
# Developed by Lorenco Mingla - February 20, 2025
# Advanced ML project using Kaggle House Prices dataset (train.csv)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import umap
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
import tensorflow as tf
import optuna
import shap
import matplotlib.pyplot as plt
import joblib
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
print("Initializing house price prediction model...")

# Load dataset
data = pd.read_csv('train.csv')

# Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['QualitySF'] = df['OverallQual'] * df['TotalSF']
    df['GarageScore'] = df['GarageArea'] * df['GarageCars']
    df['NeighborhoodQuality'] = df.groupby('Neighborhood')['OverallQual'].transform('mean')
    return df

data = engineer_features(data)

# Define feature sets
numeric_features = ['TotalSF', 'HouseAge', 'QualitySF', 'GarageScore', 'NeighborhoodQuality', 'GrLivArea', 'OverallQual']
categorical_features = ['Neighborhood', 'ExterQual']

# Preprocessing pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Prepare data
X = data[numeric_features + categorical_features]
y = np.log1p(data['SalePrice'])
X_processed = preprocessor.fit_transform(X)

# Dimensionality reduction with UMAP
umap_reducer = umap.UMAP(n_components=6, random_state=42, n_neighbors=5)
X_final = umap_reducer.fit_transform(X_processed)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Custom Transformer layer
class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.feed_forward = Dense(d_model, activation='gelu')
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn_output = self.attention(inputs, inputs)
        x = self.norm(inputs + self.dropout(attn_output, training=training))
        ff_output = self.feed_forward(x)
        return self.norm(x + ff_output)

# Bayesian Transformer model
def build_model(input_dim: int, d_model: int, num_heads: int) -> Model:
    inputs = Input(shape=(input_dim,))
    x = Dense(d_model, activation='gelu')(inputs)
    x = tf.expand_dims(x, axis=1)
    x = TransformerLayer(d_model, num_heads)(x)
    x = tf.squeeze(x, axis=1)
    x = Dropout(0.1)(x)
    
    # Compute the distribution parameters
    loc = Dense(1)(x)
    scale = tf.nn.softplus(Dense(1)(x) + 0.01)
    
    # Wrap the distribution in a DistributionLambda layer
    dist = tfpl.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Normal(loc=t[0], scale=t[1])
    )([loc, scale])
    
    model = Model(inputs, dist)
    
    def neg_log_likelihood(y_true: tf.Tensor, y_pred: tfd.Distribution) -> tf.Tensor:
        return -tf.reduce_mean(y_pred.log_prob(y_true))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=neg_log_likelihood)
    return model

# Hyperparameter tuning with Optuna
def objective(trial: optuna.Trial) -> float:
    d_model = trial.suggest_int('d_model', 64, 128)
    num_heads = trial.suggest_int('num_heads', 2, 4)
    epochs = trial.suggest_int('epochs', 30, 100)
    model = build_model(X_train.shape[1], d_model, num_heads)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    # Get mean predictions from the distribution
    dist = model(X_test)
    predictions = dist.mean().numpy().flatten()
    # Ensure y_test is in numpy format
    y_test_np = y_test.numpy() if isinstance(y_test, tf.Tensor) else np.array(y_test)
    return np.sqrt(mean_squared_error(y_test_np, predictions))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
print(f"Optimal hyperparameters: {study.best_params}")

# Train final model
final_model = build_model(X_train.shape[1], study.best_params['d_model'], study.best_params['num_heads'])
final_model.fit(X_train, y_train, epochs=study.best_params['epochs'], batch_size=32, verbose=1)
y_pred_dist = final_model(X_test)
y_pred = y_pred_dist.mean().numpy().flatten()

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred)) / np.expm1(y_test))) * 100
print(f"RMSE (log scale): {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# SHAP analysis
print("Generating SHAP explainability...")
explainer = shap.KernelExplainer(lambda x: final_model(x).mean().numpy(), shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_test[:100])
shap.summary_plot(shap_values, X_test[:100], show=False)
plt.savefig('shap_analysis.png', bbox_inches='tight')
plt.close()

# Save outputs
results = pd.DataFrame({'Actual': np.expm1(y_test), 'Predicted': np.expm1(y_pred)})
results.to_csv('predictions.csv', index=False)
final_model.save('house_price_model.h5')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(umap_reducer, 'umap_reducer.pkl')
print("Model, preprocessor, and results saved successfully.")
