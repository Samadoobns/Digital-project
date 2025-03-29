import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from keras import callbacks
from tqdm import tqdm 
# Désactivation de certaines options de TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
# Chargement des données
X_full = pd.read_excel(r"C:\Users\samad\OneDrive\Bureau\ml\electrical_motors\data_pn-fm-machines.xlsx")

# Suppression des valeurs manquantes dans la colonne cible
X_full.dropna(axis=0, subset=['CEM'], inplace=True)

# Séparation des features (X) et de la variable cible (y)
y = X_full['CEM']
X_full.drop(['CEM', 'CEM_Numerique_vrai', 'Erreur', 'Erreur en %', 'CEM_Numerique_faux'], axis=1, inplace=True)

# Division des données en ensemble d'entraînement et de validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_full, y, train_size=0.8, test_size=0.2, random_state=0
)

print("train set dim:", X_train.shape)
print("full dim:", X_full.shape)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Définition du modèle
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # attendre 20 epochs pour arreter si y a pas de progression sup à min_delta
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # Correctif : Utilisation de Input()
    layers.Dense(1)
])
model1 = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'), 
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])
model2 = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'), 
    layers.Dense(512, activation='relu'), 
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),       
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])
model3 = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1),
])
# Compilation du modèle
model.compile(optimizer='adam', loss='mae')
model1.compile(optimizer='adam', loss='mae')
model2.compile(optimizer='adam', loss='mae')
model3.compile(optimizer='adam', loss='mae')

# Classe TQDMProgressBar pour afficher la barre de progression
class TQDMProgressBar(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.params['epochs'] = self.params['epochs']  # Assure que le nombre d'epochs est bien pris en compte
        pbar.set_postfix(logs)
        pbar.update(1)
# Entraînement du modèle avec barre de progression
histories = []
models = [model, model1, model2, model3]
for model in models:
    with tqdm(total=5000, desc=f"Training {model.name}", unit="epoch") as pbar:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            batch_size=512,
            epochs=5000,
            callbacks=[early_stopping, TQDMProgressBar()],
            verbose=0
        )
        histories.append(history)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Liste des historiques et des sous-graphes associés
titles = ["Model 0 (Linear)", "Model 1 (3 layers)", "Model 2 (5 layers)", "Model 3 (BatchNorm)"]

# Boucle pour tracer les courbes d'apprentissage sur chaque sous-graphe
for ax, hist, title in zip(axes.flatten(), histories, titles):
    history_df = pd.DataFrame(hist.history)
    history_df.loc[:, ['loss', 'val_loss']].plot(ax=ax)
    ax.set_title(f"{title}\nMin Val Loss: {history_df['val_loss'].min():.4f}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(["Train Loss", "Val Loss"])
plt.tight_layout()
plt.show()  # Ajout pour afficher correctement le graphe
