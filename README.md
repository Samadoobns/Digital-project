# ⚙️ Electric Motor Torque Prediction with Deep Learning

This repository implements machine / deep learning models to predict the torque (CEM) of small electric motors based on various physical and electrical parameters.

## 🧠 Key Features

- **Data Preprocessing**: Handles missing values, splits data into training and validation sets, and scales the features for model compatibility.
- **Multiple Regression Models**:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Neural Network models with varying architectures:
    - Simple linear model
    - Multi-layer feedforward models
    - Batch normalization-based models for improved performance.
- **Model Evaluation**: Compares the models' performances using R² score, visualizes learning curves (training vs. validation loss), and applies early stopping to prevent overfitting.
- **Hyperparameter Tuning**: Model architectures and training parameters (such as epochs, batch size, and optimizer) are fine-tuned for optimal performance.

## ⚡ Technologies Used

- **Libraries**: 
  - `pandas` (data manipulation)
  - `scikit-learn` (machine learning models and preprocessing)
  - `matplotlib` (visualization)
  - `keras` and `tensorflow` (deep learning)
  - `tqdm` (progress bar during training)

- **Machine Learning**: Regression models (Random Forest, Gradient Boosting, Neural Networks)
- **Deep Learning**: Keras-based deep neural networks with various architectures, early stopping, and batch normalization techniques.

## 📊 Training Setup

- **Training-Validation Split**: Data is split into training (80%) and validation (20%) sets.
- **Feature Scaling**: Features are normalized using `StandardScaler` to improve model performance.
- **Training Process**: Models are trained over several epochs (max 5000), and early stopping is applied to prevent overfitting. The training progress is displayed using a tqdm progress bar.

## 📈 Performance

- **Learning Curves**: Training and validation losses are plotted for each model to visualize the model's performance over epochs.
- **Model Evaluation**: Models are evaluated using the Mean Absolute Error (MAE) loss function. The best-performing models are selected based on the lowest validation loss.

## 📁 Data

The data used for training comes from the file `data_pn-fm-machines.xlsx`. It contains various features related to the motors and the target variable, CEM (torque). 

**Important Columns**:
- Features: Various electrical and mechanical parameters of the motor.
- Target: `CEM` (torque).

## 🔧 How to Run

1. Clone this repository:
    ```bash
    git clone <repository-url>
    ```

2. Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script:
    ```bash
    python train_model.py
    ```

4. The results will be saved in the `output` directory. You can find the model training curves and evaluation results there.

## 💡 Future Improvements

- Hyperparameter tuning using grid search or random search.
- Adding more complex architectures or fine-tuning existing models.
- Exploring other regression models like XGBoost or neural networks with advanced techniques such as transfer learning.

---

Feel free to modify the content of this `README.md` to suit your specific project needs.
