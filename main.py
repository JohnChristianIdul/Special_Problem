import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader
from darts import TimeSeries
import matplotlib.pyplot as plt

from data_loader import TimeSeriesLoader
from informer_model import Informer
from mtcn_model import (
    TimeSeriesDataset,
    train_and_predict
)


def preprocess_data_feature(file_path, target_column):
    df = pd.read_csv(file_path)

    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns found: {df.columns[df.columns.duplicated()]}")

    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df.dropna(inplace=True)

    target_series = df[target_column]
    feature_data = df.drop(columns=[target_column])

    # Add datetime feature as hours since the start of the dataset
    start_date = df['Date Time'].min()
    df['datetime_feature'] = (df['Date Time'] - start_date).dt.total_seconds() / 3600
    datetime_feature = df['datetime_feature'].values.reshape(-1, 1)

    # Drop the original 'Date Time' column to avoid non-numerical data
    feature_data = feature_data.drop(columns=['Date Time'])

    numerical_cols = feature_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = feature_data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Initialize transformers list
    transformers = [('num', MinMaxScaler(), numerical_cols)]

    # Only add categorical transformer if there are categorical columns
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols))

    preprocessor = ColumnTransformer(transformers)

    # Fit and transform the features
    transformed_features = preprocessor.fit_transform(feature_data)

    # Convert sparse matrix to dense if necessary
    if scipy.sparse.issparse(transformed_features):
        transformed_features = transformed_features.toarray()

    # Append datetime feature to the processed features
    features_with_datetime = np.hstack((transformed_features, datetime_feature))

    # Prepare feature names
    feature_names = numerical_cols
    if categorical_cols:
        try:
            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_feature_names)
        except (KeyError, AttributeError):
            pass
    feature_names.append('datetime_feature')

    return {
        'features': features_with_datetime,
        'feature_names': feature_names,
        'date': df['Date Time'],
        'target': target_series,
        'scalers': {
            'numeric': preprocessor.named_transformers_['num'],
            'categorical': preprocessor.named_transformers_['cat'] if categorical_cols else None
        }
    }


def perform_feature_selection(file_path, selection_method, selection_threshold):
    data_config = preprocess_data_feature(file_path, target_column='T (degC)')

    features = data_config['features']
    feature_names = data_config['feature_names']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(features, dtype=torch.float32).to(device)

    model = Informer(
        enc_in=x.shape[1],
        selection_method=selection_method,
        selection_threshold=selection_threshold
    ).to(device)

    with torch.no_grad():
        importance_scores = model.compute_feature_importance(x)

    max_importance = importance_scores.max().item()
    selected_indices = [i for i, score in enumerate(importance_scores) if score.item() == max_importance]
    selected_feature_names = [feature_names[i] for i in selected_indices]
    selected_feature_data = features[:, selected_indices]

    final_df = pd.concat([
        data_config['date'],
        pd.DataFrame(selected_feature_data, columns=selected_feature_names),
        data_config['target']
    ], axis=1)

    return {
        'selected_features': selected_feature_names,
        'full_dataframe': final_df,
        'date': data_config['date'],
        'target': data_config['target']
    }


def collate_fn(batch):
    inputs = [item['inputs'] for item in batch]
    labels = [item['labels'] for item in batch]

    max_length = max(label.size(0) for label in labels)
    labels_padded = [torch.nn.functional.pad(label, (0, max_length - label.size(0))) for label in labels]

    inputs_padded = torch.stack(inputs)
    labels_padded = torch.stack(labels_padded)

    return {'inputs': inputs_padded, 'labels': labels_padded}


def main():
    file_path = "jena_climate_2009_2016.csv/jena_climate_2009_2016.csv"

    # Get selected features and preprocessed data
    selection_results = perform_feature_selection(file_path, 'importance', 0.5)
    print(f"Features selected for training: {selection_results['selected_features']}")
    training_df = selection_results['full_dataframe']
    training_df.ffill(inplace=True)

    # Handle any NaT (Not a Time) values resulting from coercion
    training_df.dropna(subset=['Date Time'], inplace=True)
    training_df.drop_duplicates(subset=['Date Time'], inplace=True)

    # Prepare features and target
    features = training_df[selection_results['selected_features']].values
    targets = training_df['T (degC)'].values

    # Ensure all features are numerical
    features = features.astype(np.float32)

    # Convert to PyTorch tensors
    features_tensor = torch.FloatTensor(features)
    targets_tensor = torch.FloatTensor(targets)

    # Training parameters
    sequence_length = 30
    num_epochs = 50  # 100
    batch_size = 250

    print("Starting training...")
    trainer, history = train_and_predict(
        features=features_tensor,
        targets=targets_tensor,
        sequence_length=sequence_length,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Make predictions
    model = trainer.model
    model.eval()
    predictions = []
    actual_values = []

    # Create dataset for testing
    test_dataset = TimeSeriesDataset(features, targets, sequence_length)
    print(f"-----\nfeatures: {features}\ntargets: {targets}\nsequence_length: {sequence_length}\n-----")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(trainer.device)
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy().flatten())
            actual_values.append(batch_y.numpy().flatten())

    predictions = np.concatenate(predictions)
    actual_values = np.concatenate(actual_values)

    # Plot predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label='Actual')
    plt.plot(predictions, label='Predicted', linestyle='--')
    plt.title('Actual vs Predicted T (degC)')
    plt.xlabel('Time Steps')
    plt.ylabel('T (degC)')
    plt.legend()
    plt.show()

    # Print some sample predictions
    print("\nSample Predictions:")
    for i in range(10):
        print(f"Actual: {actual_values[i]:.4f}, Predicted: {predictions[i]:.4f}")


if __name__ == "__main__":
    main()
