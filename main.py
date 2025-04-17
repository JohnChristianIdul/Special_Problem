import joblib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader
from informer_model import Informer
from mtcn_model import train_and_predict, TimeSeriesDataset


csv_path = "E:\School\\4th year - 1st sem\Thesis\Model\processed_data.csv"
csv_path_one = "E:\School\\4th year - 1st sem\Thesis\Model\processed_data_1.csv"


def preprocess_data_feature(file_path, target_column='wl-a', save_csv=True):
    rolling_window = 3
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')
    df.sort_values('Datetime', inplace=True)
    df.set_index('Datetime', inplace=True)

    # Define categorical and numerical columns for specific treatment
    numerical_cols = ["rf-a", "rf-a-sum", "wl-ch-a"]

    # convert all (*) to nan
    df = df.replace(r'\(\*\)', np.nan, regex=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Handle missing values globally using linear interpolation, then forward and backward filling
    df.infer_objects(copy=False)
    df.interpolate(method='linear', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Perform Min-Max scaling on all numerical columns
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    all_feature_cols = numerical_cols

    # ColumnTransformer setup for potential pipeline integration (optional here)
    transformers = [
        ('num', MinMaxScaler(), numerical_cols),
    ]
    preprocessor = ColumnTransformer(transformers, remainder='passthrough')

    # Extract features and apply transformations if needed
    remaining_features = df.drop(columns=[target_column], errors='ignore')
    prepared_features = preprocessor.fit_transform(remaining_features)
    feature_names = all_feature_cols

    # Prepare the final structure of the data
    date_series = pd.Series(df.index, index=df.index)

    # Create the structured dataset
    final_df = pd.DataFrame(prepared_features, columns=feature_names, index=df.index)
    final_df[target_column] = df[target_column] if target_column in df.columns else None

    return {
        'features': prepared_features,
        'feature_names': feature_names,
        'date': date_series,
        'target': df[target_column] if target_column in df.columns else None,
        'scaler': preprocessor
    }


def perform_feature_selection(file_path, selection_method, selection_threshold):
    data_config = preprocess_data_feature(file_path, target_column='wl-a')
    features = data_config['features']
    feature_names = data_config['feature_names']
    print("feature names: ", feature_names, "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(features, dtype=torch.float32).to(device)

    model = Informer(enc_in=x.shape[1], selection_method=selection_method, selection_threshold=selection_threshold).to(device)

    with torch.no_grad():
        importance_scores = model.compute_feature_importance(x)

    selected_indices = [i for i, score in enumerate(importance_scores) if score.item() >= selection_threshold]
    selected_feature_names = [feature_names[i] for i in selected_indices]
    selected_feature_data = features[:, selected_indices]

    date_series = pd.Series(data_config['date']).reset_index(drop=True)
    target_series = pd.Series(data_config['target']).reset_index(drop=True)
    features_df = pd.DataFrame(selected_feature_data, columns=selected_feature_names).reset_index(drop=True)

    print("Importance Scores: ", importance_scores)

    final_df = pd.concat([date_series, features_df, target_series], axis=1)

    # Save as csv to check output
    # final_df.to_csv(csv_path_one)
    return {
        'selected_features': selected_feature_names,
        'full_dataframe': final_df,
        'date': data_config['date'],
        'target': data_config['target'],
        'scaler': data_config['scaler']
    }


def time_temporal_features_extraction(df):
    # Extracting temporal features directly
    df['day_of_week'] = df['Datetime'].dt.dayofweek
    df['week'] = df['Datetime'].dt.isocalendar().week
    df['month'] = df['Datetime'].dt.month
    df['year'] = df['Datetime'].dt.year

    return df


def rolling_features(df, rolling_windows, lags):
    features = ["rf-a", "rf-a-sum", "wl-ch-a"]

    for feature in features:
        if feature in df.columns:
            for stat, window_sizes in rolling_windows.items():
                for window_size in window_sizes:
                    if stat == 'mean':
                        df[f'{feature}_{window_size}0min_avg'] = df[feature].rolling(window=window_size,
                                                                                    min_periods=1).mean()
                    elif stat == 'max':
                        df[f'{feature}_{window_size}0min_max'] = df[feature].rolling(window=window_size,
                                                                                    min_periods=1).max()
                    elif stat == 'var':
                        df[f'{feature}_{window_size}0min_var'] = df[feature].rolling(window=window_size,
                                                                                    min_periods=1).var()

    # Add lagged features
    for feature in features:
        if feature in df.columns:
            for lag in lags:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

    df.bfill(inplace=True)

    return df


def main():
    file_path = "PAGASA/A/A-cummulative.csv"
    selection_results = perform_feature_selection(file_path, 'importance', 0.2)
    training_df = selection_results['full_dataframe']

    training_df = time_temporal_features_extraction(training_df)

    # 10-minute data interval, adjust as necessary
    rolling_windows = {
        'mean': [6, 12],
        'max': [6, 12],
        'var': [6, 12]
    }
    lags = [12, 24]
    training_df = rolling_features(training_df, rolling_windows, lags)

    # Updated feature names list
    selected_feature_names = selection_results['selected_features'] + ['day_of_week', 'week', 'month', 'year']

    # Define new rolling features correctly
    new_features_1hr = [
        f'{feat}_60min_avg' for feat in ["rf-a", "rf-a-sum", "wl-ch-a"]
        if f'{feat}_60min_avg' in training_df.columns
    ]

    new_features_2hr = [
        f'{feat}_120min_avg' for feat in ["rf-a", "rf-a-sum", "wl-ch-a"]
        if f'{feat}_120min_avg' in training_df.columns
    ]

    lagged_features = [f'{feat}_lag_{lag}' for feat in ["rf-a", "rf-a-sum", "wl-ch-a"] for lag in lags if
                       f'{feat}_lag_{lag}' in training_df.columns]

    # Extend the original list by the new calculated features along with lagged features
    selected_feature_names.extend(new_features_1hr + new_features_2hr + lagged_features)

    training_df.drop(columns=['Datetime'], inplace=True)

    print("Features Selected: ", selected_feature_names)

    features = training_df[selected_feature_names].values
    targets = training_df['wl-a'].values
    features = features.astype(np.float32)

    features_tensor = torch.FloatTensor(features)
    targets_tensor = torch.FloatTensor(targets)

    sequence_length = 6
    num_epochs = 50
    batch_size = 24
    trainer, history = train_and_predict(features=features_tensor, targets=targets_tensor, sequence_length=sequence_length, num_epochs=num_epochs, batch_size=batch_size)

    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    model = trainer.model
    model.eval()
    predictions = []
    actual_values = []
    test_dataset = TimeSeriesDataset(features, targets, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(trainer.device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy().flatten())
            actual_values.extend(batch_y.numpy().flatten())

    predictions = np.array(predictions)
    actual_values = np.array(actual_values)

    # Plot predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label='Actual')
    plt.plot(predictions, label='Predicted', linestyle='--')
    plt.title('Actual vs Predicted wl-a Levels')
    plt.xlabel('Time Steps')
    plt.ylabel('wl-a Level')
    plt.legend()
    plt.show()

    # Print some sample predictions
    print("\nSample Predictions:")
    for i in range(10):
        print(f"Actual: {actual_values[i]:.4f}, Predicted: {predictions[i]:.4f}")

    model_path = "E:\\School\\4th year - 1st sem\\Thesis\\Model\\trained_model\\wl_a_model_ver_1.2_0.1_dropout.pth"
    scaler_path = "E:\\School\\4th year - 1st sem\\Thesis\\Model\\trained_model\\scalers_ver_1.2_0.1_dropout.joblib"

    torch.save(trainer.model.state_dict(), model_path)
    joblib.dump(selection_results['scaler'], scaler_path)

    print("Model and scaler saved successfully!")


if __name__ == "__main__":
    main()
