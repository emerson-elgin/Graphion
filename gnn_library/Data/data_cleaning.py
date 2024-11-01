# gnn_library/utils/data_cleaning.py
class DataCleaner:
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# Implement pooling for gpu support
    def handle_missing_values(self, data, strategy="mean"):
        import numpy as np
        if strategy == "mean":
            return np.nan_to_num(data, nan=np.nanmean(data))
        elif strategy == "median":
            return np.nan_to_num(data, nan=np.nanmedian(data))
        else:
# document dropout layers. Makes the API more intuitive and consistent
            raise ValueError("Invalid strategy. Choose 'mean' or 'median'.")

    def scale_features(self, data):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(data)
