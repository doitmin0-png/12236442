import time
import numpy as np
import scipy.io as sio
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

from main import make_features_batch, MAT_PATH, MODEL_PATH, RANDOM_STATE


def mean_position_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def main():
    start = time.time()

    data = sio.loadmat(MAT_PATH, squeeze_me=False)
    BS_positions = np.asarray(data["BS_positions"], dtype=float)  # (2, 18)
    d_hat = np.asarray(data["d_hat"], dtype=float)                # (18, num_user)
    p = np.asarray(data["p"], dtype=float)                        # (2, num_user)

    X = make_features_batch(d_hat, BS_positions)
    y = p.T  # (num_user, 2)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Ridge": Ridge(alpha=15.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            )
        ),
    }

    print("Validation result")
    print("-----------------")
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        err = mean_position_error(y_val, pred)
        print(f"{name:16s}: mean position error = {err:.6f}")

    # Week8 결론에 맞춰 최종 제출 모델은 Gradient Boosting으로 저장
    final_model = models["GradientBoosting"]
    final_model.fit(X, y)

    save_data = {
        "model": final_model,
        "feature_dim": X.shape[1],
        "model_name": "GradientBoosting",
        "description": "RANSAC/WLS feature + Gradient Boosting correction",
    }

    joblib.dump(save_data, MODEL_PATH)

    elapsed = time.time() - start
    print("-----------------")
    print("Final saved model: GradientBoosting")
    print(f"Saved path       : {MODEL_PATH}")
    print(f"Feature shape    : {X.shape}")
    print(f"Elapsed time     : {elapsed:.2f} sec")


if __name__ == "__main__":
    main()
