import numpy as np
import scipy.io as sio
import joblib

MODEL_PATH = "model.pkl"
MAT_PATH = "DH_FR1.mat"
RANDOM_STATE = 42
RANSAC_ITER = 60

_LOADED_MODEL = None
_EXPECTED_DIM = None


def _safe_dist(d):
    d = np.asarray(d, dtype=float).reshape(-1)
    finite = np.isfinite(d)
    fill = np.nanmedian(d[finite]) if np.any(finite) else 1.0
    d = np.nan_to_num(d, nan=fill, posinf=1e6, neginf=1.0)
    return np.maximum(d, 1e-6)


def _weighted_centroid(d, anchors, k=6):
    d = _safe_dist(d)
    anchors = np.asarray(anchors, dtype=float)
    k = min(k, d.size)
    idx = np.argsort(d)[:k]
    w = 1.0 / np.maximum(d[idx], 1e-6) ** 2
    return (anchors[:, idx] * w).sum(axis=1) / np.sum(w)


def _linear_ls_position(d, anchors, indices=None):
    d = _safe_dist(d)
    anchors = np.asarray(anchors, dtype=float)
    idx = np.arange(d.size) if indices is None else np.asarray(indices, dtype=int)

    if idx.size < 3:
        return _weighted_centroid(d, anchors)

    local_d = d[idx]
    ref = idx[int(np.argmin(local_d))]
    a_ref = anchors[:, ref]
    d_ref = d[ref]

    A = []
    b = []
    for i in idx:
        if i == ref:
            continue
        ai = anchors[:, i]
        A.append(2.0 * (a_ref - ai))
        b.append(d[i] ** 2 - d_ref ** 2 - np.dot(ai, ai) + np.dot(a_ref, a_ref))

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    try:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        if not np.all(np.isfinite(x)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        x = _weighted_centroid(d, anchors)

    return np.asarray(x, dtype=float).reshape(2)


def _robust_wls_refine(x0, d, anchors, inlier_mask=None, max_iter=20):
    d = _safe_dist(d)
    anchors = np.asarray(anchors, dtype=float)
    x = np.asarray(x0, dtype=float).reshape(2).copy()

    if inlier_mask is None or np.sum(inlier_mask) < 3:
        inlier_mask = np.ones(d.size, dtype=bool)
    else:
        inlier_mask = np.asarray(inlier_mask, dtype=bool)

    span = np.max(np.ptp(anchors, axis=1))
    margin = max(1.0, 0.25 * span)
    lower = anchors.min(axis=1) - margin
    upper = anchors.max(axis=1) + margin

    for _ in range(max_iter):
        diff = x[:, None] - anchors
        pred = np.linalg.norm(diff, axis=0)
        pred = np.maximum(pred, 1e-9)
        residual = pred - d

        med = np.median(residual[inlier_mask])
        mad = np.median(np.abs(residual[inlier_mask] - med)) + 1e-9
        sigma = 1.4826 * mad + 1e-9
        c = max(1e-6, 1.5 * sigma)

        huber = np.ones_like(residual)
        big = np.abs(residual) > c
        huber[big] = c / (np.abs(residual[big]) + 1e-9)

        w_dist = 1.0 / np.maximum(d, 1.0)
        w_inlier = np.where(inlier_mask, 1.0, 0.15)
        w = huber * w_dist * w_inlier

        J = (diff / pred).T
        sw = np.sqrt(np.maximum(w, 1e-12))
        A = J * sw[:, None]
        b = -residual * sw

        try:
            step, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(step)):
            break

        norm = np.linalg.norm(step)
        if norm > 10.0:
            step = step / norm * 10.0

        x = x + step
        x = np.minimum(np.maximum(x, lower), upper)

        if np.linalg.norm(step) < 1e-5:
            break

    return x


def _trimmed_cost(x, d, anchors, keep_ratio=0.7):
    d = _safe_dist(d)
    residual = np.abs(np.linalg.norm(x[:, None] - anchors, axis=0) - d)
    k = max(3, int(np.ceil(keep_ratio * residual.size)))
    return float(np.mean(np.sort(residual)[:k] ** 2))


def _ransac_wls_position(d, anchors, seed=0, n_iter=RANSAC_ITER):
    d = _safe_dist(d)
    anchors = np.asarray(anchors, dtype=float)
    m = d.size

    if m < 3:
        return _weighted_centroid(d, anchors), 1.0, np.zeros(m)

    rng = np.random.default_rng(seed)
    all_idx = np.arange(m)

    x_all = _linear_ls_position(d, anchors)
    res_all = np.abs(np.linalg.norm(x_all[:, None] - anchors, axis=0) - d)
    med = np.median(res_all)
    mad = np.median(np.abs(res_all - med)) + 1e-9
    threshold = max(1.0, med + 2.5 * 1.4826 * mad)

    best_x = x_all
    best_mask = np.ones(m, dtype=bool)
    best_score = _trimmed_cost(best_x, d, anchors)

    for _ in range(n_iter):
        sample = rng.choice(all_idx, size=3, replace=False)
        x_tmp = _linear_ls_position(d, anchors, sample)
        residual = np.abs(np.linalg.norm(x_tmp[:, None] - anchors, axis=0) - d)
        mask = residual <= threshold

        if np.sum(mask) < 3:
            k = max(3, int(np.ceil(0.65 * m)))
            mask = np.zeros(m, dtype=bool)
            mask[np.argsort(residual)[:k]] = True

        x_refined = _linear_ls_position(d, anchors, np.where(mask)[0])
        x_refined = _robust_wls_refine(x_refined, d, anchors, mask)
        cost = _trimmed_cost(x_refined, d, anchors)
        score = cost / (0.5 + np.mean(mask))

        if score < best_score:
            best_score = score
            best_x = x_refined
            best_mask = mask

    best_x = _robust_wls_refine(best_x, d, anchors, best_mask)
    final_residual = np.linalg.norm(best_x[:, None] - anchors, axis=0) - d
    inlier_ratio = float(np.mean(best_mask))

    return best_x, inlier_ratio, final_residual


def make_feature_one(d_u, p_bs, user_seed=0):
    """
    사용자 1명에 대한 ML 입력 feature 생성
    d_u: (18,)
    p_bs: (2, 18)
    return: (1, feature_dim)
    """
    d = _safe_dist(d_u)
    anchors = np.asarray(p_bs, dtype=float)

    raw = d.reshape(1, -1)
    sorted_d = np.sort(d).reshape(1, -1)

    stats = np.array([[
        np.mean(d),
        np.std(d),
        np.min(d),
        np.max(d),
        np.median(d),
        np.percentile(d, 25),
        np.percentile(d, 75),
    ]], dtype=float)

    wc_xy = _weighted_centroid(d, anchors, k=6).reshape(1, 2)
    rwls_xy, inlier_ratio, residual = _ransac_wls_position(
        d, anchors, seed=RANDOM_STATE + int(user_seed)
    )
    rwls_xy = rwls_xy.reshape(1, 2)
    inlier_ratio = np.array([[inlier_ratio]], dtype=float)
    residual = residual.reshape(1, -1)

    abs_res = np.abs(residual.reshape(-1))
    residual_stats = np.array([[
        np.mean(abs_res),
        np.std(abs_res),
        np.min(abs_res),
        np.max(abs_res),
        np.median(abs_res),
    ]], dtype=float)

    nearest_idx = np.argsort(d)[:5].astype(float).reshape(1, -1)

    X = np.hstack([
        raw,
        sorted_d,
        stats,
        wc_xy,
        rwls_xy,
        inlier_ratio,
        residual,
        residual_stats,
        nearest_idx,
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X


def your_algorithm(d_u, p_bs):
    """
    채점기 예시 형식에 맞춘 사용자 1명 위치 추정 함수
    d_u: d_hat[:, u], shape (18,)
    p_bs: BS_positions, shape (2, 18)
    return: 측위 결과, shape (2,)
    """
    global _LOADED_MODEL, _EXPECTED_DIM

    if _LOADED_MODEL is None:
        saved = joblib.load(MODEL_PATH)
        _LOADED_MODEL = saved["model"]
        _EXPECTED_DIM = saved.get("feature_dim", None)

    X = make_feature_one(d_u, p_bs)

    if _EXPECTED_DIM is not None and X.shape[1] != _EXPECTED_DIM:
        raise ValueError(
            f"Feature dimension mismatch: X has {X.shape[1]}, model expects {_EXPECTED_DIM}"
        )

    pred = _LOADED_MODEL.predict(X)  # (1, 2)
    return np.asarray(pred, dtype=float).reshape(2)


def make_features_batch(d_hat, BS_positions):
    """
    train.py에서 학습용 feature를 만들 때 사용하는 함수
    d_hat: (18, num_user)
    return: (num_user, feature_dim)
    """
    D = np.asarray(d_hat, dtype=float)
    if D.ndim == 1:
        D = D.reshape(-1, 1)

    num_user = D.shape[1]
    X_list = []
    for u in range(num_user):
        X_list.append(make_feature_one(D[:, u], BS_positions, user_seed=u))

    return np.vstack(X_list)


def main():
    # 1) 입력 데이터 로드 — 채점기가 같은 폴더에 .mat 파일 자동 배치
    mat_path = MAT_PATH

    data = sio.loadmat(mat_path, squeeze_me=False)
    BS_positions = np.asarray(data["BS_positions"], dtype=float)  # (2, 18)
    d_hat = np.asarray(data["d_hat"], dtype=float)                # (18, num_user)

    # p는 학습 정답값이므로 main.py에서는 사용하지 않음
    # hidden test에서는 p가 없을 수 있으므로 optional로만 처리
    p = np.asarray(data["p"], dtype=float) if "p" in data else None
    _ = p

    # 2) 본인 알고리즘 — 사용자 수는 입력에서 동적으로 받기
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))

    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    # 3) 결과 반환 — numpy 배열, 모양 (2, num_user)
    return p_hat


if __name__ == "__main__":
    main()
