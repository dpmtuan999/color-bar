"""
Phân loại Chính / Phụ / Nhấn dựa trên MẬT ĐỘ vị trí trên bánh xe màu sắc
(Đã sửa đổi toàn diện lỗi nhận diện màu chuyển tiếp, lỗi sập ma trận và lỗi nhóm trống)

Yêu cầu hệ thống: numpy, scipy
"""

import numpy as np


def classify_by_wheel_density(families, bw_method=0.28):
    """
    families: list các dict có dạng giống 'families' trong app.py, mỗi phần tử:
        {
            "pct": float (0..1),
            "macro": {"h": float, "s": float, "l": float, "is_neutral": bool, ...}
        }
    Trả về chính danh sách families đó, nhưng gán thêm f["macro"]["category"] in {"Chính","Phụ","Nhấn"}
    """
    # Nhóm trung tính (đen/trắng/xám) xử lý riêng theo tỷ lệ diện tích
    hued = [f for f in families if not f["macro"]["is_neutral"]]
    neutral = [f for f in families if f["macro"]["is_neutral"]]

    for f in neutral:
        # Nếu màu trung tính chiếm diện tích cực kỳ nhỏ, nó có thể làm điểm Nhấn sáng/tối
        if f["pct"] < 0.05 and len(hued) >= 2:
            f["macro"]["category"] = "Nhấn"
        else:
            f["macro"]["category"] = "Chính" if f["pct"] > 0.30 else "Phụ"

    if len(hued) == 0:
        return families
        
    if len(hued) == 1:
        # Sửa lỗi: Nếu màu sắc duy nhất chiếm tỷ lệ nhỏ (< 15%), nó phải là màu Nhấn
        hued[0]["macro"]["category"] = "Nhấn" if hued[0]["pct"] < 0.15 else "Chính"
        return families

    # 1) Chiếu lên tọa độ 2D của bánh xe màu sắc
    pts = np.array([
        [f["macro"]["s"] * np.cos(f["macro"]["h"] * 2 * np.pi),
         f["macro"]["s"] * np.sin(f["macro"]["h"] * 2 * np.pi)]
        for f in hued
    ]).T

    weights = np.array([f["pct"] for f in hued], dtype=float)
    weights = weights / (weights.sum() + 1e-9)

    # 2) Ước lượng mật độ KDE an toàn (tránh lỗi sập LinAlgError khi ma trận suy biến)
    density = np.ones(len(hued))
    if len(hued) >= 3:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(pts, weights=weights, bw_method=bw_method)
            density = kde(pts)
            d_max = density.max()
            if d_max > 0:
                density = density / d_max
        except Exception:
            # Dự phòng trả về mảng mật độ đồng đều nếu tính toán ma trận bị lỗi
            density = np.ones(len(hued))

    # 3) Tính điểm quan trọng và khoảng cách cô lập thực tế
    importance = density * weights
    dmat = np.linalg.norm(pts.T[:, None, :] - pts.T[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    isolation = dmat.min(axis=1)
    isolation_ratio = isolation / (isolation.min() + 1e-9)

    # 4) Xếp hạng và phân loại thông minh kết hợp màng lọc bão hòa màu
    order = np.argsort(-importance)
    med_pct = np.median(weights)
    ISOLATION_RATIO_THRESHOLD = 2.5

    for rank, i in enumerate(order):
        f = hued[i]
        pct = f["pct"]
        s = f["macro"]["s"]
        iso_r = isolation_ratio[i]

        if rank == 0:
            f["macro"]["category"] = "Chính"
        elif rank == 1 and importance[i] > 0.3 * importance[order[0]]:
            f["macro"]["category"] = "Phụ"
        # BẢO VỆ MÀU NHẤN THẬT: Màu rực rỡ (S > 0.22) diện tích nhỏ, hoặc màu nằm cô lập rõ rệt trên bánh xe màu
        elif (s > 0.22 and pct < 0.12) or (iso_r >= ISOLATION_RATIO_THRESHOLD and pct < med_pct):
            f["macro"]["category"] = "Nhấn"
        else:
            f["macro"]["category"] = "Phụ"

    # QUY TẮC KHỬ MÀU NHẤN TRUNG TÍNH:
    # Nếu đã có sự xuất hiện của các màu sắc Nhấn rực rỡ khác, ta đẩy màu trung tính xám xịt về làm màu Phụ
    has_chromatic_accent = any(f["macro"]["category"] == "Nhấn" for f in hued)
    if has_chromatic_accent:
        for f in neutral:
            if f["macro"]["category"] == "Nhấn":
                f["macro"]["category"] = "Phụ"

    # Đảm bảo luôn có tối thiểu 1 màu Nhấn nếu có từ 3 nhóm màu sắc rực rỡ trở lên
    cats = [f["macro"]["category"] for f in hued]
    if "Nhấn" not in cats and len(hued) >= 3:
        weakest = order[-1]
        hued[weakest]["macro"]["category"] = "Nhấn"

    return families


if __name__ == "__main__":
    # Chạy thử nghiệm kịch bản mô phỏng hình ảnh của bạn:
    # - 1 màu nâu lớn (Chính), 1 nhóm nâu/xám chuyển tiếp (Phụ), 
    # - 1 màu đỏ rực nhỏ diện tích 1.1% (Nhấn), 1 màu vàng cát rực nhỏ 1.8% (Nhấn)
    demo = [
        {"pct": 0.343, "macro": {"h": 0.08, "s": 0.15, "l": 0.30, "is_neutral": False}}, # Nâu xám lớn
        {"pct": 0.231, "macro": {"h": 0.10, "s": 0.25, "l": 0.45, "is_neutral": False}}, # Nâu ấm chuyển tiếp
        {"pct": 0.064, "macro": {"h": 0.12, "s": 0.67, "l": 0.50, "is_neutral": False}}, # Vàng cát rực rỡ
        {"pct": 0.011, "macro": {"h": 0.01, "s": 0.71, "l": 0.60, "is_neutral": False}}, # Đỏ Coral rực rỡ
        {"pct": 0.015, "macro": {"h": 0.55, "s": 0.08, "l": 0.70, "is_neutral": True}},  # Xám xanh nhạt (Trung tính)
    ]
    result = classify_by_wheel_density(demo)
    print("--- KẾT QUẢ PHÂN LOẠI MỚI ---")
    for f in result:
        m = f["macro"]
        neutral_str = "Neutral" if m["is_neutral"] else "Chromatic"
        print(f'pct={f["pct"]*100:5.1f}%  s={m["s"]:.2f} ({neutral_str:9s}) -> {m["category"]}')