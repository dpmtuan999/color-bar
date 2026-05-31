import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import io

st.set_page_config(page_title="Beyond Photography | Color Analyzer", layout="wide", initial_sidebar_state="expanded")

RESAMPLE_FILTER = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS

# ==========================================
# CUSTOM CSS — Google Sans Flex Light Theme
# ==========================================
custom_css = """
<style>
    /* Nhập font chữ Google Sans Flex chính thức từ máy chủ Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans+Flex:opsz,wght@8..40,100..1000&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-main:    #F5F5F7;   /* Màu nền xám nhẹ đặc trưng tối giản */
        --bg-card:    #FFFFFF;   /* Thẻ nền trắng tinh khiết */
        --bg-hover:   #E8E8ED;   /* Xám nhẹ khi hover */
        --border:     #D2D2D7;   /* Đường viền mảnh xám sáng */
        --border-soft:#E8E8ED;   /* Đường phân tách mỏng hơn */
        --text-pri:   #1D1D1F;   /* Màu than đen tối giản */
        --text-sec:   #515154;   /* Màu văn bản phụ cân bằng */
        --text-mute:  #86868B;   /* Xám nhạt bổ trợ */
        --accent:     #9A8A70;   /* Màu đồng thau phong cách cổ điển */
        --accent-dim: #C5B8A5;
        --white:      #FFFFFF;
    }

    /* ── Reset & Base (Ép sử dụng Google Sans Flex đồng bộ) ── */
    html, body, p, h1, h2, h3, h4, h5, h6, li, label, table, td, th, [data-testid="stMarkdownContainer"] p, span {
        font-family: "Google Sans Flex", "Google Sans", "Plus Jakarta Sans", -apple-system, sans-serif !important;
    }
    button div p, button div, .stDownloadButton button {
        font-family: "Google Sans Flex", "Google Sans", "Plus Jakarta Sans", sans-serif !important;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-pri) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden !important; }
    .block-container { padding-top: 2rem !important; padding-bottom: 4rem !important; max-width: 1300px !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] > div { padding: 2rem 1.5rem !important; }

    /* ── Typography (Google Sans Flex) ── */
    h1 {
        font-family: "Google Sans Flex", sans-serif !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        color: var(--text-pri) !important;
        line-height: 1.1 !important;
        margin-bottom: 0 !important;
    }
    h2, h3 {
        font-family: "Google Sans Flex", sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-pri) !important;
        letter-spacing: -0.02em !important;
    }
    h4, h5, h6 {
        font-family: "Google Sans Flex", sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-pri) !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        font-size: 0.72rem !important;
    }
    p, li, label, td, th {
        color: var(--text-sec) !important;
        font-size: 0.9rem !important;
        line-height: 1.65 !important;
    }

    /* ── Dividers ── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        background-color: var(--bg-card) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 0 !important;
        padding: 2rem !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-dim) !important;
    }
    [data-testid="stFileUploadDropzone"] p {
        color: var(--text-mute) !important;
        font-size: 0.82rem !important;
    }

    /* ── Buttons ── */
    .stDownloadButton > button {
        background-color: transparent !important;
        border: 1px solid var(--border) !important;
        border-radius: 0 !important;
        color: var(--text-sec) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em;
        padding: 0.4rem 0.8rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--text-pri) !important;
        color: var(--text-pri) !important;
        background-color: var(--bg-hover) !important;
    }

    /* ── Code / HEX tags ── */
    code {
        background-color: transparent !important;
        color: var(--text-mute) !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        border: none !important;
        padding: 0 !important;
        letter-spacing: 0.04em !important;
    }

    /* ── Info boxes ── */
    [data-testid="stAlert"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 0 !important;
        padding: 1rem 1.25rem !important;
        color: var(--text-sec) !important;
    }
    [data-testid="stAlert"] p { color: var(--text-sec) !important; }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 1px solid var(--border-soft) !important;
        border-radius: 0 !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-sec) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 0 !important;
    }
    [data-testid="stExpander"] summary:hover { color: var(--text-pri) !important; }
    [data-testid="stExpander"] > div { padding: 0.5rem 0 1rem 0 !important; }

    /* ── Tables ── */
    table {
        border-collapse: collapse !important;
        width: 100% !important;
        font-size: 0.82rem !important;
    }
    th {
        color: var(--text-mute) !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        font-size: 0.68rem !important;
        border-bottom: 1px solid var(--border) !important;
        padding: 0.5rem 0.75rem !important;
    }
    td {
        border-bottom: 1px solid var(--border-soft) !important;
        padding: 0.5rem 0.75rem !important;
        color: var(--text-sec) !important;
    }

    /* ── Image ── */
    [data-testid="stImage"] img {
        border: 1px solid var(--border) !important;
    }

    /* ── Uploaded file label ── */
    [data-testid="stFileUploaderFileName"] {
        color: var(--accent) !important;
        font-size: 0.8rem !important;
    }

    /* ── Column gaps ── */
    [data-testid="column"] { padding: 0 0.5rem !important; }

    /* ── Caption ── */
    [data-testid="stCaptionContainer"] p {
        font-size: 0.72rem !important;
        color: var(--text-mute) !important;
        letter-spacing: 0.04em !important;
        margin-top: 0.4rem !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown(
        "<p style='font-family: Google Sans Flex, sans-serif; font-size: 1.3rem; font-weight: 700; color: #9A8A70; letter-spacing: 0.05em; margin-bottom: 0.15rem;'>BEYOND</p>"
        "<p style='font-family: Google Sans Flex, sans-serif; font-size: 1.3rem; font-weight: 700; color: #1D1D1F; letter-spacing: 0.05em; margin-top: 0;'>PHOTOGRAPHY</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size: 0.72rem; color: #86868B; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 1.5rem;'>Color Analysis System</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        "<p style='font-size: 0.78rem; color: #515154; line-height: 1.7;'>"
        "Phối hợp màu sắc theo quy tắc bánh xe màu hoặc tỷ lệ vàng "
        "<strong style='color: #9A8A70;'>60 – 30 – 10</strong> "
        "để đạt hiệu quả thị giác tốt nhất."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    with st.expander("● Màu Chính — Dominant"):
        st.write("Màu chủ đạo chiếm diện tích lớn nhất. Thiết lập tone, cảm xúc tổng thể và định hình phong cách cho tác phẩm.")
    with st.expander("○ Màu Phụ — Secondary"):
        st.write("Màu hỗ trợ màu chính, tạo chiều sâu, sự phong phú và cân bằng thị giác. Tránh thiết kế đơn điệu.")
    with st.expander("✦ Màu Nhấn — Accent"):
        st.write("Màu tương phản cao, dùng cho chi tiết nhỏ. Thu hút ánh nhìn và làm nổi bật thành phần quan trọng nhất.")

    st.markdown("---")

    st.markdown(
        "<p style='font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em; color: #86868B; margin-bottom: 0.75rem;'>Tỷ lệ tham chiếu</p>",
        unsafe_allow_html=True
    )
    ratio_html = """
    <div style="display: flex; flex-direction: column; gap: 6px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 60%; height: 4px; background: #9A8A70;"></div>
            <span style="font-size: 0.75rem; color: #515154;">60% Chính</span>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 30%; height: 4px; background: #C5B8A5;"></div>
            <span style="font-size: 0.75rem; color: #515154;">30% Phụ</span>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 10%; height: 4px; background: #E5E5EA;"></div>
            <span style="font-size: 0.75rem; color: #515154;">10% Nhấn</span>
        </div>
    </div>
    """
    st.markdown(ratio_html, unsafe_allow_html=True)

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 0.65rem; color: #86868B; text-transform: uppercase; letter-spacing: 0.1em;'>v2.2 — Redesigned Algorithm</p>",
        unsafe_allow_html=True
    )


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def rgb_to_hex(rgb):
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

def get_color_name(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h_deg = h * 360
    if l < 0.12: return "Đen"
    if l > 0.88 and s < 0.15: return "Trắng"
    if s < 0.12: return "Xám Sáng" if l > 0.5 else "Xám Tối"
    if (30 <= h_deg < 60) and (0.1 <= s < 0.35) and (l > 0.65): return "Kem / Be"
    if (10 <= h_deg < 45) and (l < 0.45): return "Nâu"
    if (320 <= h_deg < 350) and (l > 0.5): return "Hồng"
    if h_deg < 15 or h_deg >= 345: return "Đỏ"
    elif 15 <= h_deg < 45: return "Cam"
    elif 45 <= h_deg < 75: return "Vàng"
    elif 75 <= h_deg < 160: return "Xanh Lá"
    elif 160 <= h_deg < 195: return "Xanh Ngọc"
    elif 195 <= h_deg < 250: return "Xanh Dương"
    elif 250 <= h_deg < 285: return "Xanh Tím"
    elif 285 <= h_deg < 320: return "Tím"
    return "Màu Khác"

def get_color_family(color_name):
    if color_name in ["Đỏ", "Hồng"]: return "Đỏ/Hồng"
    if color_name in ["Cam", "Nâu", "Kem / Be"]: return "Cam/Nâu"
    if color_name in ["Xanh Dương", "Xanh Tím"]: return "Xanh Dương/Tím"
    return color_name

def sort_colors_by_hue_and_pct(colors_list):
    if not colors_list: return []
    families = {}
    for c in colors_list:
        fn = get_color_family(c["name"])
        families.setdefault(fn, []).append(c)
    for fam in families:
        families[fam] = sorted(families[fam], key=lambda x: x["pct"], reverse=True)
    sorted_families = sorted(families.items(), key=lambda x: x[1][0]["pct"], reverse=True)
    result = []
    for _, items in sorted_families:
        result.extend(items)
    return result

def analyze_color(rgb, pct):
    rgb_tuple = tuple(int(x) for x in rgb)
    r, g, b = [x / 255.0 for x in rgb_tuple]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    is_neutral = (s < 0.22) or (l < 0.15) or (l > 0.88)
    return {
        "rgb": rgb_tuple, "hex": rgb_to_hex(rgb_tuple),
        "pct": pct, "h": h, "l": l, "s": s,
        "is_neutral": is_neutral, "name": get_color_name(rgb_tuple)
    }

def get_weighted_average_color(group):
    if not group: return (128, 128, 128)
    total = sum(i["pct"] for i in group)
    return (
        int(sum(i["rgb"][0] * i["pct"] for i in group) / total),
        int(sum(i["rgb"][1] * i["pct"] for i in group) / total),
        int(sum(i["rgb"][2] * i["pct"] for i in group) / total),
    )

def generate_bar_png(items, width=1400, height=100):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    total = sum(i["pct"] for i in items)
    x = 0
    for item in items:
        w = int((item["pct"] / total) * width)
        draw.rectangle([x, 0, x + w, height], fill=item["rgb"])
        x += w
    if x < width:
        draw.rectangle([x, 0, width, height], fill=items[-1]["rgb"])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_macro_bar_png(sorted_groups, width=1400, height=100):
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = 0
    for g in sorted_groups:
        if g["total_pct"] > 0:
            w = int((g["total_pct"] / 100) * width)
            draw.rectangle([x, 0, x + w, height], fill=g["avg_rgb"])
            x += w
    if x < width:
        draw.rectangle([x, 0, width, height], fill=sorted_groups[-1]["avg_rgb"])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@st.cache_data
def get_base_color_wheel(size=1200):
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv**2 + yv**2)
    theta = np.arctan2(yv, xv)
    h = (theta + np.pi) / (2 * np.pi)
    s = np.clip(r, 0, 1.0)
    v = np.ones_like(r)
    h_6 = h * 6.0
    i = h_6.astype(int)
    f = h_6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    for k, (ri, gi, bi) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        idx = (i % 6 == k)
        rgb[idx] = np.stack([ri[idx], gi[idx], bi[idx]], axis=-1) * 255
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = (r <= 1.0).astype(np.uint8) * 255
    return Image.fromarray(rgba, 'RGBA')

def generate_plotted_wheel(data_list, size=1200, target_size=400):
    wheel_img = get_base_color_wheel(size)
    bg = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    bg.paste(wheel_img, (0, 0), wheel_img)
    wheel_img = bg
    draw = ImageDraw.Draw(wheel_img)
    center = size / 2
    max_r = (size / 2) * 0.93
    scale = size / 800.0
    ring_r = int(18 * scale)
    dot_r = int(12 * scale)
    sorted_pts = sorted(data_list, key=lambda x: x["pct"])
    for item in sorted_pts:
        r_c, g_c, b_c = item["rgb"]
        h, s, v_v = colorsys.rgb_to_hsv(r_c/255.0, g_c/255.0, b_c/255.0)
        angle = h * 2 * np.pi - np.pi
        x = center + s * max_r * np.cos(angle)
        y = center + s * max_r * np.sin(angle)
        draw.ellipse([x-ring_r, y-ring_r, x+ring_r, y+ring_r], fill=(255,255,255,230))
        draw.ellipse([x-dot_r, y-dot_r, x+dot_r, y+dot_r], fill=(r_c, g_c, b_c, 255))
    if target_size != size:
        return wheel_img.resize((target_size, target_size), resample=RESAMPLE_FILTER)
    return wheel_img


# ==========================================
# MAIN LAYOUT
# ==========================================

# ── Header ──
st.markdown(
    "<h1>Beyond Photography</h1>"
    "<p style='font-size: 0.78rem; color: #86868B; text-transform: uppercase; letter-spacing: 0.16em; margin-top: 0.3rem; margin-bottom: 0;'>Hệ thống phân tích màu sắc</p>",
    unsafe_allow_html=True
)
st.markdown("---")

col_left, col_right = st.columns([1, 2], gap="large")

sorted_dominant, sorted_secondary, sorted_accent = [], [], []

# ── LEFT COLUMN: Upload + Image ──
with col_left:
    st.markdown(
        "<p style='font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em; color: #86868B; margin-bottom: 0.5rem;'>Nhập hình ảnh</p>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Kéo thả hoặc nhấn để chọn ảnh",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown(
        "<p style='font-size: 0.72rem; color: #86868B; text-align: center; margin-top: -0.5rem;'>JPG / PNG · Kéo thả hoặc nhấn để chọn</p>",
        unsafe_allow_html=True
    )

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

    if image is not None:
        # ── Run K-means ──
        img_r = image.resize((120, 120))
        arr = np.array(img_r)
        pixels = arr[:,:,:3].reshape(-1,3) if arr.shape[-1] == 4 else arr.reshape(-1,3)
        km = KMeans(n_clusters=20, random_state=42, n_init=10)
        labels = km.fit_predict(pixels)
        colors = km.cluster_centers_.astype(int)
        percentages = np.bincount(labels, minlength=20) / len(pixels)

        data_list = [analyze_color(colors[i], percentages[i]) for i in range(20)]
        
        # ════════════════════════════════════════
        # THUẬT TOÁN PHÂN CHIA VÀ XÁC ĐỊNH CHÍNH - PHỤ - NHẤN THÔNG MINH (REDESIGNED)
        # ════════════════════════════════════════
        sorted_by_pct = sorted(data_list, key=lambda x: x["pct"], reverse=True)
        
        dominant_group = []
        secondary_group = []
        accent_group = []
        
        # 1. Nhận diện Màu Chính (Dominant):
        # Cộng dồn các màu lớn nhất cho đến khi diện tích tích lũy đạt trên 55%
        # Giới hạn số màu chính tối đa là 4 màu để tránh bị loãng
        cumulative_pct = 0.0
        for c in sorted_by_pct:
            if len(dominant_group) < 1 or (cumulative_pct < 0.55 and len(dominant_group) < 4):
                dominant_group.append(c)
                cumulative_pct += c["pct"]
                
        # Tính góc Hue trung bình của nhóm Màu Chính làm mốc tham chiếu
        dom_total_pct = sum(c["pct"] for c in dominant_group)
        dom_hue = sum(c["h"] * c["pct"] for c in dominant_group) / dom_total_pct
        
        remaining = [c for c in sorted_by_pct if c not in dominant_group]
        
        # 2. Nhận diện Màu Nhấn (Accent) dựa trên độ tương phản sắc độ và độ rực rỡ:
        scored_remaining = []
        for c in remaining:
            vibrancy = c["s"] * (1.0 - abs(2 * c["l"] - 1))
            hue_contrast = min(abs(c["h"] - dom_hue), 1.0 - abs(c["h"] - dom_hue))
            accent_score = vibrancy * (1.0 + 3.5 * hue_contrast) / (c["pct"] + 0.01)
            scored_remaining.append((accent_score, c))
            
        scored_remaining = sorted(scored_remaining, key=lambda x: x[0], reverse=True)
        
        # Chọn Màu Nhấn: Lọc màu rực rỡ, diện tích nhỏ (< 6%), tổng diện tích nhóm nhấn <= 12%
        acc_cumulative_pct = 0.0
        for score, c in scored_remaining:
            if c["s"] >= 0.15 and c["pct"] < 0.06 and acc_cumulative_pct + c["pct"] <= 0.12 and len(accent_group) < 5:
                accent_group.append(c)
                acc_cumulative_pct += c["pct"]
                
        if not accent_group:
            sorted_remaining_by_pct_asc = sorted(remaining, key=lambda x: x["pct"])
            accent_group = sorted_remaining_by_pct_asc[:3]
            
        # 3. Nhận diện Màu Phụ (Secondary): Các dải trung tính nền và màu hỗ trợ chuyển tiếp còn lại
        secondary_group = [c for c in remaining if c not in accent_group]

        sorted_dominant = sort_colors_by_hue_and_pct(dominant_group)
        sorted_secondary = sort_colors_by_hue_and_pct(secondary_group)
        sorted_accent = sort_colors_by_hue_and_pct(accent_group)

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em; color: #86868B; margin-bottom: 0.5rem;'>Ảnh đang phân tích</p>",
            unsafe_allow_html=True
        )
        st.image(image, use_container_width=True)

        # ── Quick summary chips ──
        avg_dom = get_weighted_average_color(sorted_dominant)
        avg_sec = get_weighted_average_color(sorted_secondary)
        avg_acc = get_weighted_average_color(sorted_accent)
        pct_dom = sum(i["pct"] for i in sorted_dominant) * 100
        pct_sec = sum(i["pct"] for i in sorted_secondary) * 100
        pct_acc = sum(i["pct"] for i in sorted_accent) * 100

        chips_html = f"""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 1rem;">
            <div style="display: flex; align-items: center; gap: 6px; background: #FFFFFF; border: 1px solid #E5E5EA; padding: 5px 10px;">
                <div style="width: 10px; height: 10px; background: {rgb_to_hex(avg_dom)};"></div>
                <span style="font-size: 0.68rem; color: #515154; text-transform: uppercase; letter-spacing: 0.08em;">Chính {pct_dom:.0f}%</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; background: #FFFFFF; border: 1px solid #E5E5EA; padding: 5px 10px;">
                <div style="width: 10px; height: 10px; background: {rgb_to_hex(avg_sec)};"></div>
                <span style="font-size: 0.68rem; color: #515154; text-transform: uppercase; letter-spacing: 0.08em;">Phụ {pct_sec:.0f}%</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; background: #FFFFFF; border: 1px solid #E5E5EA; padding: 5px 10px;">
                <div style="width: 10px; height: 10px; background: {rgb_to_hex(avg_acc)};"></div>
                <span style="font-size: 0.68rem; color: #515154; text-transform: uppercase; letter-spacing: 0.08em;">Nhấn {pct_acc:.0f}%</span>
            </div>
        </div>
        """
        st.markdown(chips_html, unsafe_allow_html=True)

    else:
        st.markdown(
            "<div style='background: #FFFFFF; border: 1px solid #E5E5EA; padding: 2.5rem 1.5rem; text-align: center; margin-top: 1rem;'>"
            "<p style='font-size: 0.82rem; color: #86868B; margin: 0;'>Tải ảnh lên để bắt đầu phân tích</p>"
            "</div>",
            unsafe_allow_html=True
        )


# ── RIGHT COLUMN: Analysis ──
with col_right:
    if image is not None and len(sorted_dominant) > 0:

        pct_dom = sum(i["pct"] for i in sorted_dominant) * 100
        pct_sec = sum(i["pct"] for i in sorted_secondary) * 100
        pct_acc = sum(i["pct"] for i in sorted_accent) * 100

        avg_dom = get_weighted_average_color(sorted_dominant)
        avg_sec = get_weighted_average_color(sorted_secondary)
        avg_acc = get_weighted_average_color(sorted_accent)

        group_data = [
            {"name": "Màu Chính", "sub": "Tông màu chủ đạo", "key": "dom",
             "items": sorted_dominant, "total_pct": pct_dom,
             "avg_rgb": avg_dom, "avg_hex": rgb_to_hex(avg_dom), "label_short": "Chính"},
            {"name": "Màu Phụ", "sub": "Màu trung tính nền hỗ trợ", "key": "sec",
             "items": sorted_secondary, "total_pct": pct_sec,
             "avg_rgb": avg_sec, "avg_hex": rgb_to_hex(avg_sec), "label_short": "Phụ"},
            {"name": "Màu Nhấn", "sub": "Chi tiết tạo điểm nhấn", "key": "acc",
             "items": sorted_accent, "total_pct": pct_acc,
             "avg_rgb": avg_acc, "avg_hex": rgb_to_hex(avg_acc), "label_short": "Nhấn"},
        ]
        group_data_sorted = sorted(group_data, key=lambda x: x["total_pct"], reverse=True)

        # ════════════════════════════════════════
        # SECTION A — COLOR WHEEL + MACRO BAR (side by side)
        # ════════════════════════════════════════
        st.markdown(
            "<p style='font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em; color: #86868B; margin-bottom: 1rem;'>Phân tích hệ màu</p>",
            unsafe_allow_html=True
        )

        col_wheel, col_bars_top = st.columns([1, 1.3], gap="large")

        full_pts = sorted_dominant + sorted_secondary + sorted_accent
        wheel_web = generate_plotted_wheel(full_pts, size=1200, target_size=280)

        with col_wheel:
            st.markdown(
                "<p style='font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em; color: #86868B; margin-bottom: 0.4rem;'>Bánh xe màu sắc</p>",
                unsafe_allow_html=True
            )
            st.image(wheel_web, width=280)
            buf_wheel = io.BytesIO()
            hr_wheel = generate_plotted_wheel(full_pts, size=1200, target_size=1200)
            hr_wheel.save(buf_wheel, format="PNG")
            st.download_button("↓ Tải bánh xe (PNG)", data=buf_wheel.getvalue(),
                               file_name="color_wheel.png", mime="image/png", key="dl_wheel")

        with col_bars_top:
            st.markdown(
                "<p style='font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em; color: #86868B; margin-bottom: 0.75rem;'>Màu đại diện nhóm</p>",
                unsafe_allow_html=True
            )
            for g in group_data_sorted:
                hex_c = g["avg_hex"]
                st.markdown(
                    f"""<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
                        <div style="width: 48px; height: 48px; background: {hex_c}; flex-shrink: 0; border: 1px solid rgba(0,0,0,0.06);"></div>
                        <div>
                            <p style='margin: 0; font-size: 0.75rem; color: #1D1D1F; font-weight: 500;'>{g["name"]}</p>
                            <p style='margin: 0; font-size: 0.68rem; color: #86868B; text-transform: uppercase; letter-spacing: 0.06em;'>{hex_c}</p>
                            <p style='margin: 0; font-size: 0.68rem; color: #515154;'>{g["total_pct"]:.1f}% từ ảnh gốc</p>
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # ════════════════════════════════════════
        # SECTION B — MACRO BAR (TĂNG ĐỘ DÀY LÊN 48PX, FIX LỖI NHẢY CHỮ Ô NHỎ)
        # ════════════════════════════════════════
        st.markdown(
            "<p style='font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em; color: #86868B; margin-bottom: 0.5rem;'>Thanh tỷ lệ màu — Khái quát</p>",
            unsafe_allow_html=True
        )

        col_macrobar, col_macrobtn = st.columns([6, 1], gap="small")
        with col_macrobar:
            macro_html = '<div style="display: flex; width: 100%; height: 48px; overflow: hidden; border: 1px solid #E5E5EA; gap: 1px;">'
            for g in group_data_sorted:
                if g["total_pct"] > 0:
                    lum = 0.299*g["avg_rgb"][0] + 0.587*g["avg_rgb"][1] + 0.114*g["avg_rgb"][2]
                    txt = "#0E0E0F" if lum > 140 else "#F0EDE8"
                    
                    segment_text = f'{g["label_short"]} {g["total_pct"]:.0f}%' if g["total_pct"] >= 8.0 else ""
                    
                    macro_html += (
                        f'<div style="background:{g["avg_hex"]}; width:{g["total_pct"]}%; height:100%;'
                        f' display:flex; align-items:center; justify-content:center;'
                        f' font-size:10px; color:{txt}; font-weight:600; letter-spacing:0.05em;'
                        f' font-family: Google Sans Flex, sans-serif;"'
                        f' title="{g["label_short"]}: {g["avg_hex"]} ({g["total_pct"]:.1f}%)">'
                        f'{segment_text}</div>'
                    )
            macro_html += '</div>'
            st.markdown(macro_html, unsafe_allow_html=True)
        with col_macrobtn:
            mp = generate_macro_bar_png(group_data_sorted)
            st.download_button("↓", data=mp, file_name="macro_bar.png", mime="image/png", key="dl_macro")

        # ── Ratio info row ──
        ratio_str = " · ".join([f"<span style='color:#1D1D1F;font-weight:500;'>{g['label_short']}</span> <span style='color:#9A8A70;'>{g['total_pct']:.1f}%</span>" for g in group_data_sorted])
        st.markdown(
            f"<p style='font-size: 0.75rem; color: #86868B; margin-top: 0.5rem;'>{ratio_str}</p>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        # ── Detail bar (TĂNG ĐỘ DÀY LÊN 30PX) ──
        st.markdown(
            "<p style='font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em; color: #86868B; margin-bottom: 0.5rem;'>Thanh tỷ lệ màu — Chi tiết 20 sắc độ</p>",
            unsafe_allow_html=True
        )
        col_detailbar, col_detailbtn = st.columns([6, 1], gap="small")
        master_list = []
        for g in group_data_sorted:
            master_list.extend(g["items"])
        with col_detailbar:
            detail_html = '<div style="display: flex; width: 100%; height: 30px; overflow: hidden; border: 1px solid #E5E5EA;">'
            for item in master_list:
                r2, g2, b2 = item["rgb"]
                pv = item["pct"] * 100
                detail_html += f'<div style="background:rgb({r2},{g2},{b2}); width:{pv}%; height:100%;" title="{item["hex"]} ({pv:.1f}%)"></div>'
            detail_html += '</div>'
            st.markdown(detail_html, unsafe_allow_html=True)
        with col_detailbtn:
            dp = generate_bar_png(master_list)
            st.download_button("↓", data=dp, file_name="detail_bar.png", mime="image/png", key="dl_detail")

        st.markdown("---")

        # ════════════════════════════════════════
        # SECTION C — GROUP DETAIL SWATCHES (SỬA LỖI: LƯỚI 8 CỘT CHỐNG TRÀN CHỮ)
        # ════════════════════════════════════════
        st.markdown(
            "<p style='font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em; color: #86868B; margin-bottom: 1rem;'>Dải sắc độ từng phân nhóm</p>",
            unsafe_allow_html=True
        )

        def render_group(g):
            total_pct = g["total_pct"]
            items = sorted(g["items"], key=lambda x: x["pct"], reverse=True)

            # Group header row
            col_title, col_gbtn = st.columns([5, 1], gap="small")
            with col_title:
                st.markdown(
                    f"<div style='display:flex; align-items:baseline; gap: 10px; margin-bottom: 6px;'>"
                    f"<span style='font-family: Google Sans Flex, sans-serif; font-size: 1.1rem; color: #1D1D1F; font-weight: 550;'>{g['name']}</span>"
                    f"<span style='font-size: 0.68rem; color: #86868B; text-transform: uppercase; letter-spacing: 0.1em;'>{g['sub']}</span>"
                    f"<span style='font-size: 0.82rem; color: #9A8A70; margin-left: auto;'>{total_pct:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                # Group bar - Độ dày tăng lên 20px
                bar_html = '<div style="display:flex; width:100%; height:20px; overflow:hidden; border: 1px solid #E5E5EA; margin-bottom: 14px;">'
                for item in items:
                    w = (item["pct"] / (total_pct / 100)) * 100
                    bar_html += f'<div style="background:rgb({item["rgb"][0]},{item["rgb"][1]},{item["rgb"][2]}); width:{w}%; height:100%;"></div>'
                bar_html += '</div>'
                st.markdown(bar_html, unsafe_allow_html=True)
            with col_gbtn:
                gp = generate_bar_png(items)
                st.download_button("↓", data=gp, file_name=f"bar_{g['key']}.png",
                                   mime="image/png", key=f"dl_grp_{g['key']}")

            # SỬA LỖI: Chuyển lưới từ 10 cột sang 8 cột để cột rộng rãi hơn, chống lỗi tràn chữ mã màu
            cols_per_row = 8
            for ridx in range(0, len(items), cols_per_row):
                chunk = items[ridx: ridx + cols_per_row]
                cols = st.columns(cols_per_row)
                for cidx, item in enumerate(chunk):
                    pv = item["pct"] * 100
                    with cols[cidx]:
                        # Swatch block - Bo góc vuông 0px
                        st.markdown(
                            f'<div style="background:{item["hex"]}; height: 36px; border: 1px solid rgba(0,0,0,0.06); margin-bottom: 3px;"></div>',
                            unsafe_allow_html=True
                        )
                        # Percentage
                        st.markdown(
                            f'<p style="font-size: 0.65rem; text-align: center; margin: 0 0 1px 0; color: #515154; font-weight: 500;">{pv:.1f}%</p>',
                            unsafe_allow_html=True
                        )
                        # HEX code (SỬA LỖI: Thêm white-space: nowrap và nén size 0.58rem để chống tuyệt đối việc nhảy chữ)
                        st.markdown(
                            f'<p style="font-size: 0.58rem; text-align: center; color: #86868B; font-family: Google Sans Flex, sans-serif; letter-spacing: -0.01em; white-space: nowrap; overflow: hidden;">{item["hex"]}</p>',
                            unsafe_allow_html=True
                        )
            st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        for g in group_data_sorted:
            render_group(g)
            st.markdown(
                "<div style='border-top: 1px solid #E8E8ED; margin-bottom: 1.25rem;'></div>",
                unsafe_allow_html=True
            )

    else:
        st.markdown(
            "<div style='background: #FFFFFF; border: 1px solid #E5E5EA; padding: 4rem 2rem; text-align: center; margin-top: 2rem;'>"
            "<p style='font-family: Google Sans Flex, sans-serif; font-size: 1.4rem; color: #86868B; margin: 0;'>Chưa có ảnh để phân tích</p>"
            "<p style='font-size: 0.8rem; color: #D2D2D7; margin-top: 0.5rem;'>Tải ảnh lên ở cột bên trái để bắt đầu</p>"
            "</div>",
            unsafe_allow_html=True
        )