import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import io
import base64
import os

st.set_page_config(
    page_title="Beyond Photography",
    layout="wide",
    initial_sidebar_state="expanded"
)

RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS

# ─── HỆ THỐNG GIAO DIỆN CHUYÊN NGHIỆP (CSS STYLE) ────────────────=============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans+Flex:opsz,wght@8..144,100..900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
p, h1, h2, h3, h4, h5, h6, li, label, table, td, th,
summary, [data-testid="stExpander"] summary,
button div p, button div,
.stDownloadButton button,
[data-testid="stSidebar"] * {
    font-family: "Google Sans Flex", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

:root {
    --bg:      #F5F5F7;
    --surface: #FFFFFF;
    --line:    #E5E5EA;
    --line-s:  #F2F2F7;
    --t1:      #1D1D1F;
    --t2:      #515154;
    --t3:      #86868B;
    --warm:    #1D1D1F;
}

.stApp                    { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.block-container          { padding: 2.5rem 2.75rem 5rem !important; max-width: 1400px !important; }

[data-testid="stSidebar"]            { background: var(--surface) !important; border-right: 1px solid var(--line) !important; box-shadow: none !important; }
[data-testid="stSidebar"] > div      { padding: 2.25rem 1.75rem !important; }

[data-testid="stImage"],
[data-testid="stImage"] > *,
[data-testid="stImage"] img,
figure, figure img { border: none !important; border-radius: 0 !important; box-shadow: none !important; outline: none !important; display: block !important; line-height: 0 !important; }

[data-testid="stFileUploader"]       { background: var(--surface) !important; border: 1px dashed var(--line) !important; border-radius: 4px !important; padding: 1.75rem !important; }
[data-testid="stFileUploader"]:hover { border-color: var(--warm) !important; }

.stDownloadButton, .stButton {
    display: flex !important;
    align-items: stretch !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}
.stDownloadButton > button, .stButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
    color: var(--t2) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    padding: 0 10px !important;
    width: 100% !important;
    height: 34px !important;
    min-height: 34px !important;
    line-height: 1 !important;
    box-shadow: none !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    transition: all .2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 5px !important;
    white-space: nowrap !important;
}
.stDownloadButton > button:hover, .stButton > button:hover {
    border-color: var(--t1) !important;
    color: var(--surface) !important;
    background: var(--t1) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
}
/* Fix column gap và alignment cho hàng nút */
[data-testid="column"] > div {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

hr { border: none !important; border-top: 1px solid var(--line) !important; margin: 1.75rem 0 !important; }

h1                          { font-size: 2.25rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important; color: var(--t1) !important; line-height: 1.1 !important; margin: 0 !important; }
p, li, label, td, th, span { color: var(--t2) !important; font-size: 0.875rem !important; line-height: 1.6 !important; }

[data-testid="column"]  { padding: 0 0.35rem !important; }
::-webkit-scrollbar       { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─── HÀM HỖ TRỢ XỬ LÝ MÀU SẮC ─────────────────────────────────────────────────
def rgb_to_hex(rgb):
    return f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}"

def img_to_b64(img, fmt="JPEG"):
    buf = io.BytesIO()
    if fmt == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def lum(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

def on_color(rgb):
    return "#1D1D1F" if lum(rgb) > 155 else "#FFFFFF"

def wavg(grp):
    if not grp: return (128, 128, 128)
    tot = sum(i["pct"] for i in grp)
    if tot == 0: return grp[0]["rgb"]
    return tuple(int(sum(i["rgb"][j] * i["pct"] for i in grp) / tot) for j in range(3))


# ─── HÀM TẠO ẢNH XUẤT FILE (DOWNLOAD) ─────────────────────────────────────────
def bar_png(items, w=1400, h=60):
    img = Image.new("RGB", (w, h), (245, 245, 247))
    d = ImageDraw.Draw(img)
    x = 0
    tot = sum(i["pct"] for i in items)
    if tot == 0: return io.BytesIO().getvalue()
    for it in items:
        bw = int(it["pct"] / tot * w)
        if bw > 0:
            d.rectangle([x, 0, x + bw, h], fill=it["rgb"])
            x += bw
    if x < w and items:
        d.rectangle([x, 0, w, h], fill=items[-1]["rgb"])
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()

def macro_png_tineye(ordered_macro_list, w=1400, h=80):
    img = Image.new("RGB", (w, h), (245, 245, 247))
    d = ImageDraw.Draw(img)
    x = 0
    tot = sum(f["pct"] for f in ordered_macro_list)
    if tot == 0: return io.BytesIO().getvalue()
    for f in ordered_macro_list:
        bw = int(f["pct"] / tot * w)
        if bw > 0:
            d.rectangle([x, 0, x + bw, h], fill=f["macro"]["rgb"])
            x += bw
    if x < w and ordered_macro_list:
        d.rectangle([x, 0, w, h], fill=ordered_macro_list[-1]["macro"]["rgb"])
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()

def copy_bytes_to_clipboard(png_bytes):
    try:
        import subprocess
        temp_path = os.path.join(os.path.dirname(__file__), "temp_clip.png")
        with open(temp_path, "wb") as f:
            f.write(png_bytes)
        script = f'set the clipboard to (read (POSIX file "{temp_path}") as «class PNGf»)'
        res = subprocess.run(["osascript", "-e", script], capture_output=True)
        if os.path.exists(temp_path): os.remove(temp_path)
        return res.returncode == 0
    except Exception:
        return False

@st.cache_data
def wheel_base(size=1200):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv**2 + yv**2)
    theta = np.arctan2(yv, xv)
    h = (theta + np.pi) / (2 * np.pi)
    s = np.clip(r, 0, 1)
    v = np.ones_like(r)
    h6 = h * 6; i = h6.astype(int); f = h6 - i
    p = v * (1 - s); q = v * (1 - s * f); t = v * (1 - s * (1 - f))
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    for k, (ri, gi, bi) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        idx = (i % 6 == k)
        rgb[idx] = np.stack([ri[idx], gi[idx], bi[idx]], axis=-1) * 255
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = (r <= 1).astype(np.uint8) * 255
    return Image.fromarray(rgba, "RGBA")

def make_wheel(pts, size=1200, out=320):
    wimg = wheel_base(size)
    bg = Image.new("RGBA", (size, size), (245, 245, 247, 255))
    bg.paste(wimg, (0, 0), wimg)
    wimg = bg
    d = ImageDraw.Draw(wimg)
    cx = cy = size / 2
    mr = (size / 2) * 0.93
    sc = size / 800
    ring = int(19 * sc)
    dot  = int(13 * sc)
    for it in sorted(pts, key=lambda x: x["pct"]):
        rc, gc, bc = it["rgb"]
        h, s, _ = colorsys.rgb_to_hsv(rc / 255, gc / 255, bc / 255)
        ang = h * 2 * np.pi - np.pi
        x = cx + s * mr * np.cos(ang)
        y = cy + s * mr * np.sin(ang)
        d.ellipse([x-ring, y-ring, x+ring, y+ring], fill=(255, 255, 255, 240))
        d.ellipse([x-dot,  y-dot,  x+dot,  y+dot],  fill=(rc, gc, bc, 255))
    return wimg.resize((out, out), resample=RESAMPLE)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="font-family:'Google Sans Flex',sans-serif;font-size:1.35rem;font-weight:700;
                    color:#1D1D1F;letter-spacing:-0.025em;line-height:1.15;">
            Beyond<br><span style="color:#000;font-weight:800;">Photography</span>
        </div>
        <div style="margin-top:0.4rem;font-size:0.66rem;font-weight:600;letter-spacing:0.15em;
                    text-transform:uppercase;color:#86868B;">Color Analysis System</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<p style="font-size:0.84rem;color:var(--t2);line-height:1.65;margin-bottom:1rem;">
    Hệ thống phân tích cấu trúc diện tích màu, tự động bóc tách vùng màu trung tính và phân hạng dải màu mượt mà.
    </p>""", unsafe_allow_html=True)


# ─── TIÊU ĐỀ CHÍNH APP & KHỞI TẠO BIẾN CỘT TOÀN CỤC ───────────────────────────
st.markdown("""
<h1>Beyond Photography</h1>
<div style="margin-top:.35rem;font-size:.68rem;font-weight:600;letter-spacing:.2em;
            text-transform:uppercase;color:var(--t3);">Hệ thống phân tích màu sắc cấu trúc thị giác</div>
""", unsafe_allow_html=True)
st.markdown("---")

col_left, col_right = st.columns([1.1, 2.1], gap="large")

dom = []
sec = []
acc = []
ordered_macro_list = []
pd = ps = pa = 0
ad = as_ = aa = (128,128,128)

# ─── CỘT TRÁI: INPUT & THANH TỶ LỆ ĐỨNG FLEXIBLE THEO ẢNH GỐC ────────────────
with col_left:
    st.markdown("<p style='font-size:0.8rem;font-weight:600;color:var(--t2);margin-bottom:0.3rem;'>Tải hình ảnh</p>", unsafe_allow_html=True)
    up = st.file_uploader("upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    st.markdown("<div style='text-align:center;margin:0.25rem 0;font-size:0.72rem;color:var(--t3);'>hoặc</div>", unsafe_allow_html=True)

    if st.button("📋 Dán ảnh từ Clipboard", use_container_width=True):
        try:
            from PIL import ImageGrab
            clipboard_image = ImageGrab.grabclipboard()
            if clipboard_image is not None:
                if isinstance(clipboard_image, list) and len(clipboard_image) > 0:
                    st.session_state["clipboard_image"] = Image.open(clipboard_image[0])
                else:
                    st.session_state["clipboard_image"] = clipboard_image
                st.toast("Đã dán ảnh từ Clipboard thành công!", icon="📋")
            else:
                st.warning("Không tìm thấy hình ảnh trong Clipboard.")
        except Exception as e:
            st.error(f"Không thể đọc clipboard: {e}")

    image = None
    if up:
        image = Image.open(up)
        if "clipboard_image" in st.session_state: del st.session_state["clipboard_image"]
    elif "clipboard_image" in st.session_state:
        image = st.session_state["clipboard_image"]

    if image:
        work = image.convert("RGB")
        arr  = np.array(work.resize((300, 300), RESAMPLE))
        px   = arr.reshape(-1, 3)

        # Bước 1: Tìm 40 màu chi tiết gốc bằng K-Means
        km_detail = KMeans(n_clusters=40, random_state=42, n_init=10)
        lbs_detail = km_detail.fit_predict(px)
        clr_detail = km_detail.cluster_centers_.astype(int)
        cnt_detail = np.bincount(lbs_detail, minlength=40)
        pct_detail = cnt_detail / cnt_detail.sum()

        raw_micro_colors = []
        for i in range(40):
            if pct_detail[i] == 0: continue
            rgb_t = tuple(int(x) for x in clr_detail[i])
            r, g, b = [x / 255.0 for x in rgb_t]
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            
            is_neutral = (s < 0.16 or l < 0.13 or l > 0.88)
            if is_neutral:
                if l < 0.25: bin_id = 100
                elif l > 0.78: bin_id = 101
                else: bin_id = 102
            else:
                bin_id = int(h * 12) % 12
                
            raw_micro_colors.append({
                "rgb": rgb_t, "hex": rgb_to_hex(rgb_t), "pct": pct_detail[i],
                "h": h, "l": l, "s": s, "is_neutral": is_neutral, "bin_id": bin_id
            })

        # Bước 2: Bộ lọc gộp các sắc độ trùng lặp cận biên (Distance < 35)
        micro_colors = []
        raw_micro_colors.sort(key=lambda x: x["pct"], reverse=True)
        
        for r_mc in raw_micro_colors:
            found_match = False
            for m_mc in micro_colors:
                dist = np.linalg.norm(np.array(r_mc["rgb"]) - np.array(m_mc["rgb"]))
                if dist < 35 and r_mc["bin_id"] == m_mc["bin_id"]: 
                    total_pct = m_mc["pct"] + r_mc["pct"]
                    if total_pct > 0:
                        avg_r = (m_mc["rgb"][0] * m_mc["pct"] + r_mc["rgb"][0] * r_mc["pct"]) / total_pct
                        avg_g = (m_mc["rgb"][1] * m_mc["pct"] + r_mc["rgb"][1] * r_mc["pct"]) / total_pct
                        avg_b = (m_mc["rgb"][2] * m_mc["pct"] + r_mc["rgb"][2] * r_mc["pct"]) / total_pct
                        m_mc["rgb"] = (int(avg_r), int(avg_g), int(avg_b))
                        m_mc["hex"] = rgb_to_hex(m_mc["rgb"])
                    m_mc["pct"] = total_pct
                    mr, mg, mb = [x / 255.0 for x in m_mc["rgb"]]
                    m_mc["h"], m_mc["l"], m_mc["s"] = colorsys.rgb_to_hls(mr, mg, mb)
                    found_match = True
                    break
            if not found_match:
                micro_colors.append(r_mc)

        # Phân chia các sắc độ vào túi Hue lớn
        bin_groups = {}
        for mc in micro_colors:
            bid = mc["bin_id"]
            if bid not in bin_groups: bin_groups[bid] = []
            bin_groups[bid].append(mc)

        families = []
        for bid, items in bin_groups.items():
            tot_pct = sum(it["pct"] for it in items)
            avg_r = sum(it["rgb"][0] * it["pct"] for it in items) / tot_pct
            avg_r = max(0, min(255, avg_r))
            avg_g = sum(it["rgb"][1] * it["pct"] for it in items) / tot_pct
            avg_g = max(0, min(255, avg_g))
            avg_b = sum(it["rgb"][2] * it["pct"] for it in items) / tot_pct
            avg_b = max(0, min(255, avg_b))
            macro_rgb = (int(avg_r), int(avg_g), int(avg_b))
            mr, mg, mb = [x / 255.0 for x in macro_rgb]
            mh, ml, ms = colorsys.rgb_to_hls(mr, mg, mb)
            
            families.append({
                "bin_id": bid, "pct": tot_pct, "items": items,
                "macro": {
                    "rgb": macro_rgb, "hex": rgb_to_hex(macro_rgb), "pct": tot_pct,
                    "h": mh, "l": ml, "s": ms, "is_neutral": items[0]["is_neutral"]
                }
            })

        # Phân nhóm ĐỘC QUYỀN Chính - Phụ - Nhấn không chồng chéo tỷ lệ %
        families.sort(key=lambda x: x["pct"], reverse=True)
        
        dom_families, sec_families, acc_families = [], [], []
        current_sum = 0.0
        for f in families:
            if not dom_families:
                dom_families.append(f)
                current_sum += f["pct"] * 100
            elif current_sum < 55.0:
                dom_families.append(f)
                current_sum += f["pct"] * 100
            elif current_sum < 88.0:
                sec_families.append(f)
                current_sum += f["pct"] * 100
            else:
                acc_families.append(f)
                current_sum += f["pct"] * 100

        if not sec_families and acc_families: sec_families.append(acc_families.pop(0))
        if not sec_families and len(dom_families) > 1: sec_families.append(dom_families.pop())
        if not acc_families and len(sec_families) > 1: acc_families.append(sec_families.pop())

        def smooth_sort_key(x):
            return (x["is_neutral"], x["bin_id"], x["l"], x["s"])

        dom_g = []
        for f in dom_families: dom_g.extend(f["items"])
        sec_g = []
        for f in sec_families: sec_g.extend(f["items"])
        acc_g = []
        for f in acc_families: acc_g.extend(f["items"])

        dom = sorted(dom_g, key=smooth_sort_key)
        sec = sorted(sec_g, key=smooth_sort_key)
        acc = sorted(acc_g, key=smooth_sort_key)

        ordered_macro_list = []
        for f in sorted(dom_families, key=lambda x: (x["macro"]["is_neutral"], x["bin_id"])):
            f["macro"]["category"] = "Chính"
            ordered_macro_list.append(f)
        for f in sorted(sec_families, key=lambda x: (x["macro"]["is_neutral"], x["bin_id"])):
            f["macro"]["category"] = "Phụ"
            ordered_macro_list.append(f)
        for f in sorted(acc_families, key=lambda x: (x["macro"]["is_neutral"], x["bin_id"])):
            f["macro"]["category"] = "Nhấn"
            ordered_macro_list.append(f)

        pd = sum(f["pct"] for f in dom_families) * 100
        ps = sum(f["pct"] for f in sec_families) * 100
        pa = sum(f["pct"] for f in acc_families) * 100
        ad = wavg(dom); as_ = wavg(sec); aa = wavg(acc)

        # ── KHU VỰC HIỂN THỊ KIỂU SIDE-BY-SIDE ĐỐI XỨNG THEO ẢNH GỐC ──
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        
        # Tạo bản thumb chuẩn tỉ lệ cho ảnh gốc bên trái
        thumb = work.copy()
        thumb.thumbnail((500, 500), RESAMPLE)
        img_b64 = img_to_b64(thumb)
        
        # Khởi tạo các block màu cho thanh đứng chứa trọn vẹn dải màu
        vertical_items_html = ""
        for f in ordered_macro_list:
            pct_f = f["pct"] * 100
            if pct_f > 0:
                tc = on_color(f["macro"]["rgb"])
                label = f'{pct_f:.1f}%' if pct_f >= 6 else ""
                vertical_items_html += f'<div style="background:{f["macro"]["hex"]}; flex:{pct_f}; width:100%; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600; color:{tc};" title="[{f["macro"]["category"]}] {f["macro"]["hex"]} ({pct_f:.1f}%)">{label}</div>'

        # Nhúng cấu trúc Flexbox: Đảm bảo thanh màu co giãn tự động theo chiều cao ảnh gốc bên cạnh
        side_by_side_layout = f"""
        <div style="display: flex; gap: 16px; align-items: stretch; width: 100%; margin-top: 5px;">
            <div style="flex: 1; display: flex; flex-direction: column;">
                <div style="font-size:.65rem;font-weight:600;color:var(--t3);text-transform:uppercase;margin-bottom:6px;letter-spacing:0.05em;">Ảnh gốc</div>
                <img src="data:image/jpeg;base64,{img_b64}" style="width:100%; height:auto; display:block; border:1px solid var(--line);" />
            </div>
            <div style="flex: 1; display: flex; flex-direction: column;">
                <div style="font-size:.65rem;font-weight:600;color:var(--t3);text-transform:uppercase;margin-bottom:6px;letter-spacing:0.05em;">Thanh tỷ lệ đứng</div>
                <div style="display:flex; flex-direction:column; width:100%; flex:1; border:1px solid var(--line); overflow:hidden;">
                    {vertical_items_html}
                </div>
            </div>
        </div>
        """
        st.markdown(side_by_side_layout, unsafe_allow_html=True)

        # Chips tóm tắt phần trăm diện tích
        chips = ""
        for rgb_c, lbl, p, n in [(ad, "Chính", pd, len(dom)), (as_, "Phụ", ps, len(sec)), (aa, "Nhấn", pa, len(acc))]:
            chips += (
                f'<div style="display:flex;align-items:center;gap:6px;background:#fff;border:1px solid var(--line);padding:5px 11px;">'
                f'<div style="width:8px;height:8px;background:{rgb_to_hex(rgb_c)};"></div>'
                f'<span style="font-size:.68rem;font-weight:500;color:var(--t2);">'
                f'{lbl} {p:.1f}% <span style="color:var(--t3);">({n} sắc độ)</span></span></div>'
            )
        st.markdown(f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:.7rem;">{chips}</div>', unsafe_allow_html=True)

    else:
        st.markdown("""<div style="background:#fff;border:1px solid var(--line);padding:4.5rem 1.5rem;text-align:center;margin-top:.5rem;">
        <div style="font-size:0.9rem;color:var(--t2);">Chưa có ảnh</div>
        <div style="font-size:.75rem;color:var(--t3);margin-top:.4rem;">Tải ảnh hoặc dán từ Clipboard để bắt đầu phân tích</div>
        </div>""", unsafe_allow_html=True)


# ─── CỘT PHẢI: CHI TIẾT DẢI MÀU & NÚT BẤM SÁT CẠNH NHAU ───────────────────────
with col_right:
    if image is not None and dom:
        gs = [
            {"name": "Màu Chính", "sub": f"Chủ đạo ({len(dom)} sắc độ)", "key": "dom", "items": dom, "total_pct": pd, "avg_hex": rgb_to_hex(ad), "label_short": "Chính"},
            {"name": "Màu Phụ", "sub": f"Chuyển tiếp ({len(sec)} sắc độ)", "key": "sec", "items": sec, "total_pct": ps, "avg_hex": rgb_to_hex(as_), "label_short": "Phụ"},
            {"name": "Màu Nhấn", "sub": f"Tương phản ({len(acc)} sắc độ)", "key": "acc", "items": acc, "total_pct": pa, "avg_hex": rgb_to_hex(aa), "label_short": "Nhấn"},
        ]

        # ── A · BÁNH XE MÀU SẮC ──
        wc, lc = st.columns([1, 1.3], gap="large")
        all_pts = dom + sec + acc
        wimg = make_wheel(all_pts, size=1200, out=300)

        with wc:
            st.markdown('<div style="font-size:.64rem;font-weight:600;letter-spacing:.13em;text-transform:uppercase;color:var(--t3);margin-bottom:.4rem;">Bánh xe màu sắc</div>', unsafe_allow_html=True)
            st.markdown(f'<img src="data:image/png;base64,{img_to_b64(wimg, "PNG")}" style="width:210px;max-width:100%;display:block;" />', unsafe_allow_html=True)
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            buf_w = io.BytesIO()
            make_wheel(all_pts, 1200, 1200).save(buf_w, "PNG")

            wcol1, wcol2 = st.columns([1, 1])
            with wcol1: st.download_button("📥 TẢI VỀ", data=buf_w.getvalue(), file_name="color_wheel.png", mime="image/png", key="dl_wheel", use_container_width=True)
            with wcol2:
                if st.button("📋 COPY", key="copy_wheel", use_container_width=True):
                    if copy_bytes_to_clipboard(buf_w.getvalue()): st.toast("Đã copy bánh xe màu!", icon="📋")

        with lc:
            st.markdown('<div style="font-size:.64rem;font-weight:600;letter-spacing:.13em;text-transform:uppercase;color:var(--t3);margin-bottom:.5rem;">Màu đại diện nhóm</div>', unsafe_allow_html=True)
            for g in gs:
                st.markdown(f"""
                <div style="display:flex;align-items:stretch;margin-bottom:6px;border:1px solid var(--line);overflow:hidden;height:60px;">
                    <div style="width:55px;background:{g['avg_hex']};flex-shrink:0;"></div>
                    <div style="padding:6px 12px;display:flex;flex-direction:column;justify-content:center;background:#fff;flex:1;">
                        <div style="font-size:.82rem;font-weight:600;color:var(--t1);">{g['name']} <span style="font-size:.7rem;font-weight:400;color:var(--t3);margin-left:4px;">{g['sub']}</span></div>
                        <div style="font-size:.7rem;color:var(--t3);">{g['avg_hex']} &nbsp;·&nbsp; <strong style="color:var(--t1);">{g['total_pct']:.1f}%</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── B · CÁC THANH TỶ LỆ PHÂN BỐ NGANG ──
        st.markdown('<div style="font-size:.65rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--t3);margin-bottom:.75rem;">Dải phân bố ngang</div>', unsafe_allow_html=True)

        # 1. Thanh Tổng Quát Khái Quát (Ngang)
        mcol1, mcol2, mcol3 = st.columns([4.0, 1.2, 1.2])
        with mcol1: st.markdown('<div style="font-size:.82rem;font-weight:600;color:var(--t1);margin-top:0.3rem;">Thanh tỷ lệ tổng quát</div>', unsafe_allow_html=True)
        with mcol2: st.download_button("📥 TẢI VỀ", data=macro_png_tineye(ordered_macro_list), file_name="bar_macro.png", mime="image/png", key="dl_macro", use_container_width=True)
        with mcol3:
            if st.button("📋 COPY", key="copy_macro", use_container_width=True):
                if copy_bytes_to_clipboard(macro_png_tineye(ordered_macro_list)): st.toast("Đã copy dải màu tổng quát!", icon="📋")

        bar_macro_html = '<div style="display:flex;width:100%;height:32px;overflow:hidden;border:1px solid var(--line);margin:0.4rem 0 1.2rem 0;">'
        for f in ordered_macro_list:
            pct_f = f["pct"] * 100
            if pct_f > 0:
                tc = on_color(f["macro"]["rgb"])
                label = f'{pct_f:.1f}%' if pct_f >= 5 else ""
                bar_macro_html += f'<div style="background:{f["macro"]["hex"]};width:{pct_f}%;height:100%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:{tc};" title="{f["macro"]["hex"]} ({pct_f:.1f}%)">{label}</div>'
        bar_macro_html += '</div>'
        st.markdown(bar_macro_html, unsafe_allow_html=True)


        # 2. Thanh Tổng Chi Tiết (Đã sửa lỗi định nghĩa master_detail)
        master_detail = dom + sec + acc # Khai báo lại rõ ràng chuỗi mượt trước khi render
        
        dcol1, dcol2, dcol3 = st.columns([4.0, 1.2, 1.2])
        with dcol1: st.markdown(f'<div style="font-size:.82rem;font-weight:600;color:var(--t1);margin-top:0.3rem;">Thanh dải sắc độ chi tiết</div>', unsafe_allow_html=True)
        with dcol2: st.download_button("📥 TẢI VỀ", data=bar_png(master_detail, h=30), file_name="bar_detail.png", mime="image/png", key="dl_detail", use_container_width=True)
        with dcol3:
            if st.button("📋 COPY", key="copy_detail", use_container_width=True):
                if copy_bytes_to_clipboard(bar_png(master_detail, h=30)): st.toast("Đã copy dải màu chi tiết!", icon="📋")

        bar_detail_html = '<div style="display:flex;width:100%;height:16px;overflow:hidden;border:1px solid var(--line);margin:0.4rem 0 1.5rem 0;">'
        tot_m = sum(it["pct"] for it in master_detail)
        for it in master_detail:
            pv = (it["pct"] / tot_m * 100)
            bar_detail_html += f'<div style="background:{it["hex"]};width:{pv}%;height:100%;" title="{it["hex"]} ({it["pct"]*100:.1f}%)"></div>'
        bar_detail_html += '</div>'
        st.markdown(bar_detail_html, unsafe_allow_html=True)

        st.markdown("---")

        # ── C · THẺ HIỂN THỊ SWATCHES TỪNG PHÂN VÙNG BẬC THANG ─────────────────
        st.markdown('<div style="font-size:.65rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--t3);margin-bottom:1rem;">Chi tiết từng phân vùng bậc thang</div>', unsafe_allow_html=True)

        def render_group(g):
            items_sorted_by_pct = sorted(g["items"], key=lambda x: x["pct"], reverse=True)
            items_smooth_hue = g["items"]

            gcol1, gcol2, gcol3 = st.columns([4.0, 1.2, 1.2])
            with gcol1:
                st.markdown(f'<div style="display:flex;align-items:baseline;gap:8px;margin-top:0.2rem;"><span style="font-size:1rem;font-weight:600;color:var(--t1);">{g["name"]}</span><span style="font-size:.75rem;color:var(--t3);">{g["sub"]}</span><span style="font-size:.9rem;font-weight:600;color:var(--t1);margin-left:6px;">{g["total_pct"]:.1f}%</span></div>', unsafe_allow_html=True)
            with gcol2: st.download_button("📥 TẢI VỀ", data=bar_png(items_smooth_hue), file_name=f'bar_{g["key"]}.png', mime="image/png", key=f'dl_g_{g["key"]}', use_container_width=True)
            with gcol3:
                if st.button("📋 COPY", key=f'copy_g_{g["key"]}', use_container_width=True):
                    if copy_bytes_to_clipboard(bar_png(items_smooth_hue)): st.toast(f"Đã sao chép dải màu nhóm!", icon="📋")

            # Thanh phổ ngang riêng của nhóm màu
            bh = '<div style="display:flex;width:100%;height:26px;overflow:hidden;border:1px solid var(--line);margin:0.4rem 0 .8rem 0;">'
            gtot = sum(i["pct"] for i in items_smooth_hue)
            for it in items_smooth_hue:
                w = (it["pct"] / gtot * 100) if gtot > 0 else (100 / len(items_smooth_hue))
                bh += f'<div style="background:{it["hex"]};width:{w}%;height:100%;"></div>'
            bh += '</div>'
            st.markdown(bh, unsafe_allow_html=True)

            # Grid bảng mã màu thẻ
            CPR = 5
            for ri in range(0, len(items_sorted_by_pct), CPR):
                chunk = items_sorted_by_pct[ri:ri + CPR]
                cols  = st.columns(CPR)
                for ci, it in enumerate(chunk):
                    pv = it["pct"] * 100
                    with cols[ci]:
                        st.markdown(
                            f'<div style="background:{it["hex"]};height:20px;border:1px solid rgba(0,0,0,.06);margin-bottom:3px;"></div>'
                            f'<div style="font-size:.65rem;text-align:center;font-weight:600;color:var(--t2);line-height:1.1;">{pv:.1f}%</div>'
                            f'<div style="font-size:.58rem;text-align:center;color:var(--t3);font-family:monospace;">{it["hex"]}</div>',
                            unsafe_allow_html=True
                        )
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        for idx, g in enumerate(gs):
            render_group(g)
            if idx < len(gs) - 1:
                st.markdown('<div style="border-top:1px solid var(--line);margin-bottom:1rem;"></div>', unsafe_allow_html=True)

    else:
        st.markdown("""<div style="background:#fff;border:1px solid var(--line);padding:5rem 2rem;text-align:center;margin-top:2rem;">
        <div style="font-size:1.4rem;font-weight:300;color:var(--t3);">Chưa có ảnh để phân tích</div>
        <div style="font-size:.8rem;color:var(--t3);margin-top:.5rem;">Tải ảnh hoặc dán ảnh từ Clipboard để bắt đầu</div>
        </div>""", unsafe_allow_html=True)