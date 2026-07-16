import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import colorsys
import io
import base64
import os

st.set_page_config(
    page_title="Beyond Photography",
    layout="wide"
)

RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@400;500;600&family=Google+Sans+Flex:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --canvas:           #ffffff;
    --surface-soft:     #f4f4f5;
    --surface-card:     #fafafa;
    --surface-elevated: #f4f4f5;
    --hairline:         #e4e4e7;
    --hairline-strong:  #a1a1aa;
    --ink:              #000000;
    --body:             #18181b;
    --body-strong:      #27272a;
    --muted:            #52525b;
    --muted-soft:       #a1a1aa;
    --link:             #003eff;
}

html, body, [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
p, h1, h2, h3, h4, h5, h6, li, label, table, td, th,
summary, [data-testid="stExpander"] summary,
button div p, button div,
.stDownloadButton button,
[data-testid="stSidebar"] * {
    font-family: "Google Sans Flex", -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp                    { background: var(--canvas) !important; color: var(--body) !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.block-container          { padding: 2.5rem 2.5rem 5rem !important; max-width: 1280px !important; }

[data-testid="stImage"],
[data-testid="stImage"] > *,
[data-testid="stImage"] img,
figure, figure img { border: none !important; border-radius: 0 !important; box-shadow: none !important; outline: none !important; display: block !important; line-height: 0 !important; }

[data-testid="stFileUploader"]       { background: var(--surface-card) !important; border: 1px dashed var(--hairline-strong) !important; border-radius: 0 !important; padding: 1.5rem !important; }
[data-testid="stFileUploader"]:hover { border-color: var(--ink) !important; }

.stDownloadButton, .stButton {
    display: flex !important;
    align-items: stretch !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}
.stDownloadButton > button, .stButton > button {
    background: transparent !important;
    border: 1px solid var(--ink) !important;
    border-radius: 9999px !important;
    color: var(--ink) !important;
    font-family: "IBM Plex Mono", ui-monospace, 'SF Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    padding: 0 16px !important;
    width: 100% !important;
    height: 32px !important;
    min-height: 32px !important;
    line-height: 1 !important;
    box-shadow: none !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    transition: all .2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 5px !important;
    white-space: nowrap !important;
}
.stDownloadButton > button:hover, .stButton > button:hover {
    background: var(--ink) !important;
    color: var(--canvas) !important;
    border-color: var(--ink) !important;
    box-shadow: none !important;
}

[data-testid="column"] > div {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}
[data-testid="column"]  { padding: 0 0.25rem !important; }

hr { border: none !important; border-top: 1px solid var(--hairline) !important; margin: 3rem 0 !important; }

h1 {
    font-family: "Saira Condensed", -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 32px !important;
    font-weight: 400 !important;
    letter-spacing: 2px !important;
    color: var(--ink) !important;
    line-height: 1.2 !important;
    margin: 0 !important;
    text-transform: uppercase !important;
}
p, li, label, td, th, span { color: var(--body) !important; font-size: 16px !important; line-height: 1.5 !important; }

::-webkit-scrollbar       { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: var(--hairline-strong); border-radius: 0; }

[data-testid="stFileUploader"] small { font-size: 0 !important; height: 0 !important; overflow: hidden !important; }
[data-testid="stFileUploader"] small::after { content: "Kéo thả hoặc chọn tệp (JPG, PNG)" !important; font-size: 14px !important; height: auto !important; color: var(--muted) !important; font-family: "Google Sans Flex", sans-serif !important; }
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] span { font-family: "IBM Plex Mono", monospace !important; letter-spacing: 2px !important; text-transform: uppercase !important; }
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
    return "#000000" if lum(rgb) > 155 else "#FFFFFF"

def wavg(grp):
    if not grp: return (128, 128, 128)
    tot = sum(i["pct"] for i in grp)
    if tot == 0: return grp[0]["rgb"]
    return tuple(int(sum(i["rgb"][j] * i["pct"] for i in grp) / tot) for j in range(3))

def color_distance_redmean(rgb1, rgb2):
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    mean_r = (r1 + r2) / 2.0
    delta_r = r1 - r2
    delta_g = g1 - g2
    delta_b = b1 - b2
    weight_r = 2.0 + mean_r / 256.0
    weight_g = 4.0
    weight_b = 2.0 + (255.0 - mean_r) / 256.0
    return np.sqrt(weight_r * delta_r**2 + weight_g * delta_g**2 + weight_b * delta_b**2)


# ─── THUẬT TOÁN PHÂN LOẠI LAI TINH CHỈNH ──────────────────────────────────────
def classify_hybrid(families):
    if not families:
        return families

    families_sorted = sorted(families, key=lambda x: x["pct"], reverse=True)
    
    cum_sum = 0.0
    for f in families_sorted:
        cum_sum += f["pct"]
        if cum_sum <= 0.55:
            f["macro"]["category"] = "Chính"
        elif cum_sum <= 0.88:
            f["macro"]["category"] = "Phụ"
        else:
            f["macro"]["category"] = "Nhấn"

    for f in families_sorted:
        pct = f["pct"]
        s = f["macro"]["s"]
        is_neutral = f["macro"]["is_neutral"]

        # ĐẶC CÁCH MÀU NHẤN: Độc lập rực rỡ (S >= 0.38) diện tích nhỏ (< 12%)
        if not is_neutral and s >= 0.38 and pct < 0.12:
            f["macro"]["category"] = "Nhấn"

        # KHỬ MÀU NHẤN GIẢ: Đuôi xám/trầm (S < 0.38) -> Trả về màu Phụ
        elif f["macro"]["category"] == "Nhấn" and (is_neutral or s < 0.38):
            f["macro"]["category"] = "Phụ"

    all_cats = [f["macro"]["category"] for f in families_sorted]
    if "Chính" not in all_cats:
        families_sorted[0]["macro"]["category"] = "Chính"
    if "Phụ" not in all_cats and len(families_sorted) >= 2:
        for f in families_sorted:
            if f["macro"]["category"] != "Chính":
                f["macro"]["category"] = "Phụ"
                break
    if "Nhấn" not in all_cats and len(families_sorted) >= 3:
        sorted_by_sat = sorted(families_sorted, key=lambda x: (x["macro"]["is_neutral"], -x["macro"]["s"], x["pct"]))
        sorted_by_sat[-1]["macro"]["category"] = "Nhấn"

    return families_sorted


# ─── HÀM TẠO ẢNH XUẤT FILE (DOWNLOAD) ─────────────────────────────────────────
def _theme_bg():
    """Return RGB tuple for canvas background."""
    return (255, 255, 255)

def _theme_ring():
    """Return RGBA for wheel dot rings."""
    return (0, 0, 0, 240)

def bar_png(items, w=1400, h=60):
    img = Image.new("RGB", (w, h), _theme_bg())
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
    img = Image.new("RGB", (w, h), _theme_bg())
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
def wheel_base(size=600):
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

def make_wheel(pts, size=600, out=300):
    wimg = wheel_base(size)
    bg = Image.new("RGBA", (size, size), (*_theme_bg(), 255))
    bg.paste(wimg, (0, 0), wimg)
    wimg = bg
    d = ImageDraw.Draw(wimg)
    cx = cy = size / 2
    mr = (size / 2) * 0.93
    sc = size / 800
    ring = int(19 * sc)
    dot  = int(13 * sc)
    ring_color = _theme_ring()
    for it in sorted(pts, key=lambda x: x["pct"]):
        rc, gc, bc = it["rgb"]
        h, s, _ = colorsys.rgb_to_hsv(rc / 255, gc / 255, bc / 255)
        ang = h * 2 * np.pi - np.pi
        x = cx + s * mr * np.cos(ang)
        y = cy + s * mr * np.sin(ang)
        d.ellipse([x-ring, y-ring, x+ring, y+ring], fill=ring_color)
        d.ellipse([x-dot,  y-dot,  x+dot,  y+dot],  fill=(rc, gc, bc, 255))
    return wimg.resize((out, out), resample=RESAMPLE)


# ─── TIÊU ĐỀ CHÍNH APP ────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
        <div style="font-family:'Saira Condensed',sans-serif;font-size:24px;font-weight:400;
                    color:var(--ink);letter-spacing:1.5px;line-height:1.3;text-transform:uppercase;">
            VISUAL DECONSTRUCTION | BEYOND PHOTOGRAPHY
        </div>
        <div style="margin-top:.5rem;font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:400;
                    letter-spacing:2px;text-transform:uppercase;color:var(--muted);">CẤU TRÚC MÀU SẮC</div>
</div>
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
    st.markdown("<p style='font-family:\"IBM Plex Mono\",monospace;font-size:11px;font-weight:400;color:var(--muted);margin-bottom:0.3rem;letter-spacing:2px;text-transform:uppercase;'>UP ẢNH LÊN</p>", unsafe_allow_html=True)
    up = st.file_uploader("upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")


    image = None
    if up:
        image = Image.open(up)

    if image:
        work = image.convert("RGB")
        # Downsampling xuống 150x150 tăng hiệu năng
        arr  = np.array(work.resize((150, 150), Image.Resampling.BILINEAR))
        px   = arr.reshape(-1, 3)

        # Trích xuất 40 màu cơ sở bằng MiniBatchKMeans siêu tốc
        km_detail = MiniBatchKMeans(n_clusters=40, random_state=42, n_init=1, batch_size=2048)
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
                hue_bin = int(h * 12) % 12
                # Phân rã độ bão hòa (Chroma Splitting) ngăn màu rực rỡ bị lấn át bởi màu trầm
                if s < 0.38:
                    bin_id = hue_bin + 12
                else:
                    bin_id = hue_bin
                
            raw_micro_colors.append({
                "rgb": rgb_t, "hex": rgb_to_hex(rgb_t), "pct": pct_detail[i],
                "h": h, "l": l, "s": s, "is_neutral": is_neutral, "bin_id": bin_id
            })

        # Gộp sắc độ tương cận bằng Redmean
        micro_colors = []
        raw_micro_colors.sort(key=lambda x: x["pct"], reverse=True)
        
        for r_mc in raw_micro_colors:
            found_match = False
            for m_mc in micro_colors:
                dist = color_distance_redmean(r_mc["rgb"], m_mc["rgb"])
                if dist < 45.0 and r_mc["bin_id"] == m_mc["bin_id"]: 
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

        bin_groups = {}
        for mc in micro_colors:
            bid = mc["bin_id"]
            if bid not in bin_groups: bin_groups[bid] = []
            bin_groups[bid].append(mc)

        families = []
        for bid, items in bin_groups.items():
            tot_pct = sum(it["pct"] for it in items)
            
            # Dominant Representative: Lấy sắc độ nổi trội nhất đại diện nhóm
            repr_item = max(items, key=lambda x: x["pct"])
            macro_rgb = repr_item["rgb"]
            
            mr, mg, mb = [x / 255.0 for x in macro_rgb]
            mh, ml, ms = colorsys.rgb_to_hls(mr, mg, mb)
            
            families.append({
                "bin_id": bid, "pct": tot_pct, "items": items,
                "macro": {
                    "rgb": macro_rgb, "hex": rgb_to_hex(macro_rgb), "pct": tot_pct,
                    "h": mh, "l": ml, "s": ms, "is_neutral": items[0]["is_neutral"]
                }
            })

        families_classified = classify_hybrid(families)

        dom_families = [f for f in families_classified if f["macro"]["category"] == "Chính"]
        sec_families = [f for f in families_classified if f["macro"]["category"] == "Phụ"]
        acc_families = [f for f in families_classified if f["macro"]["category"] == "Nhấn"]

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
        
        thumb = work.copy()
        thumb.thumbnail((500, 500), RESAMPLE)
        img_b64 = img_to_b64(thumb)
        
        vertical_items_html = ""
        for f in ordered_macro_list:
            pct_f = f["pct"] * 100
            if pct_f > 0:
                tc = on_color(f["macro"]["rgb"])
                label = f'{pct_f:.1f}%' if pct_f >= 6 else ""
                vertical_items_html += f'<div style="background:{f["macro"]["hex"]}; flex:{pct_f}; width:100%; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:400; color:{tc}; letter-spacing:1px;" title="[{f["macro"]["category"]}] {f["macro"]["hex"]} ({pct_f:.1f}%)">{label}</div>'

        side_by_side_layout = f"""
        <div style="display: flex; gap: 40px; align-items: stretch; width: 100%; margin-top: 5px;">
            <div style="flex: 1; display: flex; flex-direction: column;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:400;color:var(--muted);text-transform:uppercase;margin-bottom:6px;letter-spacing:2px;">ẢNH GỐC</div>
                <img src="data:image/jpeg;base64,{img_b64}" style="width:100%; height:auto; display:block; border:1px solid var(--hairline);" />
            </div>
            <div style="flex: 1; display: flex; flex-direction: column;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:400;color:var(--muted);text-transform:uppercase;margin-bottom:6px;letter-spacing:2px;">THANH TỶ LỆ ĐỨNG</div>
                <div style="display:flex; flex-direction:column; width:100%; flex:1; border:1px solid var(--hairline); overflow:hidden;">
                    {vertical_items_html}
                </div>
            </div>
        </div>
        """
        st.markdown(side_by_side_layout, unsafe_allow_html=True)

        chips = ""
        for rgb_c, lbl, p, n in [(ad, "CHÍNH", pd, len(dom)), (as_, "PHỤ", ps, len(sec)), (aa, "NHẤN", pa, len(acc))]:
            chips += (
                f'<div style="display:flex;align-items:center;gap:6px;background:var(--surface-card);border:1px solid var(--hairline);padding:5px 11px;">'
                f'<div style="width:8px;height:8px;background:{rgb_to_hex(rgb_c)};"></div>'
                f'<span style="font-family:\"IBM Plex Mono\",monospace;font-size:11px;font-weight:400;color:var(--body);letter-spacing:2px;text-transform:uppercase;">'
                f'{lbl} {p:.1f}% <span style="color:var(--muted);">({n} sắc độ)</span></span></div>'
            )
        st.markdown(f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:.7rem;">{chips}</div>', unsafe_allow_html=True)

    else:
        st.markdown("""<div style="background:var(--surface-card);border:1px solid var(--hairline);padding:4.5rem 1.5rem;text-align:center;margin-top:.5rem;">
        <div style="font-family:'Saira Condensed',sans-serif;font-size:16px;font-weight:400;color:var(--ink);letter-spacing:1.5px;text-transform:uppercase;">CHƯA CÓ ẢNH</div>
        <div style="font-family:'Google Sans Flex',sans-serif;font-size:14px;color:var(--muted);margin-top:.4rem;">Tải ảnh hoặc dán từ Clipboard để bắt đầu</div>
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
        wimg_600 = make_wheel(all_pts, size=600, out=300)

        with wc:
            st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;font-weight:400;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem;">BÁNH XE MÀU SẮC</div>', unsafe_allow_html=True)
            st.markdown(f'<img src="data:image/png;base64,{img_to_b64(wimg_600, "PNG")}" style="width:210px;max-width:100%;display:block;" />', unsafe_allow_html=True)
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            buf_w = io.BytesIO()
            wimg_600.save(buf_w, "PNG")

            wcol1, wcol2 = st.columns([1, 1])
            with wcol1: st.download_button("TẢI VỀ", data=buf_w.getvalue(), file_name="color_wheel.png", mime="image/png", key="dl_wheel", use_container_width=True)
            with wcol2:
                if st.button("COPY", key="copy_wheel", use_container_width=True):
                    if copy_bytes_to_clipboard(buf_w.getvalue()): st.toast("Đã copy bánh xe màu!", icon="")

        with lc:
            st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;font-weight:400;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;">MÀU ĐẠI DIỆN NHÓM</div>', unsafe_allow_html=True)
            for g in gs:
                st.markdown(f"""
                <div style="display:flex;align-items:stretch;margin-bottom:6px;border:1px solid var(--hairline);overflow:hidden;height:60px;">
                    <div style="width:55px;background:{g['avg_hex']};flex-shrink:0;"></div>
                    <div style="padding:6px 12px;display:flex;flex-direction:column;justify-content:center;background:var(--surface-card);flex:1;">
                        <div style="font-family:'Saira Condensed',sans-serif;font-size:16px;font-weight:400;color:var(--ink);letter-spacing:1.5px;text-transform:uppercase;">{g['name']} <span style="font-family:'Google Sans Flex',sans-serif;font-size:12px;font-weight:400;color:var(--muted);margin-left:4px;letter-spacing:0;text-transform:none;">{g['sub']}</span></div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);letter-spacing:1px;">{g['avg_hex']} &nbsp;·&nbsp; <span style="color:var(--ink);">{g['total_pct']:.1f}%</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;font-weight:400;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:.75rem;">DẢI PHÂN BÓ NGANG</div>', unsafe_allow_html=True)

        mcol1, mcol2, mcol3 = st.columns([4.0, 1.2, 1.2])
        with mcol1: st.markdown('<div style="font-family:\'Saira Condensed\',sans-serif;font-size:16px;font-weight:400;color:var(--ink);margin-top:0.3rem;letter-spacing:1.5px;text-transform:uppercase;">THANH TỶ LỆ TỔNG QUÁT</div>', unsafe_allow_html=True)
        with mcol2: st.download_button("TẢI VỀ", data=macro_png_tineye(ordered_macro_list), file_name="bar_macro.png", mime="image/png", key="dl_macro", use_container_width=True)
        with mcol3:
            if st.button("COPY", key="copy_macro", use_container_width=True):
                if copy_bytes_to_clipboard(macro_png_tineye(ordered_macro_list)): st.toast("Đã copy dải màu tổng quát!", icon="")

        bar_macro_html = '<div style="display:flex;width:100%;height:32px;overflow:hidden;border:1px solid var(--hairline);margin:0.4rem 0 1.2rem 0;">'
        for f in ordered_macro_list:
            pct_f = f["pct"] * 100
            if pct_f > 0:
                tc = on_color(f["macro"]["rgb"])
                label = f'{pct_f:.1f}%' if pct_f >= 5 else ""
                bar_macro_html += f'<div style="background:{f["macro"]["hex"]};width:{pct_f}%;height:100%;display:flex;align-items:center;justify-content:center;font-family:\'IBM Plex Mono\',monospace;font-size:11px;font-weight:400;color:{tc};letter-spacing:1px;" title="{f["macro"]["hex"]} ({pct_f:.1f}%)">{label}</div>'
        bar_macro_html += '</div>'
        st.markdown(bar_macro_html, unsafe_allow_html=True)


        # 2. Thanh dải sắc độ chi tiết
        master_detail = dom + sec + acc
        
        dcol1, dcol2, dcol3 = st.columns([4.0, 1.2, 1.2])
        with dcol1: st.markdown(f'<div style="font-family:\'Saira Condensed\',sans-serif;font-size:16px;font-weight:400;color:var(--ink);margin-top:0.3rem;letter-spacing:1.5px;text-transform:uppercase;">THANH DẢI SẮC ĐỘ CHI TIẾT</div>', unsafe_allow_html=True)
        with dcol2: st.download_button("TẢI VỀ", data=bar_png(master_detail, h=30), file_name="bar_detail.png", mime="image/png", key="dl_detail", use_container_width=True)
        with dcol3:
            if st.button("COPY", key="copy_detail", use_container_width=True):
                if copy_bytes_to_clipboard(bar_png(master_detail, h=30)): st.toast("Đã copy dải màu chi tiết!", icon="")

        bar_detail_html = '<div style="display:flex;width:100%;height:16px;overflow:hidden;border:1px solid var(--hairline);margin:0.4rem 0 1.5rem 0;">'
        tot_m = sum(it["pct"] for it in master_detail)
        for it in master_detail:
            pv = (it["pct"] / tot_m * 100)
            bar_detail_html += f'<div style="background:{it["hex"]};width:{pv}%;height:100%;" title="{it["hex"]} ({it["pct"]*100:.1f}%)"></div>'
        bar_detail_html += '</div>'
        st.markdown(bar_detail_html, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;font-weight:400;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:1rem;">CHI TIẾT TỪNG PHẦN VÙNG BẬC THANG</div>', unsafe_allow_html=True)

        def render_group(g):
            items_sorted_by_pct = sorted(g["items"], key=lambda x: x["pct"], reverse=True)
            items_smooth_hue = g["items"]

            gcol1, gcol2, gcol3 = st.columns([4.0, 1.2, 1.2])
            with gcol1:
                st.markdown(f'<div style="display:flex;align-items:baseline;gap:8px;margin-top:0.2rem;"><span style="font-family:\'Saira Condensed\',sans-serif;font-size:16px;font-weight:400;color:var(--ink);letter-spacing:1.5px;text-transform:uppercase;">{g["name"]}</span><span style="font-family:\'Google Sans Flex\',sans-serif;font-size:12px;color:var(--muted);letter-spacing:0;text-transform:none;">{g["sub"]}</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:14px;font-weight:400;color:var(--ink);margin-left:6px;letter-spacing:1px;">{g["total_pct"]:.1f}%</span></div>', unsafe_allow_html=True)
            with gcol2: st.download_button("TẢI VỀ", data=bar_png(items_smooth_hue), file_name=f'bar_{g["key"]}.png', mime="image/png", key=f'dl_g_{g["key"]}', use_container_width=True)
            with gcol3:
                if st.button("COPY", key=f'copy_g_{g["key"]}', use_container_width=True):
                    if copy_bytes_to_clipboard(bar_png(items_smooth_hue)): st.toast(f"Đã sao chép dải màu nhóm!", icon="")

            bh = '<div style="display:flex;width:100%;height:26px;overflow:hidden;border:1px solid var(--hairline);margin:0.4rem 0 .8rem 0;">'
            gtot = sum(i["pct"] for i in items_smooth_hue)
            for it in items_smooth_hue:
                w = (it["pct"] / gtot * 100) if gtot > 0 else (100 / len(items_smooth_hue))
                bh += f'<div style="background:{it["hex"]};width:{w}%;height:100%;"></div>'
            bh += '</div>'
            st.markdown(bh, unsafe_allow_html=True)

            CPR = 5
            for ri in range(0, len(items_sorted_by_pct), CPR):
                chunk = items_sorted_by_pct[ri:ri + CPR]
                cols  = st.columns(CPR)
                for ci, it in enumerate(chunk):
                    pv = it["pct"] * 100
                    with cols[ci]:
                        st.markdown(
                            f'<div style="background:{it["hex"]};height:20px;border:1px solid var(--hairline);margin-bottom:3px;"></div>'
                            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;text-align:center;font-weight:400;color:var(--body);line-height:1.1;letter-spacing:1px;">{pv:.1f}%</div>'
                            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;text-align:center;color:var(--muted);letter-spacing:0.5px;">{it["hex"]}</div>',
                            unsafe_allow_html=True
                        )
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        for idx, g in enumerate(gs):
            render_group(g)
            if idx < len(gs) - 1:
                st.markdown('<div style="border-top:1px solid var(--hairline);margin-bottom:1rem;"></div>', unsafe_allow_html=True)

    else:
        st.markdown("""<div style="background:var(--surface-card);border:1px solid var(--hairline);padding:5rem 2rem;text-align:center;margin-top:2rem;">
        <div style="font-family:'Saira Condensed',sans-serif;font-size:24px;font-weight:400;color:var(--ink);letter-spacing:1.5px;text-transform:uppercase;">CHƯA CÓ ẢNH ĐỂ PHÂN TÍCH</div>
        <div style="font-family:'Google Sans Flex',sans-serif;font-size:14px;color:var(--muted);margin-top:.5rem;">Tải ảnh hoặc dán ảnh từ Clipboard để bắt đầu</div>
        </div>""", unsafe_allow_html=True)


# ─── PHẦN CHÂN TRANG (FOOTER) THIẾT KẾ TỐI GIẢN ────────────────────────────────
st.markdown("""
<hr>
<div style="text-align: center; padding: 1.5rem 0 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--muted); letter-spacing: 2px; text-transform: uppercase;">
    Dự án Visual Deconstruction – Beyond Photography · Minh Tuấn · <a href="https://beyondphotography.vn/" target="_blank" style="color: var(--link); text-decoration: none; font-weight: 400; border-bottom: 1px solid var(--hairline);">beyondphotography.vn</a>
</div>
""", unsafe_allow_html=True)