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

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans+Flex:opsz,wght@8..144,100..900&display=swap');

*, *::before, *::after {
    box-sizing: border-box;
}

/* Force Google Sans Flex font globally */
html, body, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span, 
p, h1, h2, h3, h4, h5, h6, li, label, table, td, th, summary, [data-testid="stExpander"] summary {
    font-family: "Google Sans Flex", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
button div p, button div, .stDownloadButton button, [data-testid="stSidebar"] * {
    font-family: "Google Sans Flex", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Minimalist Black & White / Grayscale Palette */
:root {
    --bg:      #F5F5F7;  /* Cool light grey */
    --surface: #FFFFFF;  /* White */
    --line:    #E5E5EA;  /* Muted cool grey border */
    --line-s:  #F2F2F7;
    --t1:      #1D1D1F;  /* Deep rich black */
    --t2:      #515154;  /* Deep charcoal gray */
    --t3:      #86868B;  /* Muted gray */
    --warm:    #1D1D1F;  /* Black accent */
    --warm-bg: #E8E8ED;  /* Hover gray background */
}

.stApp                    { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.block-container          { padding: 2.5rem 2.75rem 5rem !important; max-width: 1400px !important; }

/* Sidebar */
[data-testid="stSidebar"]            { background: var(--surface) !important; border-right: 1px solid var(--line) !important; box-shadow: none !important; }
[data-testid="stSidebar"] > div      { padding: 2.25rem 1.75rem !important; }

/* Remove Streamlit default image styling decorations */
[data-testid="stImage"],
[data-testid="stImage"] > *,
[data-testid="stImage"] img,
figure, figure img                   { border: none !important; border-radius: 0 !important; box-shadow: none !important; outline: none !important; display: block !important; line-height: 0 !important; }

/* Clean File Uploader */
[data-testid="stFileUploader"]       { background: var(--surface) !important; border: 1px dashed var(--line) !important; border-radius: 4px !important; padding: 1.75rem !important; }
[data-testid="stFileUploader"]:hover { border-color: var(--warm) !important; }
[data-testid="stFileUploadDropzone"] p {
    color: var(--t2) !important;
    font-family: "Google Sans Flex", sans-serif !important;
}

/* Download & Regular Buttons - Monochrome minimalist button with 32px height */
.stDownloadButton, .stButton {
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}
.stDownloadButton > button, .stButton > button { 
    background: var(--surface) !important; 
    border: 1px solid var(--line) !important; 
    border-radius: 4px !important; 
    color: var(--t1) !important; 
    font-size: 0.72rem !important; 
    font-weight: 600 !important; 
    padding: 0 10px !important; 
    width: 100% !important; 
    height: 32px !important; 
    min-height: 32px !important;
    line-height: 1 !important;
    box-shadow: none !important; 
    transition: all .2s cubic-bezier(0.16, 1, 0.3, 1) !important; 
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.stDownloadButton > button:hover, .stButton > button:hover { 
    border-color: var(--t1) !important; 
    color: var(--surface) !important; 
    background: var(--t1) !important; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important; 
}

/* Expanders */
[data-testid="stExpander"]           { background: transparent !important; border: none !important; border-bottom: 1px solid var(--line-s) !important; border-radius: 0 !important; box-shadow: none !important; }
[data-testid="stExpander"] summary   { color: var(--t2) !important; font-size: 0.83rem !important; font-weight: 500 !important; padding: 0.65rem 0 !important; }
[data-testid="stExpander"] > div     { padding: 0.25rem 0 0.8rem !important; }

/* HR */
hr { border: none !important; border-top: 1px solid var(--line) !important; margin: 1.75rem 0 !important; }

/* Global text styling */
h1                          { font-size: 2.25rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important; color: var(--t1) !important; line-height: 1.1 !important; margin: 0 !important; }
p, li, label, td, th, span { color: var(--t2) !important; font-size: 0.875rem !important; line-height: 1.6 !important; }
code                        { font-family: "SF Mono", "Courier New", monospace !important; background: transparent !important; color: var(--t3) !important; font-size: 0.68rem !important; border: none !important; padding: 0 !important; }

/* Misc overrides */
[data-testid="column"]              { padding: 0 0.35rem !important; }
::-webkit-scrollbar                 { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb           { background: var(--line); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def rgb_to_hex(rgb): return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

def img_to_b64(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def lum(rgb): return 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
def on_color(rgb): return "#1D1D1F" if lum(rgb) > 155 else "#FFFFFF"

def get_color_name(rgb):
    r,g,b = [x/255 for x in rgb]
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    d = h*360
    if l<0.12: return "Đen"
    if l>0.88 and s<0.15: return "Trắng"
    if s<0.12: return "Xám" if l>0.5 else "Xám tối"
    if 30<=d<60 and 0.1<=s<0.35 and l>0.65: return "Kem"
    if 10<=d<45 and l<0.45: return "Nâu"
    if 320<=d<350 and l>0.5: return "Hồng"
    if d<15 or d>=345: return "Đỏ"
    if 15<=d<45: return "Cam"
    if 45<=d<75: return "Vàng"
    if 75<=d<160: return "Xanh lá"
    if 160<=d<195: return "Xanh ngọc"
    if 195<=d<250: return "Xanh dương"
    if 250<=d<285: return "Xanh tím"
    if 285<=d<320: return "Tím"
    return "Khác"

def fam(n):
    if n in ["Đỏ","Hồng"]: return "đỏ"
    if n in ["Cam","Nâu","Kem"]: return "cam"
    if n in ["Xanh dương","Xanh tím"]: return "xanh_tím"
    return n

def sort_hue(lst):
    if not lst: return []
    g = {}
    for c in lst: g.setdefault(fam(c["name"]), []).append(c)
    for k in g: g[k].sort(key=lambda x: x["pct"], reverse=True)
    out = []
    for _, v in sorted(g.items(), key=lambda x: x[1][0]["pct"], reverse=True):
        out.extend(v)
    return out

def analyze(rgb, pct):
    t = tuple(int(x) for x in rgb)
    r,g,b = [x/255 for x in t]
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    return dict(rgb=t, hex=rgb_to_hex(t), pct=pct, h=h, l=l, s=s,
                is_neutral=(s<0.22 or l<0.15 or l>0.88), name=get_color_name(t))

def wavg(grp):
    if not grp: return (128,128,128)
    tot = sum(i["pct"] for i in grp)
    return tuple(int(sum(i["rgb"][j]*i["pct"] for i in grp)/tot) for j in range(3))

def bar_png(items, w=1400, h=100):
    img = Image.new("RGB",(w,h),(245,245,247))
    d = ImageDraw.Draw(img); x=0
    tot = sum(i["pct"] for i in items)
    for it in items:
        bw=int(it["pct"]/tot*w); d.rectangle([x,0,x+bw,h],fill=it["rgb"]); x+=bw
    if x<w: d.rectangle([x,0,w,h],fill=items[-1]["rgb"])
    buf=io.BytesIO(); img.save(buf,"PNG"); return buf.getvalue()

def macro_png(grps, w=1400, h=100):
    img = Image.new("RGB",(w,h),(245,245,247))
    d = ImageDraw.Draw(img); x=0
    for g in grps:
        if g["total_pct"]>0:
            bw=int(g["total_pct"]/100*w); d.rectangle([x,0,x+bw,h],fill=g["avg_rgb"]); x+=bw
    if x<w: d.rectangle([x,0,w,h],fill=grps[-1]["avg_rgb"])
    buf=io.BytesIO(); img.save(buf,"PNG"); return buf.getvalue()

def copy_bytes_to_clipboard(png_bytes):
    try:
        import subprocess
        temp_path = os.path.join(os.path.dirname(__file__), "temp_clip.png")
        with open(temp_path, "wb") as f:
            f.write(png_bytes)
        script = f'set the clipboard to (read (POSIX file "{temp_path}") as «class PNGf»)'
        res = subprocess.run(["osascript", "-e", script], capture_output=True)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return res.returncode == 0
    except Exception:
        return False

@st.cache_data
def wheel_base(size=1200):
    x=np.linspace(-1,1,size); y=np.linspace(-1,1,size)
    xv,yv=np.meshgrid(x,y); r=np.sqrt(xv**2+yv**2); theta=np.arctan2(yv,xv)
    h=(theta+np.pi)/(2*np.pi); s=np.clip(r,0,1); v=np.ones_like(r)
    h6=h*6; i=h6.astype(int); f=h6-i
    p=v*(1-s); q=v*(1-s*f); t=v*(1-s*(1-f))
    rgb=np.zeros((size,size,3),dtype=np.uint8)
    for k,(ri,gi,bi) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        idx=(i%6==k); rgb[idx]=np.stack([ri[idx],gi[idx],bi[idx]],axis=-1)*255
    rgba=np.zeros((size,size,4),dtype=np.uint8)
    rgba[:,:,:3]=rgb; rgba[:,:,3]=(r<=1).astype(np.uint8)*255
    return Image.fromarray(rgba,"RGBA")

def make_wheel(pts, size=1200, out=320):
    wimg=wheel_base(size)
    bg=Image.new("RGBA",(size,size),(245,245,247,255))
    bg.paste(wimg,(0,0),wimg); wimg=bg
    d=ImageDraw.Draw(wimg); cx=cy=size/2; mr=(size/2)*.93; sc=size/800
    ring=int(19*sc); dot=int(13*sc)
    for it in sorted(pts,key=lambda x:x["pct"]):
        rc,gc,bc=it["rgb"]
        h,s,_=colorsys.rgb_to_hsv(rc/255,gc/255,bc/255)
        ang=h*2*np.pi-np.pi
        x=cx+s*mr*np.cos(ang); y=cy+s*mr*np.sin(ang)
        d.ellipse([x-ring,y-ring,x+ring,y+ring],fill=(255,255,255,240))
        d.ellipse([x-dot,y-dot,x+dot,y+dot],fill=(rc,gc,bc,255))
    return wimg.resize((out,out),resample=RESAMPLE)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="font-family: 'Google Sans Flex', sans-serif; font-size:1.35rem;font-weight:700;color:#1D1D1F;letter-spacing:-0.025em;line-height:1.15;">
            Beyond<br><span style="color:#000000;font-weight:800;">Photography</span>
        </div>
        <div style="margin-top:0.4rem;font-size:0.66rem;font-weight:600;letter-spacing:0.15em;
                    text-transform:uppercase;color:#86868B;">Color Analysis System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<p style="font-size:0.84rem;color:var(--t2);line-height:1.65;margin-bottom:1rem;">
    Phối hợp màu theo tỷ lệ vàng <strong style="color:var(--t1);">60 – 30 – 10</strong>
    để đạt hiệu quả thị giác tốt nhất.
    </p>""", unsafe_allow_html=True)

    with st.expander("Màu Chính — Dominant"):
        st.write("Chiếm diện tích lớn nhất. Thiết lập tone và cảm xúc tổng thể cho tác phẩm.")
    with st.expander("Màu Phụ — Secondary"):
        st.write("Hỗ trợ màu chính, tạo chiều sâu và cân bằng thị giác.")
    with st.expander("Màu Nhấn — Accent"):
        st.write("Tương phản cao, dùng cho chi tiết nhỏ. Thu hút ánh nhìn, làm nổi bật điểm quan trọng.")

    st.markdown("---")
    st.markdown("""
    <div style="margin-bottom:.5rem;font-size:0.64rem;font-weight:600;letter-spacing:.13em;
                text-transform:uppercase;color:var(--t3);">Tỷ lệ tham chiếu</div>
    <div style="display:flex;flex-direction:column;gap:9px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:60%;height:3px;background:var(--t1);"></div>
            <span style="font-size:0.75rem;color:var(--t2);font-weight:500;">60% Chính</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:30%;height:3px;background:var(--t3);"></div>
            <span style="font-size:0.75rem;color:var(--t2);font-weight:500;">30% Phụ</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:10%;height:3px;background:var(--line);"></div>
            <span style="font-size:0.75rem;color:var(--t2);font-weight:500;">10% Nhấn</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<h1>Beyond Photography</h1>
<div style="margin-top:.35rem;font-size:.68rem;font-weight:600;letter-spacing:.2em;
            text-transform:uppercase;color:var(--t3);">Hệ thống phân tích màu sắc</div>
""", unsafe_allow_html=True)
st.markdown("---")

col_left, col_right = st.columns([1, 2.2], gap="large")
dom = sec = acc = []

# ─── LEFT ─────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: var(--t2); margin-bottom: 0.3rem;'>Tải hình ảnh</p>", unsafe_allow_html=True)
    up = st.file_uploader("upload", type=["jpg","jpeg","png"], label_visibility="collapsed")
    
    st.markdown("<div style='text-align: center; margin: 0.25rem 0; font-size: 0.72rem; color: var(--t3);'>hoặc</div>", unsafe_allow_html=True)
    
    if st.button("📋 Dán ảnh từ Clipboard", use_container_width=True):
        try:
            from PIL import ImageGrab
            clipboard_image = ImageGrab.grabclipboard()
            if clipboard_image is not None:
                if isinstance(clipboard_image, list) and len(clipboard_image) > 0:
                    first_file = clipboard_image[0]
                    try:
                        st.session_state["clipboard_image"] = Image.open(first_file)
                        st.toast("Đã dán ảnh từ tệp tin Clipboard!", icon="📋")
                    except Exception:
                        st.warning("Tệp tin trong clipboard không phải là hình ảnh hợp lệ.")
                else:
                    st.session_state["clipboard_image"] = clipboard_image
                    st.toast("Đã dán ảnh từ Clipboard thành công!", icon="📋")
            else:
                st.warning("Không tìm thấy hình ảnh nào trong Clipboard. Hãy copy ảnh trước.")
        except Exception as e:
            st.error(f"Không thể đọc clipboard: {e}")

    st.markdown("<div style='font-size:.7rem;color:var(--t3);text-align:center;margin-top:0.4rem;'>JPG · PNG</div>", unsafe_allow_html=True)

    image = None
    if up: 
        image = Image.open(up)
        if "clipboard_image" in st.session_state:
            del st.session_state["clipboard_image"]
    elif "clipboard_image" in st.session_state:
        image = st.session_state["clipboard_image"]

    if image:
        # K-means
        arr = np.array(image.resize((120,120)))
        px  = arr[:,:,:3].reshape(-1,3) if arr.shape[-1]==4 else arr.reshape(-1,3)
        km  = KMeans(n_clusters=20, random_state=42, n_init=10)
        lbs = km.fit_predict(px)
        clr = km.cluster_centers_.astype(int)
        pct = np.bincount(lbs, minlength=20)/len(px)
        dl  = [analyze(clr[i], pct[i]) for i in range(20)]

        # Smart split: Dominant · Accent · Secondary
        by_pct = sorted(dl, key=lambda x:x["pct"], reverse=True)
        dom_g  = []
        cum    = 0.0
        for c in by_pct:
            if len(dom_g)<1 or (cum<0.55 and len(dom_g)<4):
                dom_g.append(c); cum+=c["pct"]
        dom_hue = sum(c["h"]*c["pct"] for c in dom_g)/sum(c["pct"] for c in dom_g)
        rest = [c for c in by_pct if c not in dom_g]
        scored = []
        for c in rest:
            vib = c["s"]*(1-abs(2*c["l"]-1))
            hc  = min(abs(c["h"]-dom_hue), 1-abs(c["h"]-dom_hue))
            scored.append((vib*(1+3.5*hc)/(c["pct"]+.01), c))
        scored.sort(key=lambda x:x[0], reverse=True)
        acc_g=[]; acp=0.0
        for sc,c in scored:
            if c["s"]>=.15 and c["pct"]<.06 and acp+c["pct"]<=.12 and len(acc_g)<5:
                acc_g.append(c); acp+=c["pct"]
        if not acc_g: acc_g=sorted(rest, key=lambda x:x["pct"])[:3]
        sec_g=[c for c in rest if c not in acc_g]

        dom = sort_hue(dom_g)
        sec = sort_hue(sec_g)
        acc = sort_hue(acc_g)

        st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)

        # Render image via HTML to bypass Streamlit border injection
        thumb = image.copy(); thumb.thumbnail((900,1400),RESAMPLE)
        st.markdown(
            f'<img src="data:image/jpeg;base64,{img_to_b64(thumb)}" '
            f'style="width:100%;display:block;border:1px solid var(--line);" />',
            unsafe_allow_html=True
        )

        # Summary chips
        ad=wavg(dom); as_=wavg(sec); aa=wavg(acc)
        pd=sum(i["pct"] for i in dom)*100
        ps=sum(i["pct"] for i in sec)*100
        pa=sum(i["pct"] for i in acc)*100
        chips=""
        for rgb_c,lbl,p in [(ad,"Chính",pd),(as_,"Phụ",ps),(aa,"Nhấn",pa)]:
            chips+=(f'<div style="display:flex;align-items:center;gap:6px;background:#fff;'
                    f'border:1px solid var(--line);padding:5px 11px;">'
                    f'<div style="width:8px;height:8px;background:{rgb_to_hex(rgb_c)};"></div>'
                    f'<span style="font-size:.68rem;font-weight:500;color:var(--t2);">'
                    f'{lbl} {p:.0f}%</span></div>')
        st.markdown(f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:.9rem;">{chips}</div>', unsafe_allow_html=True)

    else:
        st.markdown("""<div style="background:#fff;border:1px solid var(--line);padding:3.5rem 1.5rem;
        text-align:center;margin-top:.5rem;">
        <div style="font-size:1rem;font-weight:400;color:var(--t2);">Chưa có ảnh</div>
        <div style="font-size:.78rem;color:var(--t3);margin-top:.4rem;">Tải ảnh hoặc dán ảnh từ Clipboard để bắt đầu</div>
        </div>""", unsafe_allow_html=True)


# ─── RIGHT ────────────────────────────────────────────────────────────────────
with col_right:
    if image is not None and dom:

        pd=sum(i["pct"] for i in dom)*100
        ps=sum(i["pct"] for i in sec)*100
        pa=sum(i["pct"] for i in acc)*100
        ad=wavg(dom); as_=wavg(sec); aa=wavg(acc)

        groups=[
            {"name":"Màu Chính","sub":"Chủ đạo","key":"dom",
             "items":dom,"total_pct":pd,"avg_rgb":ad,"avg_hex":rgb_to_hex(ad),"label_short":"Chính"},
            {"name":"Màu Phụ","sub":"Trung tính","key":"sec",
             "items":sec,"total_pct":ps,"avg_rgb":as_,"avg_hex":rgb_to_hex(as_),"label_short":"Phụ"},
            {"name":"Màu Nhấn","sub":"Điểm nhấn","key":"acc",
             "items":acc,"total_pct":pa,"avg_rgb":aa,"avg_hex":rgb_to_hex(aa),"label_short":"Nhấn"},
        ]
        gs = sorted(groups, key=lambda x:x["total_pct"], reverse=True)

        # Section eyebrow
        st.markdown('<div style="font-size:.65rem;font-weight:600;letter-spacing:.15em;'
                    'text-transform:uppercase;color:var(--t3);margin-bottom:1.25rem;">Phân tích hệ màu</div>',
                    unsafe_allow_html=True)

        # ── A · WHEEL + LEGEND ────────────────────────────────────────────────
        wc, lc = st.columns([1, 1.4], gap="large")
        all_pts = dom+sec+acc
        wimg = make_wheel(all_pts, size=1200, out=300)

        with wc:
            st.markdown('<div style="font-size:.64rem;font-weight:600;letter-spacing:.13em;'
                        'text-transform:uppercase;color:var(--t3);margin-bottom:.6rem;">Bánh xe màu sắc</div>',
                        unsafe_allow_html=True)
            # HTML img — no Streamlit border override
            st.markdown(
                f'<img src="data:image/png;base64,{img_to_b64(wimg,"PNG")}" '
                f'style="width:300px;max-width:100%;display:block;" />',
                unsafe_allow_html=True
            )
            st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
            buf_w=io.BytesIO(); make_wheel(all_pts,1200,1200).save(buf_w,"PNG")
            
            wcol1, wcol2 = st.columns([1, 1])
            with wcol1:
                st.download_button("↓ Tải bánh xe", data=buf_w.getvalue(),
                                   file_name="color_wheel.png", mime="image/png", key="dl_wheel")
            with wcol2:
                if st.button("📋 Copy bánh xe", key="copy_wheel"):
                    if copy_bytes_to_clipboard(buf_w.getvalue()):
                        st.toast("Đã copy bánh xe màu vào Clipboard!", icon="📋")
                    else:
                        st.error("Không thể copy.")

        with lc:
            st.markdown('<div style="font-size:.64rem;font-weight:600;letter-spacing:.13em;'
                        'text-transform:uppercase;color:var(--t3);margin-bottom:.75rem;">Màu đại diện nhóm</div>',
                        unsafe_allow_html=True)
            for g in gs:
                st.markdown(f"""
                <div style="display:flex;align-items:stretch;gap:0;
                            margin-bottom:8px;border:1px solid var(--line);overflow:hidden;">
                    <div style="width:68px;min-height:68px;flex-shrink:0;background:{g['avg_hex']};"></div>
                    <div style="padding:11px 15px;display:flex;flex-direction:column;
                                justify-content:center;gap:3px;background:#fff;flex:1;">
                        <div style="font-size:.88rem;font-weight:600;color:var(--t1);">{g['name']}</div>
                        <div style="font-size:.7rem;color:var(--t3);letter-spacing:.03em;">
                            <span style="font-family:'Google Sans Flex',sans-serif;">{g['avg_hex']}</span>
                        </div>
                        <div style="font-size:.8rem;font-weight:600;color:var(--t1);">{g['total_pct']:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── B · BARS ──────────────────────────────────────────────────────────
        st.markdown('<div style="font-size:.65rem;font-weight:600;letter-spacing:.15em;'
                    'text-transform:uppercase;color:var(--t3);margin-bottom:1rem;">Thanh tỷ lệ màu</div>',
                    unsafe_allow_html=True)

        # Macro bar Header
        mcol1, mcol2, mcol3 = st.columns([4, 1.2, 1.2])
        with mcol1:
            st.markdown('<div style="font-size:.85rem;font-weight:600;color:var(--t1);margin-top:0.4rem;">Khái quát theo nhóm</div>',
                        unsafe_allow_html=True)
        with mcol2:
            st.download_button("↓ Tải dải màu", data=macro_png(gs),
                               file_name="bar_macro.png", mime="image/png", key="dl_macro")
        with mcol3:
            if st.button("📋 Copy dải màu", key="copy_macro"):
                if copy_bytes_to_clipboard(macro_png(gs)):
                    st.toast("Đã copy dải màu khái quát!", icon="📋")
                else:
                    st.error("Không thể copy.")

        # Macro bar spans 100% width, height=36px
        bar = '<div style="display:flex;width:100%;height:36px;overflow:hidden;border:1px solid var(--line);margin:0.5rem 0 0.75rem 0;">'
        for g in gs:
            if g["total_pct"]>0:
                tc      = on_color(g["avg_rgb"])
                label   = f'{g["label_short"]} {g["total_pct"]:.0f}%' if g["total_pct"]>=8 else ""
                gshort  = g["label_short"]
                ghex    = g["avg_hex"]
                gpct    = g["total_pct"]
                bar+=(f'<div style="background:{ghex};width:{gpct}%;height:100%;'
                      f'display:flex;align-items:center;justify-content:center;'
                      f'font-size:12px;font-weight:600;color:{tc};letter-spacing:.04em;"'
                      f' title="{gshort}: {ghex} ({gpct:.1f}%)">'
                      f'{label}</div>')
        bar+='</div>'
        st.markdown(bar, unsafe_allow_html=True)

        # Ratio info
        ratio = " &nbsp;·&nbsp; ".join(
            f'<span style="color:var(--t1);font-weight:600;">{g["label_short"]}</span>'
            f' <span style="color:var(--t1);">{g["total_pct"]:.1f}%</span>'
            for g in gs)
        st.markdown(f'<div style="font-size:.78rem;color:var(--t2);margin-top:0.1rem;margin-bottom:1.5rem;">{ratio}</div>',
                    unsafe_allow_html=True)

        # Detail bar Header
        master=[]
        for g in gs: master.extend(g["items"])

        dcol1, dcol2, dcol3 = st.columns([4, 1.2, 1.2])
        with dcol1:
            st.markdown('<div style="font-size:.85rem;font-weight:600;color:var(--t1);margin-top:0.4rem;">Chi tiết 20 sắc độ</div>',
                        unsafe_allow_html=True)
        with dcol2:
            st.download_button("↓ Tải dải màu", data=bar_png(master),
                               file_name="bar_detail.png", mime="image/png", key="dl_detail")
        with dcol3:
            if st.button("📋 Copy dải màu", key="copy_detail"):
                if copy_bytes_to_clipboard(bar_png(master)):
                    st.toast("Đã copy dải màu chi tiết!", icon="📋")
                else:
                    st.error("Không thể copy.")

        # Detail bar spans 100% width, height=36px
        bar2='<div style="display:flex;width:100%;height:36px;overflow:hidden;border:1px solid var(--line);margin:0.5rem 0 1.25rem 0;">'
        for it in master:
            r2,g2,b2=it["rgb"]; pv=it["pct"]*100
            bar2+=f'<div style="background:rgb({r2},{g2},{b2});width:{pv}%;height:100%;" title="{it["hex"]} {pv:.1f}%"></div>'
        bar2+='</div>'
        st.markdown(bar2, unsafe_allow_html=True)

        st.markdown("---")

        # ── C · SWATCHES ──────────────────────────────────────────────────────
        st.markdown('<div style="font-size:.65rem;font-weight:600;letter-spacing:.15em;'
                    'text-transform:uppercase;color:var(--t3);margin-bottom:1.25rem;">Dải sắc độ từng nhóm</div>',
                    unsafe_allow_html=True)

        def render_group(g):
            tot   = g["total_pct"]
            items = sorted(g["items"], key=lambda x:x["pct"], reverse=True)

            # Group header with title, download, and copy buttons
            gcol1, gcol2, gcol3 = st.columns([4, 1.2, 1.2])
            with gcol1:
                st.markdown(
                    f'<div style="display:flex;align-items:baseline;gap:10px;margin-top:0.25rem;">'
                    f'<span style="font-size:1.05rem;font-weight:600;color:var(--t1);">{g["name"]}</span>'
                    f'<span style="font-size:.75rem;color:var(--t3);">{g["sub"]}</span>'
                    f'<span style="font-size:.9rem;font-weight:600;color:var(--t1);margin-left:8px;">{tot:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with gcol2:
                st.download_button("↓ Tải dải màu", data=bar_png(items),
                                   file_name=f'bar_{g["key"]}.png', mime="image/png",
                                   key=f'dl_g_{g["key"]}')
            with gcol3:
                if st.button("📋 Copy dải màu", key=f'copy_g_{g["key"]}'):
                    if copy_bytes_to_clipboard(bar_png(items)):
                        st.toast(f"Đã copy dải màu {g['label_short'].lower()}!", icon="📋")
                    else:
                        st.error("Không thể copy.")

            # Group bar — spans 100% width, height=36px
            bh='<div style="display:flex;width:100%;height:36px;overflow:hidden;border:1px solid var(--line);margin:0.5rem 0 1rem 0;">'
            for it in items:
                w=(it["pct"]/(tot/100))*100
                bh+=f'<div style="background:rgb({it["rgb"][0]},{it["rgb"][1]},{it["rgb"][2]});width:{w}%;height:100%;"></div>'
            bh+='</div>'
            st.markdown(bh, unsafe_allow_html=True)

            # Swatch tiles — 8 per row (compacted for secondary hierarchy)
            CPR=8
            for ri in range(0,len(items),CPR):
                chunk=items[ri:ri+CPR]
                cols=st.columns(CPR)
                for ci,it in enumerate(chunk):
                    pv=it["pct"]*100
                    with cols[ci]:
                        st.markdown(
                            f'<div style="background:{it["hex"]};height:24px;'
                            f'border:1px solid rgba(0,0,0,.08);margin-bottom:4px;"></div>'
                            f'<div style="font-size:.65rem;text-align:center;font-weight:600;'
                            f'color:var(--t2);line-height:1.2;">{pv:.1f}%</div>'
                            f'<div style="font-size:.58rem;text-align:center;color:var(--t3);'
                            f'font-family:\'Google Sans Flex\',sans-serif;letter-spacing:-0.01em;white-space:nowrap;overflow:hidden;">{it["hex"]}</div>',
                            unsafe_allow_html=True
                        )
            st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)

        for idx,g in enumerate(gs):
            render_group(g)
            if idx<len(gs)-1:
                st.markdown('<div style="border-top:1px solid var(--line);margin-bottom:1.25rem;"></div>',
                            unsafe_allow_html=True)

    else:
        st.markdown("""<div style="background:#fff;border:1px solid var(--line);padding:5rem 2rem;
        text-align:center;margin-top:2rem;">
        <div style="font-size:1.4rem;font-weight:300;color:var(--t3);">Chưa có ảnh để phân tích</div>
        <div style="font-size:.8rem;color:var(--t3);margin-top:.5rem;">Tải ảnh hoặc dán ảnh từ Clipboard để bắt đầu</div>
        </div>""", unsafe_allow_html=True)