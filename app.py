import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import io

# Cấu hình giao diện chế độ rộng (wide mode)
st.set_page_config(page_title="Phân Tích Hệ Màu 20 Sắc Độ", layout="wide")

# ==========================================
# THANH BÊN (SIDEBAR) - ĐỊNH NGHĨA LÝ THUYẾT
# ==========================================
with st.sidebar:
    st.header("Định nghĩa Màu Chính - Phụ - Nhấn")
    st.write(
        "Trong phối hợp và thiết kế ứng dụng, hệ thống màu sắc thường được phân chia rõ ràng "
        "thành 3 loại vai trò chính để đạt được hiệu quả thị giác tốt nhất:"
    )
    
    st.markdown("---")
    
    st.subheader("Màu Chính (Dominant Color)")
    st.write(
        "Là màu sắc chủ đạo, chiếm diện tích lớn nhất trong thiết kế. Vai trò của nó là thiết lập tone giọng, "
        "cảm xúc chung và làm nền tảng để định hình phong cách, giúp các yếu tố khác nổi bật lên."
    )
    
    st.subheader("Màu Phụ (Secondary Color)")
    st.write(
        "Là màu đi kèm để hỗ trợ cho màu chính. Vai trò của nó là tạo sự phong phú, chiều sâu và tính cân bằng "
        "cho tổng thể, giúp thiết kế không bị đơn điệu hoặc quá trống trải."
    )
    
    st.subheader("Màu Nhấn (Accent Color)")
    st.write(
        "Là màu sắc có tính tương phản mạnh hoặc độ rực cao, được dùng cho các chi tiết nhỏ. Vai trò của nó là "
        "thu hút ánh nhìn đầu tiên, tạo điểm nhấn thị giác và làm nổi bật các thành phần quan trọng nhất."
    )
    
    st.markdown("---")
    st.subheader("Bảng tỷ lệ chuẩn tham chiếu")
    st.markdown(
        "| Loại màu | Tỷ lệ chuẩn | Mục đích sử dụng |\n"
        "| --- | --- | --- |\n"
        "| **Màu Chính** | **60%** | Tạo phông nền chủ đạo |\n"
        "| **Màu Phụ** | **30%** | Tạo sự cân bằng |\n"
        "| **Màu Nhấn** | **10%** | Điểm nhấn thị giác |"
    )

# ==========================================
# CÁC HÀM HỖ TRỢ XỬ LÝ MÀU SẮC & XUẤT ẢNH PNG
# ==========================================
def rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}".upper()

def analyze_color(rgb, pct):
    rgb_tuple = tuple(int(x) for x in rgb)
    r, g, b = [x / 255.0 for x in rgb_tuple]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    is_neutral = (s < 0.22) or (l < 0.15) or (l > 0.88)
    
    return {
        "rgb": rgb_tuple,
        "hex": rgb_to_hex(rgb_tuple),
        "pct": pct,
        "h": h,
        "l": l,
        "s": s,
        "is_neutral": is_neutral
    }

def get_weighted_average_color(group):
    if not group:
        return (128, 128, 128)
    total_pct = sum(item["pct"] for item in group)
    r_avg = sum(item["rgb"][0] * item["pct"] for item in group) / total_pct
    g_avg = sum(item["rgb"][1] * item["pct"] for item in group) / total_pct
    b_avg = sum(item["rgb"][2] * item["pct"] for item in group) / total_pct
    return (int(r_avg), int(g_avg), int(b_avg))

# TẠO ẢNH PNG CHO CÁC THANH TỶ LỆ MÀU 20 SẮC ĐỘ
def generate_bar_png(items, width=1200, height=100):
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    total_pct = sum(item["pct"] for item in items)
    current_x = 0
    for item in items:
        pct = item["pct"] / total_pct
        box_width = int(pct * width)
        draw.rectangle([current_x, 0, current_x + box_width, height], fill=item["rgb"])
        current_x += box_width
    if current_x < width:
        draw.rectangle([current_x, 0, width, height], fill=items[-1]["rgb"])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# TẠO ẢNH PNG CHO THANH MÀU KHÁI QUÁT (THEO THỨ TỰ SẮP XẾP GIẢM DẦN)
def generate_macro_bar_png_dynamic(sorted_groups, width=1200, height=100):
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    current_x = 0
    for g in sorted_groups:
        if g["total_pct"] > 0:
            box_width = int((g["total_pct"] / 100) * width)
            draw.rectangle([current_x, 0, current_x + box_width, height], fill=g["avg_rgb"])
            current_x += box_width
    if current_x < width:
        draw.rectangle([current_x, 0, width, height], fill=sorted_groups[-1]["avg_rgb"])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ==========================================
# PHẦN HIỂN THỊ CHÍNH CỦA ỨNG DỤNG
# ==========================================
st.title("Hệ Thống Phân Tích Hệ Màu Thiết Kế")
st.write("Tải hình ảnh lên để phân tích chi tiết hệ màu và đối chiếu với định nghĩa thiết kế ở bảng bên trái.")

uploaded_file = st.file_uploader("Tải hình ảnh của bạn lên...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col_img, col_analysis = st.columns([1, 2])
    
    with col_img:
        st.image(image, caption="Hình ảnh gốc", use_container_width=True)
        
    img_resized = image.resize((100, 100))
    img_array = np.array(img_resized)
    pixels = img_array[:, :, :3].reshape(-1, 3) if img_array.shape[-1] == 4 else img_array.reshape(-1, 3)
    
    n_colors = 20
    
    with col_analysis:
        with st.spinner("Đang phân tích cấu trúc và tỷ lệ màu..."):
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            colors = kmeans.cluster_centers_.astype(int)
            percentages = np.bincount(labels, minlength=n_colors) / len(pixels)
            
        data_list = [analyze_color(colors[i], percentages[i]) for i in range(n_colors)]
        
        # --- PHÂN CHIA VAI TRÒ CHÍNH - PHỤ - NHẤN ---
        neutrals = [c for c in data_list if c["is_neutral"]]
        chromatics = [c for c in data_list if not c["is_neutral"]]
        chromatics_sorted = sorted(chromatics, key=lambda x: x["pct"], reverse=True)
        
        secondary_group = []
        for c in neutrals:
            if len(secondary_group) < 8:
                secondary_group.append(c)
                
        remaining_colors = [c for c in data_list if c not in secondary_group]
        remaining_colors = sorted(remaining_colors, key=lambda x: x["pct"], reverse=True)
        
        while len(secondary_group) < 6 and remaining_colors:
            mid_idx = len(remaining_colors) // 2
            secondary_group.append(remaining_colors.pop(mid_idx))
            
        remaining_colors = sorted(remaining_colors, key=lambda x: x["pct"], reverse=True)
        split_point = max(1, len(remaining_colors) // 2)
        
        dominant_group = remaining_colors[:split_point]
        accent_group = remaining_colors[split_point:]
        
        # --- TÍNH TOÁN % TỔNG VÀ MÀU ĐẠI DIỆN ---
        pct_dom = sum(item["pct"] for item in dominant_group) * 100
        pct_sec = sum(item["pct"] for item in secondary_group) * 100
        pct_acc = sum(item["pct"] for item in accent_group) * 100
        
        avg_dom = get_weighted_average_color(dominant_group)
        avg_sec = get_weighted_average_color(secondary_group)
        avg_acc = get_weighted_average_color(accent_group)
        
        # --- THIẾT LẬP CẤU TRÚC PHÂN NHÓM VÀ SẮP XẾP ---
        group_data = [
            {
                "name": "🎨 Màu Chính (Tông màu chủ đạo)",
                "key": "dom",
                "items": sorted(dominant_group, key=lambda x: x["pct"], reverse=True),
                "total_pct": pct_dom,
                "avg_rgb": avg_dom,
                "avg_hex": rgb_to_hex(avg_dom),
                "label_short": "Chính"
            },
            {
                "name": "📐 Màu Phụ (Màu trung tính nền hỗ trợ)",
                "key": "sec",
                "items": sorted(secondary_group, key=lambda x: x["pct"], reverse=True),
                "total_pct": pct_sec,
                "avg_rgb": avg_sec,
                "avg_hex": rgb_to_hex(avg_sec),
                "label_short": "Phụ"
            },
            {
                "name": "⚡ Màu Nhấn (Chi tiết tạo điểm nhấn)",
                "key": "acc",
                "items": sorted(accent_group, key=lambda x: x["pct"], reverse=True),
                "total_pct": pct_acc,
                "avg_rgb": avg_acc,
                "avg_hex": rgb_to_hex(avg_acc),
                "label_short": "Nhấn"
            }
        ]
        
        # SẮP XẾP THỨ TỰ NHÓM: Nhóm có % lớn hơn sẽ đứng trước (bên trái)
        group_data_sorted = sorted(group_data, key=lambda x: x["total_pct"], reverse=True)

        # --- 1. THANH TỶ LỆ MÀU KHÁI QUÁT (GÓC VUÔNG) ---
        st.subheader("1. Thanh tỷ lệ màu khái quát:")
        st.write("Thể hiện trực quan tỷ lệ 3 nhóm màu đại diện (Sắp xếp từ lớn đến nhỏ từ trái qua phải).")
        
        macro_bar_html = '<div style="display: flex; width: 100%; height: 50px; border-radius: 0px; overflow: hidden; border: 1.5px solid #888; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">'
        for g in group_data_sorted:
            if g["total_pct"] > 0:
                macro_bar_html += f'<div style="background-color: {g["avg_hex"]}; width: {g["total_pct"]}%; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 13px; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);" title="Nhom {g["label_short"]}: {g["avg_hex"]} ({g["total_pct"]:.1f}%)">{g["label_short"]} ({g["total_pct"]:.1f}%)</div>'
        macro_bar_html += '</div>'
        st.markdown(macro_bar_html, unsafe_allow_html=True)
        
        # Nút tải ảnh thanh màu khái quát
        macro_png = generate_macro_bar_png_dynamic(group_data_sorted)
        st.download_button("📥 Tải ảnh thanh màu khái quát (PNG)", data=macro_png, file_name="thanh_ti_le_khai_quat.png", mime="image/png")
        
        # Thông tin tỷ lệ thực tế xếp từ lớn đến nhỏ
        info_text = "📊 **Tỷ lệ thực tế (Lớn -> Nhỏ):** " + " — ".join([f"{g['label_short']} **{g['total_pct']:.1f}%**" for g in group_data_sorted])
        st.info(info_text)

        # --- 2. THANH TỶ LỆ MÀU CHI TIẾT ---
        st.subheader("2. Thanh tỷ lệ màu chi tiết:")
        
        master_bar_html = '<div style="display: flex; width: 100%; height: 50px; border-radius: 0px; overflow: hidden; border: 1.5px solid #aaa; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">'
        
        master_list = []
        for g in group_data_sorted:
            master_list.extend(g["items"])
        
        for item in master_list:
            r, g, b = item["rgb"]
            pct_val = item["pct"] * 100
            hex_val = item["hex"]
            
            role_label = "Mau Nhan"
            for g_grp in group_data_sorted:
                if item in g_grp["items"]:
                    role_label = g_grp["label_short"]
                    break
            
            master_bar_html += f'<div style="background-color: rgb({r},{g},{b}); width: {pct_val}%; height: 100%;" title="Nhom {role_label}: {hex_val} ({pct_val:.1f}%)"></div>'
            
        master_bar_html += '</div>'
        st.markdown(master_bar_html, unsafe_allow_html=True)
        
        # Nút tải ảnh thanh chi tiết 20 màu
        master_bar_png = generate_bar_png(master_list)
        st.download_button("📥 Tải ảnh thanh màu chi tiết (PNG)", data=master_bar_png, file_name="thanh_ti_le_chi_tiet.png", mime="image/png")
        
        st.write("---")

        # --- 3. CHI TIẾT TỪNG PHÂN NHÓM MÀU ---
        st.write("### 3. Chi tiết từng phân nhóm màu:")
        
        # Hàm hiển thị các màu dưới dạng lưới cố định 10 cột, hiển thị phần trăm tinh gọn
        def render_group_details_grid(group_items):
            if not group_items:
                st.write("*Không phát hiện màu phù hợp cho nhóm này.*")
                return
            
            cols_per_row = 10
            for r_idx in range(0, len(group_items), cols_per_row):
                chunk = group_items[r_idx : r_idx + cols_per_row]
                cols = st.columns(cols_per_row)
                
                for c_idx, item in enumerate(chunk):
                    pct = item["pct"] * 100
                    hex_val = item["hex"]
                    
                    with cols[c_idx]:
                        # Ô màu thu hẹp độ cao còn 18px và bỏ bo tròn
                        st.markdown(f'<div style="background-color: {hex_val}; height: 18px; border-radius: 0px; border: 1px solid #ccc; margin-bottom: 2px;"></div>', unsafe_allow_html=True)
                        # Hiển thị Tỷ lệ % của cụm màu
                        st.markdown(f'<p style="font-size: 10px; text-align: center; margin-bottom: 1px; color: #555;"><b>{pct:.1f}%</b></p>', unsafe_allow_html=True)
                        st.code(hex_val)

        # Duyệt qua các nhóm màu đã sắp xếp theo tỷ lệ phần trăm từ lớn đến nhỏ
        for g in group_data_sorted:
            total_pct = g["total_pct"]
            st.markdown(f"#### {g['name']} — Chiếm **{total_pct:.1f}%** toàn hình")
            
            # Khởi tạo thanh tỷ lệ nhỏ của riêng nhóm đó
            bar_html = '<div style="display: flex; width: 100%; height: 30px; border-radius: 0px; overflow: hidden; border: 1px solid #ddd; margin-bottom: 12px;">'
            for item in g["items"]:
                width_relative = (item["pct"] / (total_pct / 100)) * 100
                bar_html += f'<div style="background-color: rgb({item["rgb"][0]},{item["rgb"][1]},{item["rgb"][2]}); width: {width_relative}%; height: 100%;"></div>'
            bar_html += '</div>'
            st.markdown(bar_html, unsafe_allow_html=True)
            
            # Nút tải ảnh thanh màu của riêng nhóm này
            group_png = generate_bar_png(g["items"])
            st.download_button(f"📥 Tải thanh màu {g['label_short'].lower()}.png", data=group_png, file_name="thanh_mau_" + g["key"] + ".png", mime="image/png", key=f"btn_dl_{g['key']}_bar")
            st.write("")
            
            # Hiển thị lưới các ô màu lẻ tinh gọn
            render_group_details_grid(g["items"])
            
            st.markdown("<hr style='margin: 20px 0; border: 0; border-top: 1px solid #eee;'/>", unsafe_allow_html=True)