"""
Sửa chữ nằm TRONG ẢNH (ảnh chụp màn hình, bản scan).

Ảnh không có "font" thật, nên cách làm là:
  1. Đo màu nền, màu chữ, vùng mực chữ trong khung người dùng chọn.
  2. Nếu biết chữ gốc (OCR hoặc người dùng gõ): thử vẽ chữ gốc bằng nhiều font có trên máy,
     chọn font + cỡ chữ cho hình giống bản gốc nhất (so từng điểm ảnh).
  3. Xóa chữ cũ bằng màu nền (hoặc vá nền nếu nền không phẳng), vẽ chữ mới đúng
     đường chân chữ, màu chữ, font vừa chọn.
Kết quả: GẦN GIỐNG, độ giống phụ thuộc máy có font gốc hay không.
"""
import io
import os
import base64
import numpy as np
from collections import defaultdict
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from engine import SYSFONTS, EditError

try:
    import cv2
except Exception:   # không bắt buộc
    cv2 = None

CANDIDATE_FAMILIES = [
    "timesnewroman", "cambria", "georgia", "bookantiqua", "palatinolinotype", "garamond",
    "centuryschoolbook", "constantia", "arialnarrow", "tinos", "notoserif",
    "segoeui", "arial", "calibri", "aptos", "tahoma", "verdana", "timesnewroman", "cambria",
    "consolas", "couriernew", "roboto", "inter", "helvetica", "helveticaneue", "sfpro", "sfprotext",
    "notosans", "dejavusans", "liberationsans", "liberationserif", "dejavuserif", "opensans",
    "carlito", "caladea", "arimo", "cousine", "liberationmono",
]


def _sgn(v):
    return 0 if abs(v) < 1e-6 else (1 if v > 0 else -1)


def page_images(doc, pno):
    """Ảnh trên trang, tọa độ HIỂN THỊ (đã tính góc xoay trang) và chiều đặt ảnh."""
    page = doc[pno]
    rm = page.rotation_matrix
    out = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            rects = page.get_image_rects(xref, transform=True)
        except Exception:
            continue
        for r, m in rects:
            M = m * rm                       # ô vuông đơn vị của ảnh -> trang hiển thị
            ax = (_sgn(M.a), _sgn(M.b))      # trục ngang của ảnh nằm theo hướng nào trên màn hình
            ay = (_sgn(M.c), _sgn(M.d))
            card = (ax[1] == 0 and ay[0] == 0 and ax[0] and ay[1]) or (ax[0] == 0 and ay[1] == 0 and ax[1] and ay[0])
            d = r * rm
            out.append({"xref": xref, "bbox": [d.x0, d.y0, d.x1, d.y1], "w": img[2], "h": img[3],
                        "editable": bool(card), "ax": ax, "ay": ay})
    return out


def to_display(a, ax, ay):
    """Xoay/lật mảng điểm ảnh cho đúng chiều đang hiển thị trên trang."""
    if ax[0] != 0:
        out = a
        if ax[0] < 0:
            out = out[:, ::-1]
        if ay[1] < 0:
            out = out[::-1, :]
    else:
        out = np.swapaxes(a, 0, 1)
        if ax[1] < 0:
            out = out[::-1, :]
        if ay[0] < 0:
            out = out[:, ::-1]
    return np.ascontiguousarray(out)


def from_display(d, ax, ay):
    if ax[0] != 0:
        out = d
        if ay[1] < 0:
            out = out[::-1, :]
        if ax[0] < 0:
            out = out[:, ::-1]
    else:
        out = d
        if ay[0] < 0:
            out = out[:, ::-1]
        if ax[1] < 0:
            out = out[::-1, :]
        out = np.swapaxes(out, 0, 1)
    return np.ascontiguousarray(out)


def rot_box(b, k, H, W):
    """Toạ độ khung sau np.rot90(mảng, k) (xoay ngược chiều kim đồng hồ k lần)."""
    x0, y0, x1, y1 = b
    for _ in range(k % 4):
        x0, y0, x1, y1 = y0, W - x1, y1, W - x0
        H, W = W, H
    return (x0, y0, x1, y1)


def text_rot_k(text_rot):
    """Góc xoay chữ (độ, theo chiều kim đồng hồ) -> số lần np.rot90."""
    return (-(int(text_rot or 0) // 90)) % 4


def _load_rgb(doc, xref):
    pix = pymupdf.Pixmap(doc, xref)
    alpha = None
    if pix.alpha:
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
        alpha = a[:, :, -1].copy()
        pix = pymupdf.Pixmap(pix, 0)
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].copy()
    smask = doc.xref_get_key(xref, "SMask")
    if alpha is None and smask[0] == "xref":
        m = pymupdf.Pixmap(doc, int(smask[1].split()[0]))
        if m.w == pix.w and m.h == pix.h:
            alpha = np.frombuffer(m.samples, np.uint8).reshape(m.h, m.w, m.n)[:, :, 0].copy()
    return arr, alpha


def _to_px(box_pt, img_rect, w, h):
    x0, y0, x1, y1 = box_pt
    sx = w / (img_rect[2] - img_rect[0]); sy = h / (img_rect[3] - img_rect[1])
    px0 = int(np.floor((x0 - img_rect[0]) * sx)); px1 = int(np.ceil((x1 - img_rect[0]) * sx))
    py0 = int(np.floor((y0 - img_rect[1]) * sy)); py1 = int(np.ceil((y1 - img_rect[1]) * sy))
    return max(0, px0), max(0, py0), min(w, px1), min(h, py1)


def _find_image(doc, pno, box_pt):
    imgs = [i for i in page_images(doc, pno) if i["editable"]]
    bx = pymupdf.Rect(box_pt)
    best = None
    for i in imgs:
        inter = bx & pymupdf.Rect(i["bbox"])
        if not inter.is_empty and (best is None or inter.get_area() > best[0]):
            best = (inter.get_area(), i)
    if best is None:
        raise EditError("Khung chọn không nằm trên ảnh nào.")
    return best[1]


def crop_png(doc, pno, box_pt, upscale=3):
    info = _find_image(doc, pno, box_pt)
    arr, _ = _load_rgb(doc, info["xref"])
    arr = to_display(arr, info["ax"], info["ay"])
    x0, y0, x1, y1 = _to_px(box_pt, info["bbox"], arr.shape[1], arr.shape[0])
    im = Image.fromarray(arr[y0:y1, x0:x1])
    im = im.resize((max(1, im.width * upscale), max(1, im.height * upscale)), Image.LANCZOS)
    bio = io.BytesIO(); im.save(bio, "PNG")
    return bio.getvalue()


def ocr_png(doc, pno, box_pt, scale=5):
    """Ảnh sạch cho bộ nhận dạng chữ: chữ đen trên nền trắng (bỏ hoa văn nền, bỏ đường kẻ bảng),
    phóng to và chừa lề trắng — giúp không sót dấu chấm, dấu thanh tiếng Việt."""
    from PIL import ImageOps
    info = _find_image(doc, pno, box_pt)
    raw, _ = _load_rgb(doc, info["xref"])
    disp = to_display(raw, info["ax"], info["ay"])
    box = _to_px(box_pt, info["bbox"], disp.shape[1], disp.shape[0])
    try:
        an = _analyze(disp, box)
        im = Image.fromarray((255 * (1 - an["inten"])).clip(0, 255).astype(np.uint8))
    except EditError:
        return crop_png(doc, pno, box_pt)
    sc = max(1, min(scale, int(2400 / max(1, max(im.width, im.height)))))
    im = im.resize((im.width * sc, im.height * sc), Image.BICUBIC)
    im = ImageOps.expand(im, border=20, fill=255)
    bio = io.BytesIO(); im.save(bio, "PNG")
    return bio.getvalue()


def _write_image(doc, xref, rgb):
    """Ghi đè điểm ảnh của ảnh trong PDF, giữ nguyên kích thước, vị trí, góc đặt và lớp trong suốt.
    (Không dùng page.replace_image vì hàm đó xoay/kéo giãn ảnh khi trang có góc xoay.)"""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    filt = doc.xref_get_key(xref, "Filter")[1] or ""
    if "DCTDecode" in filt:                     # ảnh gốc là JPEG: giữ JPEG, chất lượng cao
        bio = io.BytesIO()
        Image.fromarray(rgb).save(bio, "JPEG", quality=95, subsampling=0)
        doc.update_stream(xref, bio.getvalue(), compress=False)
        doc.xref_set_key(xref, "Filter", "/DCTDecode")
    else:                                       # còn lại: nén không mất dữ liệu
        doc.update_stream(xref, rgb.tobytes(), compress=True)
        doc.xref_set_key(xref, "Filter", "/FlateDecode")
    for k, v in (("Width", str(w)), ("Height", str(h)), ("ColorSpace", "/DeviceRGB"),
                 ("BitsPerComponent", "8")):
        doc.xref_set_key(xref, k, v)
    for k in ("DecodeParms", "Decode", "ColorTransform", "ImageMask"):
        if doc.xref_get_key(xref, k)[0] != "null":
            doc.xref_set_key(xref, k, "null")


def _bg_color(pix):
    """Màu nền = màu xuất hiện nhiều nhất trong vùng (chữ và đường kẻ chỉ chiếm phần nhỏ).
    Không lấy trung vị viền ngoài vì viền có thể nằm đúng trên đường kẻ bảng."""
    q = (pix // 32).astype(np.int64)
    key = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    top = np.bincount(key).argmax()
    center = np.median(pix[key == top], axis=0)
    near = pix[np.sqrt(((pix - center) ** 2).sum(-1)) < 30]
    return np.median(near, axis=0) if len(near) else center


def _analyze(arr, box):
    x0, y0, x1, y1 = box
    H, W = arr.shape[:2]
    reg = arr[y0:y1, x0:x1].astype(np.float32)
    # nền: viền 2px ngay bên ngoài khung (hoặc viền trong nếu khung sát mép ảnh)
    ring = []
    for (a0, a1, b0, b1) in ((y0 - 3, y0 - 1, x0, x1), (y1 + 1, y1 + 3, x0, x1),
                             (y0, y1, x0 - 3, x0 - 1), (y0, y1, x1 + 1, x1 + 3)):
        a0, a1, b0, b1 = max(0, a0), min(H, a1), max(0, b0), min(W, b1)
        if a1 > a0 and b1 > b0:
            ring.append(arr[a0:a1, b0:b1].reshape(-1, 3))
    if not ring or sum(len(r) for r in ring) < 10:
        ring = [np.concatenate([reg[:2].reshape(-1, 3), reg[-2:].reshape(-1, 3)])]
    ring = np.concatenate(ring).astype(np.float32)
    bg = _bg_color(np.concatenate([reg.reshape(-1, 3), ring]))
    allpix = np.concatenate([reg.reshape(-1, 3), ring])
    dist = np.sqrt(((allpix - bg) ** 2).sum(-1))
    near = dist[dist < 60]                                  # bỏ chữ và đường kẻ khi đo độ phẳng của nền
    uniform = float(np.percentile(near, 80)) < 14 if len(near) else True
    d = np.sqrt(((reg - bg) ** 2).sum(-1))
    dmax = float(d.max()) if d.size else 0
    if dmax < 30:
        raise EditError("Không thấy chữ trong khung đã chọn.")
    rule = _detect_rules(d, dmax)
    # màu chữ = màu của 3% điểm đậm nhất (lõi nét). Ảnh scan nhòe làm phần lớn nét nhạt hơn màu mực thật.
    cand = (d >= 0.5 * dmax) & ~rule
    if not cand.any():
        cand = d >= 0.5 * dmax
    dc = d[cand]
    strong = cand & (d >= np.percentile(dc, 97))
    fg = reg[strong].mean(axis=0)
    # mức "giống màu chữ" của từng điểm: chiếu lên trục màu nền -> màu chữ
    axis = fg - bg
    inten = np.clip(((reg - bg) @ axis) / max(1.0, float(axis @ axis)), 0, 1)
    inten[rule] = 0
    text_ink = inten > 0.4
    if not text_ink.any():
        raise EditError("Không thấy chữ trong khung đã chọn (chỉ thấy đường kẻ).")
    ys, xs = np.nonzero(text_ink)
    ib = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return {"bg": bg, "fg": fg, "uniform": uniform, "ink_box": ib, "inten": inten, "d": d,
            "ink": text_ink, "rule": rule, "box": box}


def _detect_rules(d, dmax):
    """Đường kẻ bảng: nét ngang dài (>= 30% bề rộng khung) hoặc nét dọc gần hết chiều cao khung."""
    H, W = d.shape
    line_ink = d > max(25.0, 0.25 * dmax)        # ngưỡng thấp hơn: nét kẻ scan thường mờ không đều
    rule = np.zeros((H, W), bool)
    if cv2 is not None:
        m = line_ink.astype(np.uint8)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(0.3 * W)), 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(0.85 * H))))
        cand = (cv2.morphologyEx(m, cv2.MORPH_OPEN, kh) | cv2.morphologyEx(m, cv2.MORPH_OPEN, kv)) > 0
        cand |= m > 0      # xét cả thành phần liên thông gốc (nét kẻ hơi nghiêng)
        n, lab, st, _ = cv2.connectedComponentsWithStats((cand & line_ink).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, _a = st[i]
            touch_lr = x <= 1 or x + w >= W - 1
            touch_tb = y <= 1 or y + h >= H - 1
            # nét ngang: rất dài, hoặc dài vừa và chạm mép khung; luôn mảnh
            horiz = h <= max(6, 0.15 * H) and (w >= 0.6 * W or (w >= 0.3 * W and touch_lr))
            vert = w <= max(6, 0.15 * W) and (h >= 0.85 * H or (h >= 0.5 * H and touch_tb))
            if horiz or vert:
                rule |= lab == i
        # nét ngang dính vào chữ: chỉ lấy phần mở hình thái dài, chạm mép khung
        hm = cv2.morphologyEx(m, cv2.MORPH_OPEN, kh) > 0
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(hm.astype(np.uint8), connectivity=8)
        hrule = np.zeros((H, W), bool)
        for i in range(1, n2):
            x, y, w, h, _a = st2[i]
            if w >= 0.6 * W or ((x <= 1 or x + w >= W - 1) and w >= 0.3 * W):
                hrule |= lab2 == i
        rule |= hrule
        # nét dọc dính vào khung ô (khung chọn ôm cả ô bảng): lấy phần mở hình thái dọc, đủ dài
        vm = cv2.morphologyEx(m, cv2.MORPH_OPEN, kv) > 0
        n3, lab3, st3, _ = cv2.connectedComponentsWithStats(vm.astype(np.uint8), connectivity=8)
        for i in range(1, n3):
            x, y, w, h, _a = st3[i]
            if w > max(6, 0.15 * W) or h < 0.5 * H:
                continue
            # chỉ nhận khi nối vào đường kẻ ngang (khung ô) hoặc chạy suốt chiều cao khung — tránh nhầm nét chữ
            ends = hrule[max(0, y - 3):y + 3, max(0, x - 2):x + w + 2].any() or \
                hrule[max(0, y + h - 3):y + h + 3, max(0, x - 2):x + w + 2].any()
            if ends or h >= 0.97 * H:
                rule |= lab3 == i
        rule = cv2.dilate(rule.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    # đường kẻ chấm/gạch/răng cưa: dải rất mảnh trải gần hết bề rộng (hoặc chiều cao) khung
    thin_h, thin_w = max(4, int(0.06 * H)), max(4, int(0.06 * W))
    for axis, thin in ((1, thin_h), (0, thin_w)):
        prof = line_ink.any(axis=axis)                  # hàng (axis=1) hoặc cột (axis=0) có mực
        idx = np.nonzero(prof)[0]
        if not len(idx):
            continue
        groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for g in groups:
            if len(g) <= thin:
                sl = line_ink[g[0]:g[-1] + 1, :] if axis == 1 else line_ink[:, g[0]:g[-1] + 1]
                cover = sl.any(axis=0 if axis == 1 else 1).mean()
                if cover >= 0.6:
                    if axis == 1:
                        rule[max(0, g[0] - 1):g[-1] + 2, :] = True
                    else:
                        rule[:, max(0, g[0] - 1):g[-1] + 2] = True
    if cv2 is None:
        rows = line_ink.mean(axis=1) > 0.6
        cols = line_ink.mean(axis=0) > 0.85
        rule = rule.copy()
        for i in np.nonzero(rows)[0]:
            rule[max(0, i - 1):i + 2, :] = True
        for j in np.nonzero(cols)[0]:
            rule[:, max(0, j - 1):j + 2] = True
    return rule


def _bands(an, want=None):
    """Tách khung thành các dòng chữ theo khoảng trống ngang. Dấu tiếng Việt được gộp vào dòng."""
    ink = an["ink"]
    rows = np.nonzero(ink.any(axis=1))[0]
    if not len(rows):
        return []
    bands, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r > prev + 1:
            bands.append([start, prev + 1]); start = r
        prev = r
    bands.append([start, prev + 1])
    # gộp dải quá thấp (dấu, chấm) vào dải gần nhất
    changed = True
    while changed and len(bands) > 1:
        changed = False
        hmax = max(b[1] - b[0] for b in bands)
        for i, b in enumerate(bands):
            if b[1] - b[0] < 0.4 * hmax:
                gaps = []
                if i > 0:
                    gaps.append((b[0] - bands[i - 1][1], i - 1))
                if i < len(bands) - 1:
                    gaps.append((bands[i + 1][0] - b[1], i + 1))
                j = min(gaps)[1]
                lo, hi = min(i, j), max(i, j)
                bands[lo] = [min(bands[lo][0], bands[hi][0]), max(bands[lo][1], bands[hi][1])]
                del bands[hi]
                changed = True
                break
    # khớp số dòng với chữ gốc: gộp các cặp có khoảng trống nhỏ nhất
    while want and len(bands) > want:
        gaps = [(bands[i + 1][0] - bands[i][1], i) for i in range(len(bands) - 1)]
        _, i = min(gaps)
        bands[i] = [bands[i][0], bands[i + 1][1]]
        del bands[i + 1]
    out = []
    for y0, y1 in bands:
        cols = np.nonzero(ink[y0:y1].any(axis=0))[0]
        out.append((int(cols.min()), int(y0), int(cols.max()) + 1, int(y1)))
    return out


def _font(entry, size):
    return ImageFont.truetype(entry["path"], size=max(1.0, float(size)), index=entry["index"])


def _render(entry, size, text, frac=(0.0, 0.0), ss=4, margin=3, sx=1.0):
    """Vẽ chữ thành mảng 0..1, gốc đường chân chữ tại (ox, oy) với phần lẻ đúng bằng `frac`.
    ss=4: vẽ to gấp 4 rồi thu nhỏ (giống chữ in rồi scan); ss=1: vẽ trực tiếp, căn lưới điểm ảnh
    (giống chữ màn hình). sx: hệ số co/giãn ngang (ảnh scan hay bị lệch tỉ lệ ngang/dọc)."""
    fb = _font(entry, size * ss)
    l, t, r, b = fb.getbbox(text, anchor="ls")
    ox = float(np.ceil(-l * sx / ss + margin)) + frac[0]
    oy = float(np.ceil(-t / ss + margin)) + frac[1]
    W = int(np.ceil(ox + r * sx / ss + margin)) + 1
    H = int(np.ceil(oy + b / ss + margin)) + 1
    W, H = max(W, 1), max(H, 1)
    if ss == 1 and abs(sx - 1.0) < 1e-3:
        im = Image.new("L", (W, H), 0)
        ImageDraw.Draw(im).text((ox, oy), text, font=fb, fill=255, anchor="ls")
    else:
        Wb = int(np.ceil(W * ss / sx))
        big = Image.new("L", (Wb, H * ss), 0)
        ImageDraw.Draw(big).text((ox * ss / sx, oy * ss), text, font=fb, fill=255, anchor="ls")
        im = big.resize((W, H), Image.BOX, box=(0, 0, W * ss / sx, H * ss)) if ss > 1 else \
            big.resize((W, H), Image.BILINEAR, box=(0, 0, W / sx, H))
    a = np.asarray(im, np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.4)
    if not len(xs):
        return None
    return a, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), (ox, oy)


_CAND_LOGGED = False
_LAST_FITS = []


def _debug_dump(arr, box, an, bands, old_lines, new_lines, plan, out):
    """Lưu dữ liệu chẩn đoán vào thư mục debug/ cạnh app (chỉ trên máy người dùng)."""
    import json, time, shutil
    from PIL import ImageFilter
    from engine import APP_DIR
    root = os.path.join(APP_DIR, "debug")
    os.makedirs(root, exist_ok=True)
    olds = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for d in olds[:-9]:
        shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    folder = os.path.join(root, time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}")
    os.makedirs(folder, exist_ok=True)
    x0, y0, x1, y1 = box
    m = 10
    cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(arr.shape[1], x1 + m), min(arr.shape[0], y1 + m)
    Image.fromarray(arr[cy0:cy1, cx0:cx1]).save(os.path.join(folder, "goc.png"))
    Image.fromarray(out[cy0:cy1, cx0:cx1]).save(os.path.join(folder, "sau.png"))
    Image.fromarray((255 * (1 - an["inten"])).clip(0, 255).astype(np.uint8)).save(os.path.join(folder, "inten.png"))
    info = {"box": list(map(int, box)), "bands": [list(map(int, b)) for b in bands], "old": old_lines, "new": new_lines,
            "fg": [float(v) for v in an["fg"]], "bg": [float(v) for v in an["bg"]], "lines": []}
    for i, fits in enumerate(_LAST_FITS):
        ranked = sorted(fits.values(), key=lambda f: -f["score"])
        rows = [{"font": f"{f['e']['family']} {f['e']['sub']}", "file": os.path.basename(f["e"]["path"]),
                 "score": round(f["score"], 4), "size": round(f["size"], 2), "blur": f["blur"], "ss": f.get("ss", 4),
                 "sx": round(f.get("sx", 1.0), 3), "spread": f.get("spread", 1.0)}
                for f in ranked]
        info["lines"].append(rows)
        # ảnh so sánh: chữ gốc | chữ vẽ thử của từng font (đã đặt đúng chỗ, đã làm nhòe)
        if i >= len(bands):
            continue
        bx0, by0, bx1, by1 = bands[i]
        pad = 6
        gx0, gy0 = max(0, bx0 - pad), max(0, by0 - pad)
        gx1, gy1 = min(an["inten"].shape[1], bx1 + pad), min(an["inten"].shape[0], by1 + pad)
        tiles = [an["inten"][gy0:gy1, gx0:gx1]]
        names = ["GOC"]
        pick = ranked[:6] + [f for f in ranked if f["e"]["fam_n"] in ("timesnewroman", "cambria") and f not in ranked[:6]]
        for f in pick:
            a = f["a"]
            if f["blur"]:
                a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(f["blur"])),
                               np.float32) / 255.0
            a = np.clip(a * f.get("spread", 1.0), 0, 1)
            tiles.append(_paste(np.zeros_like(an["inten"]), a, f["tx"], f["ty"])[gy0:gy1, gx0:gx1])
            names.append(f"{f['e']['family']} {f['e']['sub']} {round(f['score'] * 100)}%")
        th, tw = tiles[0].shape
        sheet = Image.new("L", (tw * 3, (th * 3 + 16) * len(tiles)), 255)
        dr = ImageDraw.Draw(sheet)
        for k, (t, nm) in enumerate(zip(tiles, names)):
            im = Image.fromarray((255 * (1 - t)).clip(0, 255).astype(np.uint8)).resize((tw * 3, th * 3), Image.NEAREST)
            sheet.paste(im, (0, k * (th * 3 + 16) + 16))
            dr.text((2, k * (th * 3 + 16) + 2), nm, fill=0)
        sheet.save(os.path.join(folder, f"so_sanh_dong{i + 1}.png"))
    with open(os.path.join(folder, "info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)
    print(f"[Sửa PDF] đã lưu dữ liệu chẩn đoán: {folder}", flush=True)


def _candidates():
    global _CAND_LOGGED
    if SYSFONTS.entries is None:
        SYSFONTS._scan()
    seen, out = set(), []
    for fam in CANDIDATE_FAMILIES:
        for e in SYSFONTS.entries:
            if e["fam_n"] == fam and not e["italic"] and (e["path"], e["index"]) not in seen:
                if e["weight"] in (400, 700) or e["sub"].lower() in ("regular", "bold"):
                    seen.add((e["path"], e["index"])); out.append(e)
    out = out[:90]
    if not _CAND_LOGGED:
        _CAND_LOGGED = True
        fams = sorted({e["family"] for e in out})
        print(f"[Sửa PDF] {len(SYSFONTS.entries)} font trên máy; {len(out)} font dùng để so khớp chữ trong ảnh: "
              + ", ".join(fams), flush=True)
    return out


def _paste(canvas, a, tx, ty):
    H, W = canvas.shape
    h, w = a.shape
    x0, y0 = max(0, tx), max(0, ty)
    x1, y1 = min(W, tx + w), min(H, ty + h)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], a[y0 - ty:y1 - ty, x0 - tx:x1 - tx])
    return canvas


def _score(an, band, a, tx, ty, rbox):
    """Độ giống (0..1) giữa chữ vẽ thử đặt ở (tx, ty) và chữ gốc quanh dòng `band`:
    hệ số tương quan chuẩn hóa — so hình dạng, ít phụ thuộc nét đậm/nhạt."""
    I = an["inten"]
    bx0, by0, bx1, by1 = band
    rx0, ry0, rx1, ry1 = rbox
    gx0, gy0 = max(0, min(bx0, tx + rx0) - 2), max(0, min(by0, ty + ry0) - 2)
    gx1, gy1 = min(I.shape[1], max(bx1, tx + rx1) + 2), min(I.shape[0], max(by1, ty + ry1) + 2)
    R = _paste(np.zeros_like(I), a, tx, ty)[gy0:gy1, gx0:gx1]
    O = I[gy0:gy1, gx0:gx1]
    o = O - O.mean(); r = R - R.mean()
    den = float(np.sqrt((o * o).sum() * (r * r).sum()))
    return max(0.0, float((o * r).sum()) / den) if den > 1e-9 else 0.0


BLURS = (0.0, 0.8, 1.4, 2.0)


def _place(an, band, e, text, size=None, frac=(0.0, 0.0), blurs=BLURS, ss=4, sx=None):
    """Tìm cỡ chữ, vị trí, độ nhòe để `text` vẽ bằng font e trùng với dòng chữ gốc nhất."""
    from PIL import ImageFilter
    bx0, by0, bx1, by1 = band
    ow, oh = bx1 - bx0, by1 - by0
    if sx is None:
        sx = 1.0
    if size is None:
        s0 = max(8.0, oh * 1.2)
        r0 = _render(e, s0, text, ss=ss)
        if not r0:
            return None
        w0 = r0[1][2] - r0[1][0]
        size = max(4.0, s0 * ow / max(1, w0) / sx)       # cỡ chữ suy từ bề rộng dòng (ổn định hơn chiều cao)
    r = _render(e, size, text, frac, ss=ss, sx=sx)
    if not r:
        return None
    a, rbox, (ox, oy) = r
    rx0, ry0, rx1, ry1 = rbox
    if not (0.35 * oh < ry1 - ry0 < 2.2 * oh):
        return None
    tx = bx0 - rx0
    # vị trí dọc: dò theo hình chiếu ngang (bền khi chữ gốc nhận dạng thiếu dấu)
    I = an["inten"]
    PI = I[:, bx0:bx1].sum(axis=1)
    PA = a[:, rx0:rx1].sum(axis=1)
    Ha, Hb = a.shape[0], I.shape[0]
    ty0 = by0 - ry0
    best_ty, best_c = ty0, -1.0
    for ty in range(int(ty0 - 0.6 * oh), int(ty0 + 0.6 * oh) + 1):
        lo, hi = max(0, -ty), min(Ha, Hb - ty)
        if hi <= lo:
            continue
        c = float((PA[lo:hi] * PI[lo + ty:hi + ty]).sum())
        if c > best_c:
            best_c, best_ty = c, ty
    base = Image.fromarray((a * 255).astype(np.uint8))
    best = (-1.0, 0.0, 1.0)
    for sg in blurs:
        ab0 = np.asarray(base.filter(ImageFilter.GaussianBlur(sg)), np.float32) / 255.0 if sg else a
        for kk in (1.0, 1.6, 2.4):                      # mực in loang: nét dày hơn font gốc
            ab = np.clip(ab0 * kk, 0, 1) if kk != 1.0 else ab0
            sc = _score(an, band, ab, tx, best_ty, rbox)
            if sc > best[0] + (0.004 if (sg or kk != 1.0) else 0.0):   # chỉ thêm nhòe/loang khi giống hơn rõ
                best = (sc, sg, kk)
    # co/giãn ngang chỉ là phương án phụ: trừ điểm theo mức lệch khỏi tỉ lệ chuẩn
    best = (best[0] - 0.15 * abs(float(np.log(sx))), best[1], best[2])
    return {"score": best[0], "blur": best[1], "spread": best[2], "size": size, "base_x": tx + ox, "base_y": best_ty + oy,
            "a": a, "tx": tx, "ty": best_ty, "rbox": rbox, "e": e, "ss": ss, "sx": sx}


def _refine(an, band, fit, text):
    """Tinh chỉnh cỡ chữ (±3%) và vị trí lệch dưới 1 điểm ảnh — quan trọng với ảnh độ phân giải thấp."""
    best = fit
    near = tuple(sorted({max(0.0, fit["blur"] - 0.6), fit["blur"], fit["blur"] + 0.6}))
    # bước 1: tỉ lệ ngang (chữ gốc có thể hẹp/rộng hơn font chuẩn), giữ nguyên bề rộng dòng
    for sxv in (0.88, 0.94, 1.06, 1.12):
        f = _place(an, band, fit["e"], text, size=fit["size"] / sxv, blurs=near, ss=4, sx=sxv)
        if f and f["score"] > best["score"]:
            best = f
    base_sx, base_size = best["sx"], best["size"]
    # bước 2: tinh chỉnh cỡ chữ, vị trí lệch dưới 1 điểm ảnh, kiểu vẽ (scan / màn hình)
    for ss in (4, 1):
        for sf in (0.97, 1.0, 1.03):
            for fx in (0.0, 0.33, 0.67):
                for fy in (0.0, 0.33, 0.67):
                    f = _place(an, band, fit["e"], text, size=base_size * sf, frac=(fx, fy), blurs=near,
                               ss=ss, sx=base_sx)
                    if f and f["score"] > best["score"]:
                        best = f
    return best


def _choose_fonts(an, bands, old_lines, font_choice):
    """Chọn font cho từng dòng; ưu tiên một font chung cho cả khung nếu gần bằng font tốt nhất."""
    cands = [c for c in _candidates() if not font_choice or c["family"] == font_choice] or _candidates()
    per_line = []
    for i, band in enumerate(bands):
        fits = {}
        if i < len(old_lines):
            for e in cands:
                if any(not SYSFONTS.has_glyph(e, c) for c in old_lines[i]):
                    continue
                try:
                    f = _place(an, band, e, old_lines[i])
                except Exception:
                    f = None
                if f:
                    fits[(e["path"], e["index"])] = f
            for k in sorted(fits, key=lambda k: -fits[k]["score"])[:6]:
                try:
                    fits[k] = _refine(an, band, fits[k], old_lines[i])
                except Exception:
                    pass
        per_line.append(fits)
    totals = defaultdict(float)
    for fits in per_line:
        for k, f in fits.items():
            totals[k] += f["score"]
    ranked = sorted(totals, key=lambda k: -totals[k])
    chosen = []
    for fits in per_line:
        if not fits:
            chosen.append(None); continue
        own = max(fits.values(), key=lambda f: f["score"])
        common = fits.get(ranked[0]) if ranked else None
        chosen.append(common if common and common["score"] >= own["score"] - 0.03 else own)
    try:
        print("[Sửa PDF] xếp hạng font: " + ", ".join(
            f"{next(f['e'] for fits in per_line for kk, f in fits.items() if kk == k)['family']} "
            f"{next(f['e'] for fits in per_line for kk, f in fits.items() if kk == k)['sub']} "
            f"{round(totals[k] / max(1, sum(1 for fits in per_line if k in fits)) * 100)}%" for k in ranked[:8]), flush=True)
    except Exception:
        pass
    alts = []
    for k in ranked[1:6]:
        e = next(f["e"] for fits in per_line for kk, f in fits.items() if kk == k)
        n = sum(1 for fits in per_line if k in fits)
        alts.append(f"{e['family']} {e['sub']} ({round(totals[k] / max(1, n) * 100)}%)")
    global _LAST_FITS
    _LAST_FITS = per_line
    return chosen, alts


class _Metrics:
    """Số đo chữ khớp với cách vẽ (siêu lấy mẫu ss lần)."""
    def __init__(self, entry, size, ss, sx=1.0):
        self.f, self.ss, self.sx = _font(entry, size * ss), ss, sx

    def getlength(self, text):
        return self.f.getlength(text) / self.ss * self.sx

    def getbbox(self, text, anchor="ls"):
        l, t, r, b = self.f.getbbox(text, anchor=anchor)
        return (l / self.ss * self.sx, t / self.ss, r / self.ss * self.sx, b / self.ss)


def _text_canvas(shape, x, y, text, entry, size, ss=4, blur=0.0, sx=1.0):
    """Mặt nạ chữ (0..1) cỡ `shape`, gốc đường chân chữ tại (x, y) — vẽ giống hệt lúc so khớp."""
    from PIL import ImageFilter
    canvas = np.zeros(shape, np.float32)
    fx, fy = x - np.floor(x), y - np.floor(y)
    r = _render(entry, size, text, frac=(fx, fy), ss=ss, margin=3 + int(np.ceil(3 * blur)), sx=sx)
    if not r:
        return canvas
    a, _, (ox, oy) = r
    if blur:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur)),
                       np.float32) / 255.0
    return _paste(canvas, a, int(round(x - ox)), int(round(y - oy)))


def _detect_align(bands):
    if len(bands) < 2:
        return "left"
    lefts = [b[0] for b in bands]; rights = [b[2] for b in bands]
    centers = [(b[0] + b[2]) / 2 for b in bands]
    spread = lambda v: max(v) - min(v)
    best = min((spread(lefts), "left"), (spread(centers), "center"), (spread(rights), "right"))
    return best[1]


def _cell_align(arr, box, band, bg):
    """Canh lề cho khung chỉ có 1 dòng: tìm 2 đường kẻ dọc của ô bảng hai bên chữ (tìm cả ngoài khung chọn).
    Chữ cách đều 2 đường kẻ -> canh giữa; sát bên nào -> canh bên đó. Không thấy ô bảng: xét lề trong khung chọn."""
    x0, y0, x1, y1 = box
    H, W = arr.shape[:2]
    tx0, tx1 = x0 + band[0], x0 + band[2]
    ty0, ty1 = y0 + band[1], y0 + band[3]
    h = max(4, ty1 - ty0)
    # cửa sổ dọc cao hơn dòng chữ: nét chữ không đủ dài để bị nhầm là đường kẻ
    wy0, wy1 = max(0, ty0 - h // 2), min(H, ty1 + h // 2)
    win = np.sqrt(((arr[wy0:wy1].astype(np.float32) - bg) ** 2).sum(-1))
    dmax = float(win.max()) if win.size else 0.0
    if dmax < 30:
        return None
    ink = win > max(25.0, 0.3 * dmax)
    colfrac = ink.mean(axis=0)
    reach = int(8 * h)

    def find(xs):
        for x in xs:
            if colfrac[x] >= 0.85:
                return x
        return None
    L = find(range(tx0 - 1, max(-1, tx0 - reach), -1))
    R = find(range(tx1, min(W, tx1 + reach)))
    if L is not None and R is not None:
        ml, mr = tx0 - (L + 1), R - tx1
        cw = R - L
    else:
        ml, mr = tx0 - x0, x1 - tx1
        cw = x1 - x0
        if min(ml, mr) < 0.6 * h:        # khung chọn ôm sát chữ: không đủ căn cứ, giữ canh trái
            return None
    if abs(ml - mr) <= max(2.0, 0.12 * cw):
        return "center"
    return "left" if ml < mr else "right"


def edit_image_text(doc, pno, box_pt, old_text, new_text, font_choice=None, commit=False, text_rot=0,
                    align="auto", fit=True):
    info = _find_image(doc, pno, box_pt)
    raw, alpha = _load_rgb(doc, info["xref"])
    disp = to_display(raw, info["ax"], info["ay"])          # đúng chiều đang thấy trên trang
    box = _to_px(box_pt, info["bbox"], disp.shape[1], disp.shape[0])
    k_rot = text_rot_k(text_rot)                             # chữ nằm dọc: xoay cho chữ đứng thẳng
    arr = np.ascontiguousarray(np.rot90(disp, k_rot))
    box = rot_box(box, k_rot, disp.shape[0], disp.shape[1])
    if box[2] - box[0] < 3 or box[3] - box[1] < 3:
        raise EditError("Khung chọn quá nhỏ.")
    x0, y0, x1, y1 = box
    an = _analyze(arr, box)

    old_lines = [l.strip() for l in (old_text or "").replace("\r", "").split("\n") if l.strip()]
    new_lines = (new_text or "").replace("\r", "").split("\n")
    while new_lines and not new_lines[-1].strip():
        new_lines.pop()
    bands = _bands(an, want=len(old_lines) or None)
    if old_lines and len(bands) < len(old_lines):
        # không tách được đủ dòng: coi cả khung là một dòng
        old_lines = [" ".join(old_lines)]
        bands = _bands(an, want=1)
    if not bands:
        raise EditError("Không thấy chữ trong khung đã chọn.")
    if align in (None, "", "auto"):
        al = (_cell_align(arr, box, bands[0], an["bg"]) or "left") if len(bands) == 1 else _detect_align(bands)
    else:
        al = align

    # 1. chọn font/cỡ chữ cho từng dòng
    plan, notes = [], []
    chosen, alts = _choose_fonts(an, bands, old_lines, font_choice)
    for i, bb in enumerate(bands):
        bx0, by0, bx1, by1 = bb
        fit = chosen[i]
        common = {"left": x0 + bx0, "right": x0 + bx1, "center": x0 + (bx0 + bx1) / 2,
                  "old": old_lines[i] if i < len(old_lines) else ""}
        if fit:
            plan.append({"e": fit["e"], "size": fit["size"], "base_x": x0 + fit["base_x"],
                         "base_y": y0 + fit["base_y"], "score": fit["score"], "blur": fit["blur"],
                         "ss": fit.get("ss", 4), "sx": fit.get("sx", 1.0), "spread": fit.get("spread", 1.0),
                         **common})
        else:
            cands = [c for c in _candidates() if not font_choice or c["family"] == font_choice] or _candidates()
            if not cands:
                raise EditError("Máy không có font nào để vẽ chữ.")
            plan.append({"e": cands[0], "size": (by1 - by0) / 0.72, "base_x": None, "base_y": y0 + by1,
                         "score": None, "blur": 0.0, **common})
    # dòng mới nhiều hơn dòng cũ: xếp tiếp bên dưới với cùng khoảng cách dòng
    pitch = (plan[-1]["base_y"] - plan[0]["base_y"]) / (len(plan) - 1) if len(plan) > 1 else plan[-1]["size"] * 1.25
    while len(plan) < len(new_lines):
        p = dict(plan[-1]); p["base_y"] = plan[-1]["base_y"] + pitch; p["old"] = ""; p["score"] = None
        plan.append(p)
        notes.append("Thêm dòng mới bên dưới, có thể tràn ra ngoài khung.")

    for i, p in enumerate(plan):
        txt = new_lines[i] if i < len(new_lines) else ""
        missing = [c for c in txt if not SYSFONTS.has_glyph(p["e"], c)]
        if missing:
            raise EditError(f"Font {p['e']['family']} không có ký tự: {''.join(sorted(set(missing)))}")

    out = arr.copy()
    Hh, Ww = arr.shape[:2]
    # dòng không đổi chữ: giữ nguyên điểm ảnh gốc
    keep = {i for i in range(len(plan)) if i < len(old_lines) and i < len(new_lines)
            and i < len(bands) and new_lines[i].strip() == old_lines[i]}
    axis = an["fg"] - an["bg"]
    proj = lambda a: np.clip(((a.astype(np.float32) - an["bg"]) @ axis) / max(1.0, float(axis @ axis)), 0, 1)
    rule_full = np.zeros((Hh, Ww), bool)
    rule_full[y0:y1, x0:x1] = an["rule"]
    hmax = max(b[3] - b[1] for b in bands)
    max_blur = max([p.get("blur") or 0.0 for p in plan] + [0.0])
    pad = int(max(2, round(2.5 * max_blur) + 1))          # chữ nhòe có quầng rộng hơn

    def erase(zone):
        """Xóa điểm ảnh chữ trong vùng zone, lấp bằng nền ước lượng từ các điểm nền xung quanh."""
        ys, xs = np.nonzero(zone)
        if not len(ys):
            return
        ey0, ey1, ex0, ex1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        sub = out[ey0:ey1, ex0:ex1]
        zsub = zone[ey0:ey1, ex0:ex1]
        mask = (proj(sub) > 0.12) & zsub & ~rule_full[ey0:ey1, ex0:ex1]
        if cv2 is None:
            sub[mask] = an["bg"].astype(np.uint8)
            return
        m8 = (cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & zsub
        m8 &= ~rule_full[ey0:ey1, ex0:ex1]
        ks = int(max(7, min(61, 0.6 * hmax))) | 1
        gy0, gx0 = max(0, ey0 - ks), max(0, ex0 - ks)
        gy1, gx1 = min(Hh, ey1 + ks), min(Ww, ex1 + ks)
        reg = out[gy0:gy1, gx0:gx1].astype(np.float32)
        textish = cv2.dilate((proj(reg) > 0.08).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        w = (~textish & ~rule_full[gy0:gy1, gx0:gx1]).astype(np.float32)
        est = an["bg"][None, None, :] * np.ones_like(reg)
        for win in (ks, ks * 2 + 1, ks * 4 + 1):        # cửa sổ lớn dần nếu quanh đó toàn chữ
            num = cv2.blur(reg * w[:, :, None], (win, win))
            den = cv2.blur(w, (win, win))[:, :, None]
            ok = den > 0.05
            est = np.where(ok, num / np.maximum(den, 1e-6), est)
            if ok.all():
                break
        bgpix = reg[w > 0]
        sd = float(np.median(np.abs(bgpix - np.median(bgpix, axis=0)))) * 1.4826 if len(bgpix) > 50 else 0.0
        if sd > 0.5:      # hạt nhiễu giống giấy scan
            est = est + np.random.default_rng(0).normal(0, min(sd, 6.0), est.shape)
        est = est.clip(0, 255).astype(np.uint8)[ey0 - gy0:ey1 - gy0, ex0 - gx0:ex1 - gx0]
        sub[m8] = est[m8]

    def line_tone(i, p):
        """Đường cong độ đậm: biến chữ vẽ thử thành chữ giống hệt chữ gốc (cùng độ dày, cùng độ đậm lõi nét).
        Cách làm: vẽ thử đúng chữ cũ ở đúng chỗ, rồi ghép phân bố độ phủ mực của bản vẽ thử với phân bố
        của chữ gốc trong cùng vùng (ghép theo thứ hạng/phân vị) — không bị nhạt như khi khớp bình phương
        tối thiểu, vì chữ vẽ ra có đúng tỉ lệ điểm đậm/nhạt như chữ gốc."""
        ident = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        spread = float(p.get("spread") or 1.0)
        if not (p["old"] and p["score"] is not None and p["base_x"] is not None and i < len(bands)):
            return spread, ident
        mo = _text_canvas((Hh, Ww), p["base_x"], p["base_y"], p["old"], p["e"], p["size"],
                          p.get("ss", 4), p.get("blur") or 0.0, p.get("sx", 1.0))[y0:y1, x0:x1]
        band = bands[i]
        hh, ww = an["inten"].shape
        sl = (slice(max(0, band[1] - 2), min(hh, band[3] + 2)), slice(max(0, band[0] - 2), min(ww, band[2] + 2)))
        ok = ~an["rule"][sl]
        ms = np.sort(mo[sl][ok])
        is_ = np.sort(an["inten"][sl][ok])
        if (ms > 0.05).sum() <= 20:
            return spread, ident
        # ghép theo thứ hạng; các điểm có cùng giá trị (vd. lõi nét bão hòa) lấy trung bình phần chữ gốc tương ứng
        vk, inv = np.unique(ms, return_inverse=True)
        fk = np.bincount(inv, weights=is_) / np.bincount(inv)
        fk = np.maximum.accumulate(fk)
        if vk[0] <= 1e-6:
            fk[0] = 0.0
        else:
            vk, fk = np.concatenate([[0.0], vk]), np.concatenate([[0.0], fk])
        return spread, (vk, fk)

    # màu mực theo độ phủ, học từ chính chữ gốc: lõi nét và mép nét có sắc màu riêng (mực xanh tím khi scan)
    reg_box = arr[y0:y1, x0:x1].astype(np.float32)
    knots = np.linspace(0, 1, 11)
    cols = np.full((len(knots), 3), np.nan, np.float32)
    cols[0] = an["bg"]
    okp = ~an["rule"]
    for k_i, c in enumerate(knots[1:], 1):
        sel = okp & ((an["inten"] >= 0.95) if c >= 1 else (np.abs(an["inten"] - c) < 0.05))
        if sel.sum() >= 5:
            cols[k_i] = reg_box[sel].mean(axis=0)
    if np.isnan(cols[-1, 0]):
        cols[-1] = an["fg"]
    good = ~np.isnan(cols[:, 0])
    for ch in range(3):
        cols[:, ch] = np.interp(knots, knots[good], cols[good, ch])

    def ink_color(a):
        """Màu điểm ảnh trên nền chuẩn khi độ phủ mực là a (mảng HxW) -> HxWx3."""
        return np.stack([np.interp(a, knots, cols[:, ch]) for ch in range(3)], axis=-1)

    canvas = np.zeros((Hh, Ww), np.float32)

    def draw(txt, x, p, size, tone):
        nonlocal canvas
        mk = _text_canvas((Hh, Ww), x, p["base_y"], txt, p["e"], size, p.get("ss", 4), p.get("blur") or 0.0,
                          p.get("sx", 1.0))
        spread, (vk, fk) = tone
        canvas = np.maximum(canvas, np.interp(mk, vk, fk).astype(np.float32))

    # 2–3. với từng dòng: sửa một phần (giữ nguyên ký tự không đổi) hoặc vẽ lại cả dòng
    avail = (x1 - x0) - 2
    drawn, zone, moves = [], np.zeros((Hh, Ww), bool), []
    for i, p in enumerate(plan):
        txt = new_lines[i] if i < len(new_lines) else ""
        if i in keep:
            continue
        band = bands[i] if i < len(bands) else None
        f = _Metrics(p["e"], p["size"], p.get("ss", 4), p.get("sx", 1.0))
        part = None
        if band and p["old"] and p["score"] is not None and p["score"] >= 0.88 and txt.strip():
            O, N = p["old"], txt
            pp = 0
            while pp < min(len(O), len(N)) and O[pp] == N[pp]:
                pp += 1
            ss = 0
            while ss < min(len(O), len(N)) - pp and O[-1 - ss] == N[-1 - ss]:
                ss += 1
            if pp + ss > 0:
                bx = p["base_x"]
                X = lambda k: bx + f.getlength(O[:k])

                def ink(k0, k1):
                    if k1 <= k0:
                        return None
                    l_, _t, r_, _b = f.getbbox(O[k0:k1], anchor="ls")
                    return (X(k0) + l_, X(k0) + r_) if r_ > l_ else None
                pre, mid, suf = ink(0, pp), ink(pp, len(O) - ss), ink(len(O) - ss, len(O))
                xa, xb = X(pp), X(len(O) - ss)
                if pre:
                    cut_a = (pre[1] + (mid[0] if mid else (suf[0] if suf else xa))) / 2
                else:
                    cut_a = (mid[0] if mid else xa) - pad
                if suf:
                    cut_b = ((mid[1] if mid else (pre[1] if pre else xb)) + suf[0]) / 2
                else:
                    cut_b = (mid[1] if mid else xb) + pad
                M = N[pp:len(N) - ss]
                delta = f.getlength(M) - (xb - xa)
                sh_pre = {"center": -delta / 2, "right": -delta}.get(al, 0.0)
                sh_suf = sh_pre + delta
                left_new = (pre[0] if pre else xa) + sh_pre
                right_new = (suf[1] if suf else xa + f.getlength(M)) + sh_suf
                if not fit or (left_new >= x0 - 1 and right_new <= x1 + 1):
                    part = (M, xa + sh_pre, cut_a, cut_b, int(round(sh_pre)), int(round(sh_suf)))
        tone = line_tone(i, p)
        if part is None:
            if not txt.strip():
                if band:
                    zone[max(0, y0 + band[1] - pad):y0 + band[3] + pad,
                         max(0, x0 + band[0] - pad):x0 + band[2] + pad] = True
                continue
            if band:
                zone[max(0, y0 + band[1] - pad):y0 + band[3] + pad,
                     max(0, x0 + band[0] - pad):x0 + band[2] + pad] = True
            size = p["size"]
            l, t, r, b = f.getbbox(txt, anchor="ls")
            if fit and (r - l) > avail and avail > 10:
                scale = avail / (r - l)
                size *= scale
                f = _Metrics(p["e"], size, p.get("ss", 4), p.get("sx", 1.0))
                l, t, r, b = f.getbbox(txt, anchor="ls")
                notes.append(f"Dòng {i + 1} dài hơn khung: đã thu nhỏ chữ còn {round(scale * 100)}%.")
            if al == "center":
                ox = p["center"] - (l + r) / 2
            elif al == "right":
                ox = p["right"] - r
            else:
                ox = p["base_x"] if p.get("base_x") is not None else p["left"] - l
            ox = min(max(ox, x0 - l), max(x0 - l, x1 - r)) if fit else ox
            draw(txt, ox, p, size, tone)
            drawn.append({"font": f"{p['e']['family']} {p['e']['sub']}", "size": round(size, 1),
                          "score": None if p["score"] is None else round(p["score"] * 100), "partial": False,
                          "sx": round(p.get("sx", 1.0), 2)})
        else:
            M, xm, cut_a, cut_b, s_pre, s_suf = part
            ry0, ry1 = max(0, y0 + band[1] - pad), min(Hh, y0 + band[3] + pad)
            lx, rx = max(0, x0 + band[0] - pad), min(Ww, x0 + band[2] + pad)
            ca, cb = max(lx, int(np.floor(cut_a))), min(rx, int(np.ceil(cut_b)))
            zone[ry0:ry1, ca:cb] = True
            if s_pre:                    # khối chữ phía trước dịch theo (canh giữa/phải)
                zone[ry0:ry1, lx:ca] = True
                moves.append((ry0, ry1, lx, ca, s_pre))
            if s_suf:                    # khối chữ phía sau dịch theo độ dài chữ mới
                zone[ry0:ry1, cb:rx] = True
                moves.append((ry0, ry1, cb, rx, s_suf))
            if M.strip():
                draw(M, xm, p, p["size"], tone)
            drawn.append({"font": f"{p['e']['family']} {p['e']['sub']}", "size": round(p["size"], 1),
                          "score": round(p["score"] * 100), "partial": True, "sx": round(p.get("sx", 1.0), 2)})
            notes.append(f"Dòng {i + 1}: chỉ vẽ lại phần chữ thay đổi, phần giống giữ nguyên ảnh gốc.")

    erase(zone)
    for (ry0, ry1, c0, c1, sh) in moves:     # dán lại phần chữ gốc đã dịch chỗ
        blk = arr[ry0:ry1, c0:c1].astype(np.float32)
        al_ = proj(blk)
        al_[rule_full[ry0:ry1, c0:c1]] = 0
        d0, d1 = c0 + sh, c1 + sh
        s0, s1 = max(0, -d0), (c1 - c0) - max(0, d1 - Ww)
        d0, d1 = max(0, d0), min(Ww, d1)
        if d1 <= d0:
            continue
        a2 = al_[:, s0:s1]                 # độ phủ mực của chữ gốc -> vẽ lại đúng độ phủ đó ở chỗ mới
        dst = out[ry0:ry1, d0:d1].astype(np.float32)
        out[ry0:ry1, d0:d1] = (ink_color(a2) + (dst - an["bg"]) * (1 - a2[:, :, None])).round().clip(0, 255).astype(np.uint8)

    # điểm ảnh = màu mực theo độ phủ (học từ chữ gốc) + phần lệch của nền tại chỗ (giữ hạt nhiễu giấy)
    ys_, xs_ = np.nonzero(canvas > 0)
    if len(ys_):
        cy0, cy1, cx0, cx1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
        a2 = canvas[cy0:cy1, cx0:cx1]
        dst = out[cy0:cy1, cx0:cx1].astype(np.float32)
        out[cy0:cy1, cx0:cx1] = (ink_color(a2) + (dst - an["bg"]) * (1 - a2[:, :, None])).round().clip(0, 255).astype(np.uint8)

    # ảnh xem trước trước/sau (đúng vùng khung chọn)
    mg = 10
    cx0, cy0 = max(0, x0 - mg), max(0, y0 - mg)
    cx1, cy1 = min(arr.shape[1], x1 + mg), min(arr.shape[0], y1 + mg)
    prev = []
    for a in (arr, out):
        pim = Image.fromarray(a[cy0:cy1, cx0:cx1])
        sc = max(1, min(4, 600 // max(1, pim.width)))
        pim = pim.resize((pim.width * sc, pim.height * sc), Image.NEAREST)
        bio = io.BytesIO(); pim.save(bio, "PNG")
        prev.append("data:image/png;base64," + base64.b64encode(bio.getvalue()).decode())

    scores = [p["score"] for p in plan[:len(bands)] if p["score"] is not None]
    fonts = []
    for dd in drawn:
        if dd["font"] not in fonts:
            fonts.append(dd["font"])
    match_info = {"font": ", ".join(fonts) or f"{plan[0]['e']['family']} {plan[0]['e']['sub']}",
                  "score": round(min(scores) * 100) if scores else None,
                  "alternatives": alts, "lines": drawn}
    result = {"ok": True, "match": match_info, "size_px": drawn[0]["size"] if drawn else round(plan[0]["size"], 1),
              "n_lines": len(bands), "align": al, "notes": notes,
              "text_color": "#%02x%02x%02x" % tuple(int(v) for v in an["fg"]),
              "bg_color": "#%02x%02x%02x" % tuple(int(v) for v in an["bg"]),
              "bg_uniform": an["uniform"], "before": prev[0], "after": prev[1],
              "shared": sum(1 for i in page_images(doc, pno) if i["xref"] == info["xref"]) > 1}
    if not commit and os.environ.get("SUAPDF_SERVER") != "1":   # máy chủ: không lưu ảnh của người dùng
        try:
            _debug_dump(arr, box, an, bands, old_lines, new_lines, plan, out)
        except Exception as ex:
            print("[Sửa PDF] không lưu được dữ liệu chẩn đoán:", ex, flush=True)
    if commit:
        out = from_display(np.rot90(out, -k_rot), info["ax"], info["ay"])   # trả về chiều gốc của ảnh
        if out.shape[:2] != raw.shape[:2]:
            raise EditError(f"Lỗi nội bộ: kích thước ảnh sau khi sửa {out.shape[:2]} không khớp ảnh gốc {raw.shape[:2]}.")
        _write_image(doc, info["xref"], out)
    return result
