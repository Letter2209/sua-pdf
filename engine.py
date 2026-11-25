"""
Lõi sửa chữ trong PDF, ưu tiên giữ nguyên font gốc.

Nguyên tắc:
  1. Chữ không đổi (phần đầu dòng) giữ nguyên, không đụng vào.
  2. Phần chữ thay đổi + phần sau nó bị xóa thật (redaction), rồi được ghi lại
     bằng CHÍNH font object có sẵn trong file (cùng mã ký tự, cùng độ rộng).
  3. Ký tự mà font gốc không có -> tìm font cùng tên trên máy -> nếu không có thì
     dùng font thay thế, nhưng PHẢI được người dùng xác nhận.
"""
import io
import os
import re
import sys
import glob
import json
import statistics
import unicodedata
from collections import defaultdict

import logging
import pymupdf
from fontTools.ttLib import TTFont
from fontTools import agl
from fontTools import subset as ft_subset

logging.getLogger("fontTools").setLevel(logging.ERROR)
SUBSET_RE = re.compile(r"^[A-Z]{6}\+")
APP_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Tiện ích đọc object PDF
# ----------------------------------------------------------------------------
def strip_subset(name):
    return SUBSET_RE.sub("", name or "")


def decode_pdf_name(v):
    v = (v or "").lstrip("/")
    return re.sub(r"#([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), v)


def norm_name(n):
    return re.sub(r"[\s_]", "", strip_subset(n)).lower()


TOKEN_RE = re.compile(r"\[|\]|/[^\s\[\]/<>()]+|\d+\s+\d+\s+R|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def tokens(s):
    return TOKEN_RE.findall(s or "")


def key_text(doc, xref, key):
    """Giá trị của key, nếu là tham chiếu gián tiếp thì đọc object đó."""
    t, v = doc.xref_get_key(xref, key)
    if t == "null":
        return None, None
    if t == "xref":
        rx = int(v.split()[0])
        return rx, doc.xref_object(rx, compressed=True)
    return None, v


def key_stream(doc, xref, key):
    t, v = doc.xref_get_key(xref, key)
    if t != "xref":
        return None, None
    rx = int(v.split()[0])
    return rx, doc.xref_stream(rx)


def parse_W(text):
    toks = [t for t in tokens(text)]
    if toks and toks[0] == "[":
        toks = toks[1:-1]
    w, i = {}, 0
    try:
        while i < len(toks):
            c = int(float(toks[i])); i += 1
            if toks[i] == "[":
                i += 1
                while toks[i] != "]":
                    w[c] = float(toks[i]); c += 1; i += 1
                i += 1
            else:
                c2 = int(float(toks[i])); val = float(toks[i + 1]); i += 2
                for cc in range(c, c2 + 1):
                    w[cc] = val
    except (IndexError, ValueError):
        pass
    return w


def build_W(wdict):
    parts, keys = [], sorted(wdict)
    i = 0
    while i < len(keys):
        start = keys[i]; vals = [wdict[start]]
        while i + 1 < len(keys) and keys[i + 1] == keys[i] + 1:
            i += 1; vals.append(wdict[keys[i]])
        parts.append(f"{start} [{' '.join(_num(v) for v in vals)}]")
        i += 1
    return "[" + " ".join(parts) + "]"


def _num(v):
    return str(int(v)) if float(v).is_integer() else f"{v:.3f}"


def _utf16(h):
    if not h:
        return ""
    if len(h) % 4:
        h = h.rjust(((len(h) + 3) // 4) * 4, "0")
    try:
        return bytes.fromhex(h).decode("utf-16-be", errors="ignore")
    except ValueError:
        return ""


def parse_tounicode(data):
    s = data.decode("latin-1", errors="ignore")
    m = {}
    for block in re.findall(r"beginbfchar(.*?)endbfchar", s, re.S):
        for a, b in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", block):
            m[int(a, 16)] = _utf16(b)
    for block in re.findall(r"beginbfrange(.*?)endbfrange", s, re.S):
        for lo, hi, rest in re.findall(
            r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[[^\]]*\]|<[0-9A-Fa-f]*>)", block
        ):
            lo, hi = int(lo, 16), int(hi, 16)
            if rest.startswith("["):
                for k, d in enumerate(re.findall(r"<([0-9A-Fa-f]*)>", rest)):
                    if lo + k <= hi:
                        m[lo + k] = _utf16(d)
            else:
                u = _utf16(rest[1:-1])
                if u:
                    for k in range(hi - lo + 1):
                        m[lo + k] = u[:-1] + chr(ord(u[-1]) + k)
    return m


def build_tounicode(m):
    items = sorted(m.items())
    out = [
        "/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def", "/CMapType 2 def",
        "1 begincodespacerange", "<0000> <FFFF>", "endcodespacerange",
    ]
    for i in range(0, len(items), 100):
        chunk = items[i:i + 100]
        out.append(f"{len(chunk)} beginbfchar")
        for c, u in chunk:
            out.append(f"<{c:04X}> <{u.encode('utf-16-be').hex().upper()}>")
        out.append("endbfchar")
    out += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
    return "\n".join(out).encode()


def _base_encoding(name):
    table = {}
    if name == "MacRomanEncoding":
        codec = "mac_roman"
    elif name == "StandardEncoding":
        for c in range(32, 127):
            table[c] = chr(c)
        table[0x27] = "\u2019"; table[0x60] = "\u2018"
        return table
    else:  # WinAnsiEncoding và mặc định
        codec = "cp1252"
    for c in range(32, 256):
        try:
            table[c] = bytes([c]).decode(codec)
        except UnicodeDecodeError:
            pass
    if codec == "cp1252":
        table[0xA0] = " "
        table[0xAD] = "-"
    return table


# ----------------------------------------------------------------------------
# Font có sẵn trong PDF
# ----------------------------------------------------------------------------
BASE14 = {
    "Helvetica": "helv", "Helvetica-Bold": "hebo", "Helvetica-Oblique": "heit",
    "Helvetica-BoldOblique": "hebi", "Times-Roman": "tiro", "Times-Bold": "tibo",
    "Times-Italic": "tiit", "Times-BoldItalic": "tibi", "Courier": "cour", "Courier-Bold": "cobo",
    "Courier-Oblique": "coit", "Courier-BoldOblique": "cobi", "Symbol": "symb", "ZapfDingbats": "zadb",
}


class PdfFont:
    def __init__(self, doc, xref):
        self.doc, self.xref = doc, xref
        self.subtype = decode_pdf_name(doc.xref_get_key(xref, "Subtype")[1])
        self.raw_basefont = decode_pdf_name(doc.xref_get_key(xref, "BaseFont")[1])
        self.name = strip_subset(self.raw_basefont)
        self.supported = True
        self.observed = {}          # ký tự -> mã, đã thấy được vẽ bằng font này
        self.prog = None
        self.upm = 1000
        self.embedded = False
        self.patched = False
        try:
            _n, ext, _t, buf = doc.extract_font(xref)
        except Exception:
            ext, buf = "", b""
        if buf:
            self.embedded = True
            if ext in ("ttf", "otf", "ttc"):
                try:
                    self.prog = TTFont(io.BytesIO(buf), lazy=True)
                    self.upm = self.prog["head"].unitsPerEm
                    self.prog_cmap = self.prog.getBestCmap() or {}
                    self.glyph_order = self.prog.getGlyphOrder()
                except Exception:
                    self.prog = None
        if self.subtype == "Type0":
            self._init_type0()
        elif self.subtype in ("TrueType", "Type1", "MMType1"):
            self._init_simple()
        else:
            self.supported = False      # Type3...

    # ---- Type0 (CID, Identity-H)
    def _init_type0(self):
        self.two_byte = True
        enc = decode_pdf_name(self.doc.xref_get_key(self.xref, "Encoding")[1])
        if enc != "Identity-H":
            self.supported = False
        t, v = self.doc.xref_get_key(self.xref, "DescendantFonts")
        if t == "xref":   # mảng DescendantFonts là object gián tiếp
            v = self.doc.xref_object(int(v.split()[0]), compressed=True)
        m = re.search(r"(\d+)\s+0\s+R", v or "")
        self.desc = int(m.group(1)) if m else None
        self.cid2gid = None   # None = Identity
        self.gid2cid = {}
        if self.desc:
            t, v = self.doc.xref_get_key(self.desc, "CIDToGIDMap")
            if t == "xref":
                data = self.doc.xref_stream(int(v.split()[0]))
                self.cid2gid = [int.from_bytes(data[i:i + 2], "big") for i in range(0, len(data) - 1, 2)]
                for cid, gid in enumerate(self.cid2gid):
                    if gid and gid not in self.gid2cid:
                        self.gid2cid[gid] = cid
            wx, wtext = key_text(self.doc, self.desc, "W")
            self.w_xref = wx
            self.widths = parse_W(wtext) if wtext else {}
            t, v = self.doc.xref_get_key(self.desc, "DW")
            self.dw = float(v) if t in ("int", "float") else 1000.0
        else:
            self.widths, self.dw, self.w_xref = {}, 1000.0, None
        tx, tdata = key_stream(self.doc, self.xref, "ToUnicode")
        self.tu_xref = tx
        self.cid2uni = parse_tounicode(tdata) if tdata else {}
        self.uni2code = {}
        for cid, u in sorted(self.cid2uni.items()):
            if len(u) == 1 and u not in self.uni2code:
                self.uni2code[u] = cid

    # ---- Font đơn (TrueType/Type1, mã 1 byte)
    def _init_simple(self):
        self.two_byte = False
        t, v = self.doc.xref_get_key(self.xref, "Encoding")
        if t == "name":
            enc = _base_encoding(decode_pdf_name(v))
        else:
            bt, bv = self.doc.xref_get_key(self.xref, "Encoding/BaseEncoding")
            enc = _base_encoding(decode_pdf_name(bv) if bt == "name" else "StandardEncoding")
            _dx, dtext = key_text(self.doc, self.xref, "Encoding/Differences")
            code = 0
            for tok in tokens(dtext or ""):
                if tok in "[]":
                    continue
                if tok.startswith("/"):
                    u = agl.toUnicode(decode_pdf_name(tok))
                    if u:
                        enc[code] = u
                    code += 1
                elif not tok.endswith("R"):
                    code = int(float(tok))
        self.enc = enc
        t, v = self.doc.xref_get_key(self.xref, "FirstChar")
        self.first = int(v) if t == "int" else 0
        wx, wtext = key_text(self.doc, self.xref, "Widths")
        self.w_xref = wx
        vals = [float(x) for x in tokens(wtext or "") if x not in "[]" and not x.endswith("R")]
        self.widths = {self.first + i: w for i, w in enumerate(vals)}
        t, v = self.doc.xref_get_key(self.xref, "FontDescriptor/MissingWidth")
        self.dw = float(v) if t in ("int", "float") else 0.0
        # 14 font chuẩn của PDF (Helvetica, Times...) thường không có bảng độ rộng: lấy số đo chuẩn
        self.base14 = None
        if not self.embedded and self.name in BASE14:
            try:
                self.base14 = pymupdf.Font(BASE14[self.name])
            except Exception:
                self.base14 = None
        self.uni2code = {}
        for c, u in sorted(enc.items()):
            if u not in self.uni2code or (self.widths.get(c, 0) > 0 and self.widths.get(self.uni2code[u], 0) == 0):
                self.uni2code[u] = c

    # ---- tra cứu
    def width(self, code):
        if self.two_byte:
            return self.widths.get(code, self.dw)
        if code not in self.widths and getattr(self, "base14", None) is not None:
            u = self.enc.get(code)
            if u and len(u) == 1 and self.base14.has_glyph(ord(u)):
                self.widths[code] = self.base14.glyph_advance(ord(u)) * 1000.0
        return self.widths.get(code, self.dw)

    def gid_for_code(self, code):
        if self.prog is None:
            return None
        if self.two_byte:
            if self.cid2gid is None:
                return code
            return self.cid2gid[code] if code < len(self.cid2gid) else None
        u = self.enc.get(code)
        gname = self.prog_cmap.get(ord(u)) if u and len(u) == 1 else None
        if gname is None:
            gname = self.prog_cmap.get(0xF000 + code) or self.prog_cmap.get(code)
        if gname is None:
            return None
        try:
            return self.prog.getGlyphID(gname)
        except Exception:
            return None

    def glyph_has_outline(self, gid):
        if self.prog is None or gid is None:
            return False
        try:
            if "glyf" in self.prog:
                g = self.prog["glyf"][self.glyph_order[gid]]
                return g.numberOfContours != 0
            return True
        except Exception:
            return False

    def code_for(self, ch, gid=None):
        """Mã để vẽ ký tự ch bằng font này (ưu tiên đúng glyph gid nếu biết)."""
        if gid is not None and self.prog is not None:
            if self.two_byte:
                if self.cid2gid is None:
                    return gid
                if gid in self.gid2cid:
                    return self.gid2cid[gid]
        if ch in self.observed:
            return self.observed[ch]
        return self.uni2code.get(ch)

    def can_render(self, ch):
        """Trả về (mã, độ_rộng_1000, cần_vá) hoặc None nếu font này không vẽ được ch."""
        if not self.supported:
            return None
        if ch in self.observed:
            c = self.observed[ch]
            return c, self.width(c), None
        code = self.uni2code.get(ch)
        if not self.embedded:
            if code is None or self.two_byte:
                return None
            w = self.width(code)
            if w > 0 or ch == " ":
                return code, w, None
            return None  # độ rộng chưa có -> cần nguồn số đo (xử lý ở tầng trên)
        if self.prog is None:
            return None
        if code is not None:
            gid = self.gid_for_code(code)
            if gid is not None and (self.glyph_has_outline(gid) or ch.isspace()):
                w = self.width(code)
                if (w and w > 0) or ch.isspace():
                    return code, w, None
                return code, self._prog_width(gid), ("width", code, gid)
            return None
        # Type0 Identity: ký tự chưa dùng nhưng chương trình font có glyph (font nhúng đầy đủ)
        if self.two_byte and self.cid2gid is None:
            gname = self.prog_cmap.get(ord(ch))
            if gname:
                gid = self.prog.getGlyphID(gname)
                if self.glyph_has_outline(gid):
                    return gid, self._prog_width(gid), ("cid", gid, gid)
        return None

    def _prog_width(self, gid):
        try:
            adv = self.prog["hmtx"][self.glyph_order[gid]][0]
            return adv * 1000.0 / self.upm
        except Exception:
            return 500.0

    def apply_patch(self, patch, ch, width):
        """Bổ sung độ rộng / ToUnicode cho ký tự mới vào font gốc."""
        kind, code = patch[0], patch[1]
        if self.two_byte:
            self.widths[code] = round(width)
            if self.w_xref:
                self.doc.update_object(self.w_xref, build_W(self.widths))
            elif self.desc:
                self.doc.xref_set_key(self.desc, "W", build_W(self.widths))
            self.cid2uni[code] = ch
            if self.tu_xref:
                self.doc.update_stream(self.tu_xref, build_tounicode(self.cid2uni))
            self.uni2code.setdefault(ch, code)
        else:
            self.widths[code] = round(width)
            lo, hi = min(self.widths), max(self.widths)
            arr = "[" + " ".join(_num(self.widths.get(c, 0)) for c in range(lo, hi + 1)) + "]"
            self.doc.xref_set_key(self.xref, "FirstChar", str(lo))
            self.doc.xref_set_key(self.xref, "LastChar", str(hi))
            if self.w_xref:
                self.doc.update_object(self.w_xref, arr)
            else:
                self.doc.xref_set_key(self.xref, "Widths", arr)
            self.first = lo
        self.observed[ch] = code
        self.patched = True

    def encode_hex(self, code):
        return f"{code:04X}" if self.two_byte else f"{code:02X}"


# ----------------------------------------------------------------------------
# Font cài trên máy
# ----------------------------------------------------------------------------
STYLE_WORDS = ("bold", "italic", "oblique", "regular", "roman", "book", "normal")
METRIC_COMPAT = {
    "timesnewroman": ["liberationserif", "tinos"],
    "arial": ["liberationsans", "arimo"],
    "couriernew": ["liberationmono", "cousine"],
    "calibri": ["carlito"],
    "cambria": ["caladea"],
}
SERIF_FALLBACK = ["timesnewroman", "liberationserif", "dejavuserif", "notoserif"]
SANS_FALLBACK = ["arial", "liberationsans", "segoeui", "dejavusans", "notosans"]


def parse_pdf_font_name(name):
    """'Times New Roman,BoldItalic' / 'TimesNewRomanPS-BoldMT' -> (họ, bold, italic, phần_còn_lại)."""
    n = strip_subset(name)
    if "," in n:
        fam, style = n.split(",", 1)
    elif "-" in n:
        fam, style = n.split("-", 1)
    else:
        fam, style = n, ""
    fam = re.sub(r"(PSMT|PS|MT)$", "", fam.replace(" ", ""))
    style_l = re.sub(r"(PSMT|MT)$", "", style).lower()
    bold = "bold" in style_l or "black" in style_l or "heavy" in style_l
    italic = "italic" in style_l or "oblique" in style_l
    rest = style_l
    for w in STYLE_WORDS:
        rest = rest.replace(w, "")
    return fam.lower(), bold, italic, re.sub(r"[^a-z]", "", rest)


SERIF_WORDS = ("times", "roman", "serif", "georgia", "cambria", "garamond", "minion", "palatino",
               "century", "bookman", "baskerville", "caslon")
SANS_WORDS = ("sans", "arial", "helvetica", "calibri", "aptos", "segoe", "verdana", "tahoma",
              "roboto", "inter", "noto sans", "frutiger", "myriad", "gothic", "grotesk")


def is_serif(fname, font=None, flags=0):
    if font is not None and font.prog is not None and "OS/2" in font.prog:
        try:
            pan = font.prog["OS/2"].panose
            if pan.bFamilyType == 2:
                if pan.bSerifStyle in (11, 12, 13):
                    return False
                if 2 <= pan.bSerifStyle <= 10:
                    return True
        except Exception:
            pass
    n = strip_subset(fname).lower()
    if "sans" in n:
        return False
    if any(w in n for w in SERIF_WORDS):
        return True
    if any(w in n for w in SANS_WORDS):
        return False
    return bool(flags & 4)


# số bước hoàn tác giữ trong bộ nhớ (máy chủ nhỏ nên giữ ít)
MAX_UNDO = int(os.environ.get("MAX_UNDO", 8 if os.environ.get("SUAPDF_SERVER") == "1" else 30))


def font_dirs():
    dirs = [os.path.join(APP_DIR, "fonts")]
    if os.environ.get("SUAPDF_SERVER") == "1":
        # máy chủ không có font Windows: dùng bộ font tự do đi kèm (Liberation ~ Times/Arial/Courier,
        # Carlito ~ Calibri, Caladea ~ Cambria, DejaVu)
        dirs.append(os.path.join(APP_DIR, "fonts_free"))
        if os.environ.get("SUAPDF_SYSTEM_FONTS", "0") != "1":   # bộ font cố định -> kết quả như lúc kiểm thử
            return [d for d in dirs if os.path.isdir(d)]
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        local = os.environ.get("LOCALAPPDATA", "")
        dirs += [os.path.join(windir, "Fonts"),
                 os.path.join(local, "Microsoft", "Windows", "Fonts"),
                 os.path.join(local, "Microsoft", "FontCache", "4", "CloudFonts")]
    elif sys.platform == "darwin":
        dirs += ["/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts"),
                 "/Applications/Microsoft Word.app/Contents/Resources/DFonts"]
    else:
        dirs += ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
                 os.path.expanduser("~/.local/share/fonts")]
    return [d for d in dirs if os.path.isdir(d)]


class SystemFonts:
    def __init__(self):
        self.entries = None
        self._cmap_cache = {}

    def _scan(self):
        cache_file = os.path.join(APP_DIR, ".font_index.json")
        files = []
        for d in font_dirs():
            for ext in ("ttf", "otf", "ttc", "TTF", "OTF", "TTC"):
                files += glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True)
        files = sorted(set(files))
        sig = [[f, os.path.getmtime(f)] for f in files]
        try:
            with open(cache_file, encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached.get("sig") == sig:
                self.entries = cached["entries"]
                return
        except Exception:
            pass
        entries = []
        for f in files:
            nfaces = 1
            if f.lower().endswith(".ttc"):
                try:
                    from fontTools.ttLib import TTCollection
                    nfaces = len(TTCollection(f, lazy=True).fonts)
                except Exception:
                    continue
            for idx in range(nfaces):
                try:
                    tt = TTFont(f, fontNumber=idx, lazy=True)
                    nm = tt["name"]
                    fam = nm.getDebugName(16) or nm.getDebugName(1) or ""
                    sub = nm.getDebugName(17) or nm.getDebugName(2) or ""
                    ps = nm.getDebugName(6) or ""
                    full = nm.getDebugName(4) or ""
                    mac = tt["head"].macStyle
                    sel = tt["OS/2"].fsSelection if "OS/2" in tt else 0
                    wt = tt["OS/2"].usWeightClass if "OS/2" in tt else 400
                    bold = bool(mac & 1) or bool(sel & 32) or wt >= 600
                    italic = bool(mac & 2) or bool(sel & 1)
                    tt.close()
                    entries.append({
                        "path": f, "index": idx, "family": fam, "sub": sub,
                        "fam_n": re.sub(r"[^a-z0-9]", "", fam.lower()),
                        "ps_n": re.sub(r"[^a-z0-9]", "", ps.lower()),
                        "full_n": re.sub(r"[^a-z0-9]", "", full.lower()),
                        "bold": bold, "italic": italic, "weight": wt,
                    })
                except Exception:
                    continue
        self.entries = entries
        try:
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump({"sig": sig, "entries": entries}, fh)
        except Exception:
            pass

    def _pick(self, fam_n, bold, italic):
        if self.entries is None:
            self._scan()
        cands = [e for e in self.entries if e["fam_n"] == fam_n]
        if not cands:
            return None
        exact = [e for e in cands if e["bold"] == bold and e["italic"] == italic]
        pool = exact or cands
        pool.sort(key=lambda e: abs(e["weight"] - (700 if bold else 400)))
        return pool[0] if exact else None

    def find_same(self, pdf_name, bold_flag=False, italic_flag=False):
        fam, b, i, rest = parse_pdf_font_name(pdf_name)
        b, i = b or bold_flag, i or italic_flag
        for key in ([fam + rest, fam] if rest else [fam]):
            e = self._pick(key, b, i)
            if e:
                return e
        if self.entries:
            target = re.sub(r"[^a-z0-9]", "", strip_subset(pdf_name).lower())
            for e in self.entries:
                if e["ps_n"] == target or e["full_n"] == target:
                    return e
        return None

    def find_fallback(self, pdf_name, serif, bold, italic, ch_needed):
        fam, b, i, _ = parse_pdf_font_name(pdf_name)
        b, i = b or bold, i or italic
        order = METRIC_COMPAT.get(fam, []) + (SERIF_FALLBACK if serif else SANS_FALLBACK)
        for key in order:
            e = self._pick(key, b, i) or self._pick(key, False, False)
            if e and all(self.has_glyph(e, c) for c in ch_needed):
                return e
        return None

    def cmap(self, e):
        k = (e["path"], e["index"])
        if k not in self._cmap_cache:
            tt = TTFont(e["path"], fontNumber=e["index"], lazy=True)
            self._cmap_cache[k] = (tt.getBestCmap() or {}, tt)
        return self._cmap_cache[k]

    def has_glyph(self, e, ch):
        if ch.isspace():
            return True
        return ord(ch) in self.cmap(e)[0]

    def glyph_info(self, e, ch):
        cmap, tt = self.cmap(e)
        gname = cmap.get(ord(ch))
        if gname is None:
            return None
        gid = tt.getGlyphID(gname)
        adv = tt["hmtx"][gname][0] * 1000.0 / tt["head"].unitsPerEm
        return gid, adv

    def subset_buffer(self, e, gids):
        tt = TTFont(e["path"], fontNumber=e["index"])
        opts = ft_subset.Options()
        opts.retain_gids = True
        opts.notdef_outline = True
        opts.name_IDs = ["*"]
        opts.layout_features = []
        opts.drop_tables += ["GSUB", "GPOS", "GDEF", "DSIG"]
        sub = ft_subset.Subsetter(opts)
        sub.populate(gids=sorted(set(gids) | {0}))
        sub.subset(tt)
        bio = io.BytesIO()
        tt.save(bio)
        return bio.getvalue()


SYSFONTS = SystemFonts()


# ----------------------------------------------------------------------------
# Trình sửa tài liệu
# ----------------------------------------------------------------------------
DIR_ROT = {(1, 0): 0, (0, -1): 90, (0, 1): -90, (-1, 0): 180}   # hướng chữ -> góc quay về nằm ngang


class Char:
    __slots__ = ("ch", "gid", "x", "y", "bbox", "size", "color", "cs", "font", "fname",
                 "flags", "synthetic", "seq", "lig", "rot")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def adv(self):
        return self.bbox[2] - self.x


class EditError(Exception):
    pass


class Editor:
    def __init__(self, data, filename="document.pdf"):
        self.filename = filename
        self.history = []
        self._load(data)

    # ---- nạp / hoàn tác
    def _load(self, data):
        self.doc = pymupdf.open(stream=data, filetype="pdf")
        if self.doc.needs_pass:
            raise EditError("File PDF có mật khẩu, chưa hỗ trợ.")
        self.fonts = {}
        self.version = getattr(self, "version", 0) + 1
        self._lines_cache = {}
        for pno in range(self.doc.page_count):
            self._page_chars(pno)   # quét để biết ký tự nào font nào đã có

    def snapshot(self):
        self.history.append(self.doc.tobytes())
        if len(self.history) > MAX_UNDO:
            self.history.pop(0)

    def undo(self):
        if not self.history:
            return False
        self._load(self.history.pop())
        return True

    def save_bytes(self):
        return self.doc.tobytes(garbage=3, deflate=True)

    # ---- font
    def font(self, xref):
        if xref not in self.fonts:
            self.fonts[xref] = PdfFont(self.doc, xref)
        return self.fonts[xref]

    def fonts_named(self, name):
        n = norm_name(name)
        return [f for f in self.fonts.values() if norm_name(f.name) == n and f.supported]

    def _identify(self, page_fonts, span, chars=None):
        name = norm_name(span["font"])
        cands = [self.font(x) for x in page_fonts if norm_name(self.font(x).name) == name]
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        best, best_score = None, None
        size = span["size"] or 1
        for f in cands:
            score = 0
            for u, gid, o, b in (chars if chars is not None else span["chars"]):
                ch = chr(u)
                if gid is not None and gid < 0:
                    continue
                if f.two_byte:
                    code = f.code_for(ch, gid) if f.prog is not None else f.uni2code.get(ch)
                    if code is None:
                        score -= 5
                        continue
                    # mã suy từ glyph phải khớp ToUnicode, nếu không thì glyph này không thuộc font này
                    score += 3 if f.cid2uni.get(code) == ch else -4
                else:
                    code = f.uni2code.get(ch)
                    if code is None:
                        score -= 5
                        continue
                    if f.prog is not None:
                        score += 2 if f.gid_for_code(code) == gid else -2
                adv = (b[2] - o[0]) / size * 1000.0
                w = f.width(code)
                score += 3 if abs(adv - w) <= max(3.0, 0.02 * w) else -3
            if best_score is None or score > best_score:
                best, best_score = f, score
        return best

    def _page_chars(self, pno):
        page = self.doc[pno]
        page_fonts = [f[0] for f in page.get_fonts(full=True)]
        chars = []
        for s in page.get_texttrace():
            if s["type"] != 0 or s["wmode"] != 0 or s["opacity"] == 0:
                continue
            dx, dy = round(s["dir"][0]), round(s["dir"][1])
            card = abs(s["dir"][0] - dx) < 1e-3 and abs(s["dir"][1] - dy) < 1e-3 and (dx, dy) in DIR_ROT
            rot = DIR_ROT[(dx, dy)] if card else None
            if card and rot != 0:
                R = pymupdf.Matrix(rot)
                span_chars = [(u, gid, tuple(pymupdf.Point(o) * R), tuple(pymupdf.Rect(b) * R))
                              for u, gid, o, b in s["chars"]]
            else:
                span_chars = s["chars"]
            f = self._identify(page_fonts, s, span_chars) if card else None
            head = None
            for u, gid, origin, bbox in span_chars:
                ch = chr(u) if u >= 0 else "\ufffd"
                lig = gid is not None and gid < 0 and head is not None
                if lig:   # ký tự thứ 2.. của chữ ghép (vd "fl"): dùng chung glyph với ký tự đầu
                    origin, bbox = (head.x, head.y), head.bbox
                if f is not None and u > 0 and not lig and ch not in f.observed:
                    code = f.code_for(ch, gid)
                    if code is not None:
                        f.observed[ch] = code
                c = Char(ch=ch, gid=gid, x=origin[0], y=origin[1], bbox=bbox,
                         size=s["size"], color=s["color"], cs=s["colorspace"],
                         font=f, fname=s["font"], flags=s["flags"],
                         synthetic=False, seq=s["seqno"], lig=lig, rot=rot)
                chars.append(c)
                if not lig:
                    head = c
        return chars

    # ---- tách dòng
    def lines(self, pno):
        key = (pno, self.version)
        if key in self._lines_cache:
            return self._lines_cache[key]
        chars = self._page_chars(pno)
        runs = []
        for c in chars:
            r = runs[-1] if runs else None
            if r and c.rot == r[-1].rot and abs(c.y - r[-1].y) < 0.25 * max(c.size, r[-1].size) \
                    and c.x >= r[-1].x - 0.5 and c.x - r[-1].bbox[2] < 1.2 * max(c.size, r[-1].size):
                r.append(c)
            else:
                runs.append([c])
        # ghép các đoạn cùng đường chân chữ nằm sát nhau (kể cả chữ mới ghi thêm cuối trang)
        merged = True
        while merged:
            merged = False
            runs.sort(key=lambda r: (round(r[0].y, 1), r[0].x))
            for i in range(len(runs)):
                for j in range(len(runs)):
                    if i == j or not runs[i] or not runs[j]:
                        continue
                    a, b = runs[i], runs[j]
                    sz = max(a[-1].size, b[0].size)
                    if a[0].rot == b[0].rot and abs(a[-1].y - b[0].y) < 0.25 * sz and a[0].x <= b[0].x \
                            and -0.3 * sz <= b[0].x - a[-1].bbox[2] < 1.2 * sz:
                        a.extend(b); runs[j] = []
                        merged = True
            runs = [r for r in runs if r]
        lines = []
        for r in runs:
            r.sort(key=lambda c: c.x)
            out = [r[0]]
            for c in r[1:]:
                prev = out[-1]
                gap = c.x - prev.bbox[2]
                if not prev.ch.isspace() and not c.ch.isspace() and gap > 0.15 * c.size:
                    out.append(Char(ch=" ", gid=None, x=prev.bbox[2], y=prev.y,
                                    bbox=(prev.bbox[2], prev.bbox[1], c.x, prev.bbox[3]),
                                    size=prev.size, color=prev.color, cs=prev.cs, font=None,
                                    fname=prev.fname, flags=prev.flags, synthetic=True, seq=prev.seq,
                                    rot=prev.rot))
                out.append(c)
            text = "".join(c.ch for c in out)
            if not text.strip():
                continue
            x0 = min(c.bbox[0] for c in out); y0 = min(c.bbox[1] for c in out)
            x1 = max(c.bbox[2] for c in out); y1 = max(c.bbox[3] for c in out)
            editable = all(c.font is not None or c.synthetic or c.ch.isspace() for c in out) \
                and "\ufffd" not in text
            rot = r[0].rot
            pb = pymupdf.Rect(x0, y0, x1, y1) * ~pymupdf.Matrix(rot) if rot is not None else pymupdf.Rect(x0, y0, x1, y1)
            lines.append({"chars": out, "text": text, "bbox": [x0, y0, x1, y1], "rot": rot,
                          "bbox_page": [pb.x0, pb.y0, pb.x1, pb.y1],
                          "size": round(statistics.median(c.size for c in out), 2),
                          "fonts": sorted({c.fname for c in out if not c.synthetic}),
                          "editable": editable})
        lines.sort(key=lambda l: (round(l["bbox_page"][1], 0), l["bbox_page"][0]))
        for i, l in enumerate(lines):
            l["id"] = i
        self._lines_cache[key] = lines
        return lines

    def lines_public(self, pno):
        """Tọa độ trả về là tọa độ HIỂN THỊ (đã tính góc xoay trang)."""
        rm = self.doc[pno].rotation_matrix
        out = []
        for l in self.lines(pno):
            d = pymupdf.Rect(l["bbox_page"]) * rm
            pub = {k: v for k, v in l.items() if k not in ("chars", "bbox_page", "bbox")}
            pub["bbox"] = [d.x0, d.y0, d.x1, d.y1]
            out.append(pub)
        return out

    # ---- sửa một dòng
    def edit_line(self, pno, line_id, new_text, allow_fallback=False):
        lines = self.lines(pno)
        if line_id < 0 or line_id >= len(lines):
            raise EditError("Dòng không tồn tại (trang đã thay đổi, hãy tải lại).")
        line = lines[line_id]
        if not line["editable"]:
            raise EditError("Dòng này dùng loại font chưa hỗ trợ sửa.")
        new_text = unicodedata.normalize("NFC", new_text.replace("\n", " "))
        old = line["chars"]
        old_text = line["text"]
        if new_text == old_text:
            return {"ok": True, "changed": False}

        # phần đầu & cuối giống nhau
        p = 0
        while p < min(len(old_text), len(new_text)) and old_text[p] == new_text[p]:
            p += 1
        s = 0
        while s < min(len(old_text), len(new_text)) - p and old_text[-1 - s] == new_text[-1 - s]:
            s += 1
        while p > 0 and p < len(old) and old[p].lig:
            p -= 1
        while s > 0 and old[len(old) - s].lig:
            s -= 1
        mid_new = new_text[p:len(new_text) - s]
        mid_old = old[p:len(old) - s]
        suf_old = old[len(old) - s:] if s else []
        redact = old[p:]

        line_rot = line["rot"] or 0
        # khoảng giãn chữ (canh đều) của dòng gốc
        extras = [old[i + 1].x - old[i].x - old[i].adv for i in range(len(old) - 1)
                  if old[i].ch == " " and not old[i].synthetic]
        extra_space = statistics.median(extras) if extras else 0.0
        extra_space = max(-0.12 * old[0].size, extra_space)

        # chọn font cho từng ký tự mới
        placements, report = [], defaultdict(set)
        missing = []
        sys_needed = defaultdict(list)   # entry_key -> [(index, ch)]
        sys_entries = {}
        for j, ch in enumerate(mid_new):
            if mid_old:
                tmpl = mid_old[min(len(mid_old) - 1, j * len(mid_old) // len(mid_new))]
            elif p > 0:
                tmpl = old[p - 1]
            else:
                tmpl = old[0]
            if tmpl.synthetic:
                tmpl = next((c for c in old if not c.synthetic), tmpl)
            pl = self._resolve(ch, tmpl)
            if pl is None:
                missing.append(ch)
                pl = {"ch": ch, "tmpl": tmpl, "kind": "pending"}
            placements.append(pl)
            if pl["kind"] in ("system", "fallback", "pending"):
                if pl["kind"] == "pending":
                    serif = is_serif(tmpl.fname, tmpl.font, tmpl.flags)
                    e = SYSFONTS.find_fallback(tmpl.fname, serif, bool(tmpl.flags & 16),
                                               bool(tmpl.flags & 2), [ch])
                    if e is None:
                        raise EditError(f"Không tìm được font nào có ký tự '{ch}'.")
                    pl.update(kind="fallback", entry=e)
                k = (pl["entry"]["path"], pl["entry"]["index"])
                sys_entries[k] = pl["entry"]
                sys_needed[k].append(ch)

        fallback_chars = sorted({pl["ch"] for pl in placements if pl["kind"] == "fallback"})
        if fallback_chars and not allow_fallback:
            fbp = next(pl for pl in placements if pl["kind"] == "fallback")
            fb = fbp["entry"]
            return {"ok": False, "needs_confirm": True, "missing": fallback_chars,
                    "font_goc": fbp["tmpl"].fname, "font_thay_the": fb["family"] + " " + fb["sub"]}

        # ---- bắt đầu ghi
        page = self.doc[pno]
        before_texts = sorted(re.sub(r"\s+", " ", l["text"]).strip()
                              for l in lines if l["id"] != line_id)
        self.snapshot()
        collateral = []
        try:
            # 1. xóa thật phần chữ từ vị trí thay đổi đến hết dòng
            if redact:
                real = [c for c in redact if not c.synthetic and not c.lig]
                if real:
                    sz = max(c.size for c in real)
                    y = real[0].y
                    x0 = real[0].x + 0.3 * max(real[0].adv, 0.2)
                    last = real[-1]
                    x1 = last.x + max(0.5 * last.adv, 0.2)
                    rect = pymupdf.Rect(x0, y - 0.55 * sz, x1, y - 0.15 * sz) * ~pymupdf.Matrix(line_rot)
                    keyf = lambda c: (c.ch, round(c.x, 2), round(c.y, 2), c.rot)
                    before_chars = {keyf(c): c for c in self._page_chars(pno) if not c.lig}
                    page.add_redact_annot(rect, fill=False)
                    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                                          graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                                          text=pymupdf.PDF_REDACT_TEXT_REMOVE)
                    after_keys = {keyf(c) for c in self._page_chars(pno) if not c.lig}
                    removed = set(before_chars) - after_keys
                    intended = {keyf(c) for c in real}
                    if intended - removed:
                        raise EditError("Không xóa được hết chữ cũ ở dòng này.")
                    collateral = [before_chars[k] for k in removed - intended]

            # 2. nhúng phần font hệ thống cần dùng (nếu có)
            sys_font_res = {}
            for k, e in sys_entries.items():
                gids = []
                for ch in set(sys_needed[k]):
                    gi = SYSFONTS.glyph_info(e, ch)
                    if gi:
                        gids.append(gi[0])
                buf = SYSFONTS.subset_buffer(e, gids)
                rname = f"EdS{self.doc.xref_length()}"
                fx = page.insert_font(fontname=rname, fontbuffer=buf)
                sys_font_res[k] = (rname, fx)

            # 3. vị trí từng ký tự
            items = []   # [x, glyph(res, size, color, cs, hex) | None, is_space, y]
            if p < len(old):
                cursor = old[p].x
            else:
                cursor = old[-1].x + old[-1].adv
            y = old[0].y
            for pl in placements:
                tmpl = pl["tmpl"]
                size = tmpl.size
                ch = pl["ch"]
                g = None
                if pl["kind"] == "original":
                    f = pl["font"]
                    if pl.get("patch"):
                        f.apply_patch(pl["patch"], ch, pl["w"])
                    res = self._font_res(page, f.xref)
                    g = (res, size, tmpl.color, tmpl.cs, f.encode_hex(pl["code"]))
                    report[f"{f.name} (font gốc trong file)"].add(ch)
                    adv = pl["w"] * size / 1000.0
                elif pl["kind"] == "space":
                    adv = pl["w"] * size / 1000.0
                else:
                    e = pl["entry"]
                    gi = SYSFONTS.glyph_info(e, ch)
                    rname, _fx = sys_font_res[(e["path"], e["index"])]
                    gid, advu = gi if gi else (0, 500)
                    if not ch.isspace():
                        g = (rname, size, tmpl.color, tmpl.cs, f"{gid:04X}")
                    label = "font cùng tên trên máy" if pl["kind"] == "system" else "FONT THAY THẾ"
                    report[f"{e['family']} {e['sub']} ({label})"].add(ch)
                    adv = advu * size / 1000.0
                items.append([cursor, g, ch == " ", y, line_rot])
                cursor += adv + (extra_space if ch == " " else 0.0)
            delta = cursor - (suf_old[0].x if suf_old else cursor)
            for c in suf_old:
                if c.lig:
                    continue
                g = None
                if not c.synthetic:
                    f = c.font
                    code = f.code_for(c.ch, c.gid)
                    if code is None:
                        raise EditError(f"Không mã hóa lại được ký tự '{c.ch}'.")
                    g = (self._font_res(page, f.xref), c.size, c.color, c.cs, f.encode_hex(code))
                items.append([c.x + delta, g, c.ch.isspace(), c.y, line_rot])

            # ký tự khác bị xóa theo (chỉ số trên/dưới nằm trong vùng xóa): vẽ lại, dịch theo dòng
            x_change = old[p].x if p < len(old) else cursor
            for c in collateral:
                if c.font is None:
                    raise EditError("Vùng sửa chạm vào chữ chưa hỗ trợ (chữ xoay/dọc).")
                code = c.font.code_for(c.ch, c.gid)
                if code is None:
                    raise EditError(f"Không mã hóa lại được ký tự '{c.ch}'.")
                g = (self._font_res(page, c.font.xref), c.size, c.color, c.cs, c.font.encode_hex(code))
                same = (c.rot or 0) == line_rot
                items.append([c.x + (delta if same and c.x >= x_change - 0.01 else 0.0), g, False, c.y,
                              c.rot or 0, not same])
            items.sort(key=lambda it: it[0])

            # dòng canh đều 2 bên: giãn/co khoảng trắng để giữ đúng lề phải cũ
            size0 = old[0].size
            justified = abs(extra_space) > 0.04 * size0
            if justified and suf_old and abs(delta) > 0.05:
                n_sp = sum(1 for it in items if it[2])
                if n_sp and abs(delta) / n_sp < 0.35 * size0:
                    step, k_sp = -delta / n_sp, 0
                    for it in items:
                        if len(it) > 5 and it[5]:
                            continue          # ký tự khác hướng: không dịch
                        it[0] += k_sp * step
                        if it[2]:
                            k_sp += 1
                    delta = 0.0
            glyphs = [(it[0], it[3], it[4]) + it[1] for it in items if it[1] is not None]

            # 4. ghi nội dung mới
            if glyphs:
                self._append_text(page, glyphs)

            # 5. kiểm tra lại: các dòng khác không bị ảnh hưởng
            self.version += 1
            self._lines_cache = {}
            after = self.lines(pno)
            after_texts = sorted(re.sub(r"\s+", " ", l["text"]).strip() for l in after)
            expect = sorted(before_texts + ([re.sub(r"\s+", " ", new_text).strip()]
                                            if new_text.strip() else []))
            if after_texts != expect:
                diff_after = [t for t in after_texts if t not in expect]
                diff_exp = [t for t in expect if t not in after_texts]
                raise EditError("Kiểm tra sau khi sửa không khớp, đã hoàn tác. "
                                f"Mong đợi: {diff_exp[:3]} | Thực tế: {diff_after[:3]}")
        except Exception as ex:
            self.undo()
            if isinstance(ex, EditError):
                raise
            raise EditError(f"Lỗi khi ghi: {ex}")

        new_line_end = cursor if not suf_old else suf_old[-1].bbox[2] + delta
        return {
            "ok": True, "changed": True,
            "fonts": [{"font": k, "chars": "".join(sorted(v))} for k, v in report.items()],
            "width_delta": round(delta if suf_old else new_line_end - line["bbox"][2], 2),
            "fallback_used": bool(fallback_chars),
        }

    def _resolve(self, ch, tmpl):
        """Chọn cách vẽ ký tự ch theo kiểu chữ của ký tự mẫu tmpl."""
        fonts = []
        if tmpl.font is not None:
            fonts.append(tmpl.font)
        fonts += [f for f in self.fonts_named(tmpl.fname) if f is not tmpl.font]
        for f in fonts:
            r = f.can_render(ch)
            if r:
                code, w, patch = r
                return {"ch": ch, "tmpl": tmpl, "kind": "original", "font": f,
                        "code": code, "w": w, "patch": patch}
        bold, italic = bool(tmpl.flags & 16), bool(tmpl.flags & 2)
        same = SYSFONTS.find_same(tmpl.fname, bold, italic)
        # nguồn số đo độ rộng: font cùng tên, hoặc font "cùng số đo" (vd Liberation Serif = Times New Roman)
        width_src = same
        if width_src is None:
            fam = parse_pdf_font_name(tmpl.fname)[0]
            for alias in METRIC_COMPAT.get(fam, []):
                width_src = SYSFONTS._pick(alias, bold, italic) or SYSFONTS._pick(alias, False, False)
                if width_src:
                    break
        # font gốc không nhúng: chỉ cần bổ sung độ rộng là viewer vẽ bằng font máy
        for f in fonts:
            if not f.embedded and not f.two_byte and ch in f.uni2code and width_src:
                gi = SYSFONTS.glyph_info(width_src, ch)
                if gi:
                    code = f.uni2code[ch]
                    return {"ch": ch, "tmpl": tmpl, "kind": "original", "font": f,
                            "code": code, "w": gi[1], "patch": ("width", code, None)}
        if ch.isspace():
            sw = None
            for f in fonts:
                r = f.can_render(" ")
                if r:
                    sw = r[1]; break
            if sw is None and same:
                gi = SYSFONTS.glyph_info(same, " ")
                sw = gi[1] if gi else None
            return {"ch": ch, "tmpl": tmpl, "kind": "space", "w": sw or 250.0}
        if same and SYSFONTS.has_glyph(same, ch):
            return {"ch": ch, "tmpl": tmpl, "kind": "system", "entry": same}
        return None

    def check_chars(self, pno, line_id, new_text):
        """Xem trước: ký tự nào sẽ phải dùng font khác (không ghi gì)."""
        line = self.lines(pno)[line_id]
        out = {"original": [], "system": [], "fallback": []}
        tmpl = next((c for c in line["chars"] if not c.synthetic), line["chars"][0])
        for ch in sorted(set(new_text) - set(line["text"])):
            pl = self._resolve(ch, tmpl)
            kind = "fallback" if pl is None else ("original" if pl["kind"] in ("original", "space") else "system")
            out[kind].append(ch)
        return out

    def _font_res(self, page, xref):
        rname = f"EdF{xref}"
        t, v = self.doc.xref_get_key(page.xref, "Resources")
        if t == "null":
            raise EditError("Trang không có Resources riêng (chưa hỗ trợ).")
        holder = page.xref if t == "dict" else int(v.split()[0])
        prefix = "Resources/Font" if t == "dict" else "Font"
        ft, fv = self.doc.xref_get_key(holder, prefix)
        if ft == "xref":
            holder, prefix = int(fv.split()[0]), ""
        key = f"{prefix}/{rname}" if prefix else rname
        if self.doc.xref_get_key(holder, key)[0] == "null":
            self.doc.xref_set_key(holder, key, f"{xref} 0 R")
        return rname

    def _append_text(self, page, glyphs):
        if not page.is_wrapped:
            page.wrap_contents()
        m = ~page.transformation_matrix
        ops = ["q"]
        cur = None
        for x, y, rot, res, size, color, cs, hx in glyphs:
            state = (res, round(size, 4), tuple(color or ()), cs)
            if state != cur:
                if cur is not None:
                    ops.append("ET")
                ops.append("BT")
                ops.append(f"/{res} {size:.4f} Tf")
                col = " ".join(f"{v:.4f}" for v in (color or (0,)))
                ops.append(f"{col} " + {1: "g", 3: "rg", 4: "k"}.get(cs, "g"))
                cur = state
            inv = ~pymupdf.Matrix(rot)
            pt = pymupdf.Point(x, y) * inv * m
            d = (pymupdf.Point(1, 0) * inv) - (pymupdf.Point(0, 0) * inv)   # hướng chữ trên trang
            c_, s_ = round(d.x, 6), round(-d.y, 6)                           # đổi sang hệ PDF (y hướng lên)
            ops.append(f"{c_:g} {s_:g} {-s_:g} {c_:g} {pt.x:.4f} {pt.y:.4f} Tm <{hx}> Tj")
        ops += ["ET", "Q"]
        data = "\n".join(ops).encode()
        xref = self.doc.get_new_xref()
        self.doc.update_object(xref, "<<>>")
        self.doc.update_stream(xref, data)
        conts = page.get_contents()
        self.doc.xref_set_key(page.xref, "Contents",
                              "[" + " ".join(f"{c} 0 R" for c in conts + [xref]) + "]")
