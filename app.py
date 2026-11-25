"""
Sửa PDF — sửa chữ trực tiếp trong PDF (kể cả chữ trong ảnh).

Chạy trên máy:  python app.py   rồi mở  http://127.0.0.1:5050
  -> file chỉ nằm trong bộ nhớ của chương trình trên máy bạn, không gửi đi đâu.
Chạy trên máy chủ (đặt biến môi trường SUAPDF_SERVER=1, dùng gunicorn):
  -> file nằm trong bộ nhớ máy chủ, tự xóa khi không dùng quá SESSION_TTL_MIN phút.
"""
import io
import os
import time
import uuid
import threading
import webbrowser

import pymupdf
from flask import Flask, request, jsonify, send_file, send_from_directory, abort
from werkzeug.exceptions import HTTPException

from engine import Editor, EditError
import imageedit

SERVER = os.environ.get("SUAPDF_SERVER") == "1"
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", 25 if SERVER else 200))
SESSION_TTL = int(os.environ.get("SESSION_TTL_MIN", 30 if SERVER else 24 * 60)) * 60
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", 6 if SERVER else 1000))
MAX_PAGES = int(os.environ.get("MAX_PAGES", 80 if SERVER else 100000))
SOURCE_URL = os.environ.get("SOURCE_URL", "https://github.com/Letter2209/sua-pdf")

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
SESSIONS = {}
LAST_USED = {}
LOCK = threading.Lock()


def _cleanup(make_room=False):
    """Bỏ các phiên không dùng quá hạn; khi cần chỗ thì bỏ phiên cũ nhất (giữ bộ nhớ máy chủ trong giới hạn)."""
    now = time.time()
    for sid in [s for s, t in list(LAST_USED.items()) if now - t > SESSION_TTL]:
        SESSIONS.pop(sid, None); LAST_USED.pop(sid, None)
    while make_room and len(SESSIONS) >= MAX_SESSIONS and LAST_USED:
        oldest = min(LAST_USED, key=LAST_USED.get)
        SESSIONS.pop(oldest, None); LAST_USED.pop(oldest, None)


def _janitor():
    while True:
        time.sleep(60)
        with LOCK:
            _cleanup()


threading.Thread(target=_janitor, daemon=True).start()


def get(sid):
    ed = SESSIONS.get(sid)
    if ed is None:
        abort(404, "Phiên làm việc đã hết hạn, hãy mở lại file.")
    LAST_USED[sid] = time.time()
    return ed


def info(ed):
    return {"pages": [{"w": p.rect.width, "h": p.rect.height} for p in ed.doc],
            "name": ed.filename, "version": ed.version, "can_undo": bool(ed.history)}


@app.errorhandler(EditError)
def on_edit_error(e):
    return jsonify({"ok": False, "error": str(e)}), 400


@app.errorhandler(413)
def on_too_large(e):
    return jsonify({"ok": False, "error": f"File quá lớn (tối đa {MAX_MB} MB)."}), 413


@app.errorhandler(HTTPException)
def on_http_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": e.description or e.name}), e.code
    return e


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/api/config")
def config():
    return jsonify({"server": SERVER, "max_mb": MAX_MB, "ttl_min": SESSION_TTL // 60,
                    "max_pages": MAX_PAGES, "source_url": SOURCE_URL})


@app.get("/")
def index():
    resp = send_from_directory(app.static_folder, "index.html")
    resp.headers["Cache-Control"] = "no-store"   # luôn lấy giao diện mới nhất
    return resp


@app.post("/api/open")
def open_pdf():
    f = request.files.get("file")
    if not f:
        abort(400, "Chưa chọn file.")
    try:
        ed = Editor(f.read(), os.path.basename(f.filename or "document.pdf"))
    except Exception:
        abort(400, "Không mở được file này. Hãy chọn một file PDF hợp lệ (không đặt mật khẩu).")
    if ed.doc.page_count > MAX_PAGES:
        abort(400, f"File có {ed.doc.page_count} trang, bản trên máy chủ chỉ nhận tối đa {MAX_PAGES} trang.")
    sid = uuid.uuid4().hex
    with LOCK:
        _cleanup(make_room=True)
        SESSIONS[sid] = ed
        LAST_USED[sid] = time.time()
    return jsonify({"sid": sid, **info(ed)})


@app.get("/api/<sid>/page/<int:pno>.png")
def page_png(sid, pno):
    ed = get(sid)
    scale = min(4.0, max(0.5, float(request.args.get("s", 2))))
    with LOCK:
        pix = ed.doc[pno].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        data = pix.tobytes("png")
    resp = send_file(io.BytesIO(data), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/<sid>/page/<int:pno>/items")
def page_items(sid, pno):
    ed = get(sid)
    with LOCK:
        return jsonify({"lines": ed.lines_public(pno),
                        "images": imageedit.page_images(ed.doc, pno),
                        "version": ed.version})


@app.post("/api/<sid>/edit")
def edit(sid):
    ed = get(sid)
    j = request.get_json()
    with LOCK:
        res = ed.edit_line(int(j["page"]), int(j["line"]), j["text"], bool(j.get("allow_fallback")))
    return jsonify({**res, **info(ed)})


@app.post("/api/<sid>/image/crop")
def image_crop(sid):
    ed = get(sid)
    j = request.get_json()
    with LOCK:
        data = imageedit.crop_png(ed.doc, int(j["page"]), j["box"])
    return send_file(io.BytesIO(data), mimetype="image/png")


@app.post("/api/<sid>/image/ocrsrc")
def image_ocrsrc(sid):
    ed = get(sid)
    j = request.get_json()
    with LOCK:
        data = imageedit.ocr_png(ed.doc, int(j["page"]), j["box"])
    return send_file(io.BytesIO(data), mimetype="image/png")


@app.post("/api/<sid>/image/edit")
def image_edit(sid):
    ed = get(sid)
    j = request.get_json()
    commit = bool(j.get("commit"))
    with LOCK:
        if commit:
            ed.snapshot()
        try:
            res = imageedit.edit_image_text(ed.doc, int(j["page"]), j["box"], j.get("old_text", ""),
                                            j.get("new_text", ""), j.get("font") or None, commit,
                                            int(j.get("text_rot") or 0), j.get("align") or "auto",
                                            bool(j.get("fit", True)))
        except Exception:
            if commit:
                ed.undo()
            raise
        if commit:
            ed.version += 1
            ed._lines_cache = {}
    return jsonify({**res, **info(ed)})


@app.get("/api/<sid>/fonts")
def fonts(sid):
    get(sid)
    return jsonify(sorted({e["family"] for e in imageedit._candidates()}))


@app.post("/api/<sid>/rotate")
def rotate(sid):
    """Xoay trang 90 độ (lưu vào file, hoàn tác được). page = số trang hoặc "all"."""
    ed = get(sid)
    j = request.get_json()
    delta = int(j.get("delta", 90))
    with LOCK:
        pages = range(ed.doc.page_count) if j.get("page") == "all" else [int(j["page"])]
        ed.snapshot()
        for p in pages:
            pg = ed.doc[p]
            pg.set_rotation((pg.rotation + delta) % 360)
        ed.version += 1
        ed._lines_cache = {}
    return jsonify({"ok": True, **info(ed)})


@app.post("/api/<sid>/undo")
def undo(sid):
    ed = get(sid)
    with LOCK:
        ok = ed.undo()
    return jsonify({"ok": ok, **info(ed)})


@app.get("/api/<sid>/download")
def download(sid):
    ed = get(sid)
    with LOCK:
        data = ed.save_bytes()
    base, _ = os.path.splitext(ed.filename)
    return send_file(io.BytesIO(data), mimetype="application/pdf", as_attachment=True,
                     download_name=f"{base}_da_sua.pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    host = os.environ.get("HOST", "127.0.0.1")
    url = f"http://127.0.0.1:{port}"
    print(f"\n  Sửa PDF đang chạy tại {url}  (Ctrl+C để tắt)\n")
    if not os.environ.get("NO_BROWSER") and not SERVER:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, threaded=True)
