# Sửa PDF (bản thử)

Sửa chữ trực tiếp trong PDF mà không phải chuyển sang Word — kể cả chữ nằm trong ảnh (bản scan, ảnh chụp màn hình).

Có 2 cách dùng:

| | Bản trên mạng | Chạy trên máy của bạn |
|---|---|---|
| Cách mở | Mở link web | Tải code về, chạy `chay.bat` |
| File PDF | Xử lý trên máy chủ, không lưu lại, tự xóa sau 30 phút không dùng | Không rời khỏi máy bạn |
| Font khi sửa chữ trong ảnh | Chỉ có font miễn phí (Liberation, DejaVu, Carlito, Caladea) | Mọi font cài trên máy → giống bản gốc nhất |
| Giới hạn | 25 MB, 80 trang; máy chủ miễn phí chậm, ngủ sau 15 phút không dùng | Không giới hạn |

## Chạy trên máy của bạn

Cần Python 3.10 trở lên.

**Windows:** bấm đúp `chay.bat`. Lần đầu chương trình tự cài thư viện, sau đó tự mở trình duyệt.

**Cách khác (mọi hệ điều hành):**
```
pip install -r requirements.txt
python app.py
```
Sau đó mở http://127.0.0.1:5050

File PDF chỉ được xử lý trên máy bạn, không gửi lên mạng. Riêng chức năng nhận dạng chữ trong ảnh cần internet ở lần dùng đầu tiên để tải bộ nhận dạng.

## Cách dùng

1. **Sửa chữ:** bấm vào một dòng chữ, gõ nội dung mới, nhấn Enter.
2. **Sửa chữ trong ảnh:** chuyển sang chế độ "Sửa chữ trong ảnh", kéo khung quanh chữ trên ảnh, kiểm tra lại chữ gốc, gõ chữ mới, bấm "Xem trước" rồi "Áp dụng".
   - Khung có thể chứa nhiều dòng: mỗi dòng trong ảnh tương ứng một dòng trong ô chữ (Enter để xuống dòng). Dòng nào giữ nguyên chữ thì giữ nguyên điểm ảnh gốc.
   - Đường kẻ bảng trong khung được giữ lại. Chữ mới dài hơn khung sẽ được thu nhỏ cho vừa (tắt được).
3. **Xoay trang:** nút ⟲ ⟳ cạnh mỗi trang (xoay 90°), hoặc "⟳ Xoay tất cả" trên thanh công cụ. Hướng mới được lưu vào file khi tải về.
4. **Chữ nằm dọc trong ảnh:** chương trình tự thử nhận dạng ở các chiều (ngang, đọc từ dưới lên, đọc từ trên xuống, lộn ngược) và chọn chiều tin cậy nhất. Nếu đoán sai, chọn lại ở ô "Chiều chữ trong khung".
5. **Hoàn tác:** nút Hoàn tác hoặc Ctrl+Z.
6. **Tải về PDF:** file mới có đuôi `_da_sua.pdf`.

## Cách chương trình giữ đúng font

- Phần chữ phía trước chỗ sửa: giữ nguyên, không bị đụng tới.
- Chữ cũ bị xóa thật khỏi file (không phải che bằng khung trắng), rồi chữ mới được ghi bằng **chính font có sẵn trong file**.
- Dòng canh đều hai bên: chương trình tự giãn khoảng trắng để giữ đúng lề phải.
- Nếu font trong file **không có** ký tự bạn gõ (file thường chỉ nhúng các ký tự đã dùng):
  1. Tìm font cùng tên trên máy bạn → dùng font đó, hình chữ vẫn giống y.
  2. Nếu máy không có → **hỏi bạn trước** khi dùng font thay thế.
- Muốn dùng font không cài trên máy: chép file `.ttf` / `.otf` vào thư mục `fonts/` cạnh `app.py`.

## Giới hạn của bản thử

- Sửa theo từng dòng. Chữ dài hơn không tự xuống dòng như Word.
- Dòng chứa công thức toán (chỉ số trên/dưới phức tạp) có thể không sửa được. Khi đó chương trình báo lỗi và tự hoàn tác, file không bị hỏng.
- Chữ xoay xiên (không phải 90°/180°/270°) và PDF có mật khẩu chưa hỗ trợ.
- Chữ trong ảnh chỉ **gần giống**: chương trình thử các font có trên máy và chọn font giống nhất. Máy có đúng font gốc (ví dụ Segoe UI cho ảnh chụp màn hình Windows) thì kết quả giống nhất.

## Đưa lên mạng (Render)

Repo có sẵn `render.yaml`. Trên https://render.com: **New → Blueprint**, chọn repo này, bấm **Deploy**.
Máy chủ chạy với biến môi trường `SUAPDF_SERVER=1`, khi đó:

- Phiên làm việc tự xóa sau `SESSION_TTL_MIN` phút không dùng (mặc định 30), giữ tối đa `MAX_SESSIONS` phiên (mặc định 6).
- Giới hạn `MAX_UPLOAD_MB` (25) và `MAX_PAGES` (80); giữ `MAX_UNDO` (8) bước hoàn tác.
- Không lưu dữ liệu chẩn đoán (`debug/`) như khi chạy trên máy.
- Dùng bộ font tự do trong `fonts_free/` (giấy phép trong `fonts_free/LICENSES/`).

## Giấy phép font đi kèm

`fonts_free/` gồm Liberation (SIL OFL 1.1), DejaVu (giấy phép Bitstream Vera/DejaVu), Carlito (SIL OFL 1.1), Caladea (Apache 2.0).
Không đưa font của Microsoft (Times New Roman, Arial, Calibri, Segoe UI…) lên repo vì không được phép phân phối lại.
