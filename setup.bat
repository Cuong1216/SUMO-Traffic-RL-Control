@echo off
echo ====================================================
echo BAT DAU CAI DAT MOI TRUONG DU AN AI
echo ====================================================

REM Kiểm tra xem thư mục venv đã tồn tại chưa
if not exist "venv\" (
    echo [1/4] Thư mục venv chưa tồn tại. Đang tạo môi trường ảo mới...
    python -m venv venv
    if errorlevel 1 (
        echo LỖI: Không thể tạo môi trường ảo. Hãy chắc chắn rằng bạn đã cài đặt Python.
        pause
        exit /b 1
    )
    echo Đã tạo môi trường ảo venv thành công.
) else (
    echo [1/4] Thư mục venv đã tồn tại. Bỏ qua bước tạo mới.
)

REM Kích hoạt môi trường ảo
echo [2/4] Đang kích hoạt môi trường ảo...
call venv\Scripts\activate.bat

REM Nâng cấp pip lên bản mới nhất
echo [3/4] Đang nâng cấp pip lên phiên bản mới nhất...
python -m pip install --upgrade pip

REM Cài đặt các thư viện từ requirements.txt
echo [4/4] Đang cài đặt các thư viện từ requirements.txt (bao gồm PyTorch CPU)...
pip install -r requirements.txt

echo ====================================================
echo CAI DAT HOAN TAT!
echo ====================================================
echo Để bắt đầu làm việc hoặc chạy code, hãy kích hoạt môi trường ảo bằng lệnh:
echo     venv\Scripts\activate
echo.
pause
