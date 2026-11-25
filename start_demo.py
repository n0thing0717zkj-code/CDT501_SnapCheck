"""启动网页演示：先起服务，确认能访问，再打开浏览器。

用法：python start_demo.py [--no-browser]
"""
import argparse
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8501


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_ready(url, timeout=45):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    port = args.port
    if not port_free(port):
        print(f"端口 {port} 被占用，改用 {port + 1}")
        port += 1
        if not port_free(port):
            print("8501 和 8502 都被占用：关掉旧的黑色窗口再试。")
            sys.exit(1)

    url = f"http://localhost:{port}"
    print(f"正在启动服务（端口 {port}）…")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py",
           "--server.port", str(port),
           "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, cwd=str(ROOT))

    if wait_ready(url):
        print(f"服务已就绪：{url}")
        if not args.no_browser:
            webbrowser.open(url)
        print("演示结束后关闭这个窗口即可。")
    else:
        print("服务没能在 45 秒内启动。上面的报错就是原因。")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
