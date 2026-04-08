import os
import socket
import ssl
import traceback
import requests

HOST = "api.openai.com"
URL = "https://api.openai.com/v1/models"
API_KEY = os.getenv("OPENAI_API_KEY")

def test_dns():
    print("\n[1] DNS 解析测试")
    try:
        infos = socket.getaddrinfo(HOST, 443, proto=socket.IPPROTO_TCP)
        addrs = sorted(set(item[4][0] for item in infos))
        print("DNS OK:", addrs)
    except Exception as e:
        print("DNS FAILED:", repr(e))

def test_tcp():
    print("\n[2] TCP 连接测试")
    try:
        with socket.create_connection((HOST, 443), timeout=10) as s:
            print("TCP OK:", s.getpeername())
    except Exception as e:
        print("TCP FAILED:", repr(e))

def test_tls():
    print("\n[3] TLS/SSL 握手测试")
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((HOST, 443), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=HOST) as ssock:
                cert = ssock.getpeercert()
                print("TLS OK")
                print("Protocol:", ssock.version())
                print("Subject:", cert.get("subject"))
    except Exception as e:
        print("TLS FAILED:", repr(e))

def test_http_no_auth():
    print("\n[4] HTTP 测试（不带 API key）")
    try:
        r = requests.get(URL, timeout=20)
        print("Status:", r.status_code)
        print("Body:", r.text[:500])
    except Exception as e:
        print("HTTP FAILED:", repr(e))
        traceback.print_exc()

def test_http_with_auth():
    print("\n[5] HTTP 测试（带 API key）")
    if not API_KEY:
        print("SKIP: 环境变量 OPENAI_API_KEY 未设置")
        return

    try:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        r = requests.get(URL, headers=headers, timeout=20)
        print("Status:", r.status_code)
        print("Body:", r.text[:800])
    except Exception as e:
        print("AUTH HTTP FAILED:", repr(e))
        traceback.print_exc()

if __name__ == "__main__":
    print("HTTP_PROXY =", os.getenv("HTTP_PROXY"))
    print("HTTPS_PROXY =", os.getenv("HTTPS_PROXY"))
    print("ALL_PROXY =", os.getenv("ALL_PROXY"))
    print("REQUESTS_CA_BUNDLE =", os.getenv("REQUESTS_CA_BUNDLE"))
    print("SSL_CERT_FILE =", os.getenv("SSL_CERT_FILE"))

    test_dns()
    test_tcp()
    test_tls()
    test_http_no_auth()
    test_http_with_auth()