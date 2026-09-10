import asyncio
import atexit
import os
import signal
import subprocess
import sys
import time
import urllib.request

INTERNAL_NODE_PORT = int(os.environ.get("INTERNAL_NODE_PORT", "3001"))
node_process = None

def ensure_build():
    """S'assure que les dépendances et le serveur TypeScript sont compilés."""
    if not os.path.exists("node_modules"):
        print("[NLSbox-Bridge] Installation des dépendances npm...")
        subprocess.run(["npm", "install"], check=False)

    if not os.path.exists("dist/server.cjs"):
        print("[NLSbox-Bridge] Compilation de server.ts vers dist/server.cjs...")
        subprocess.run(["npm", "run", "build"], check=False)

def start_node_server():
    global node_process
    if node_process and node_process.poll() is None:
        return

    ensure_build()

    env = os.environ.copy()
    env["PORT"] = str(INTERNAL_NODE_PORT)
    env["NODE_ENV"] = env.get("NODE_ENV", "production")

    print(f"[NLSbox-Bridge] Lancement du moteur Node.js sur le port interne {INTERNAL_NODE_PORT}...")
    node_process = subprocess.Popen(
        ["node", "dist/server.cjs"],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Attendre que le serveur Node soit prêt
    start_time = time.time()
    while time.time() - start_time < 20:
        if node_process.poll() is not None:
            print("[NLSbox-Bridge] Erreur : le processus Node.js s'est arrêté inopinément.")
            break
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{INTERNAL_NODE_PORT}/healthz")
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    print(f"[NLSbox-Bridge] Node.js opérationnel sur le port {INTERNAL_NODE_PORT} !")
                    return
        except Exception:
            time.sleep(0.3)

def stop_node_server():
    global node_process
    if node_process and node_process.poll() is None:
        print("[NLSbox-Bridge] Arrêt du processus Node.js...")
        node_process.terminate()
        try:
            node_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            node_process.kill()
        node_process = None

atexit.register(stop_node_server)

async def proxy_with_httpx(scope, receive, send, httpx_mod):
    method = scope["method"]
    path = scope.get("path", "/")
    query = scope.get("query_string", b"").decode("latin1")
    url = f"http://127.0.0.1:{INTERNAL_NODE_PORT}{path}"
    if query:
        url += f"?{query}"

    headers = []
    for k, v in scope.get("headers", []):
        name = k.decode("latin1").lower()
        if name not in ("host", "transfer-encoding"):
            headers.append((name, v.decode("latin1")))

    # Lire le corps de la requête
    body_chunks = []
    while True:
        msg = await receive()
        if msg["type"] == "http.request":
            body_chunks.append(msg.get("body", b""))
            if not msg.get("more_body", False):
                break
    req_body = b"".join(body_chunks)

    limits = httpx_mod.Limits(max_keepalive_connections=50, max_connections=200)
    async with httpx_mod.AsyncClient(limits=limits, timeout=None) as client:
        async with client.stream(method, url, headers=headers, content=req_body) as resp:
            resp_headers = []
            for name, value in resp.headers.raw:
                resp_headers.append([name, value])

            await send({
                "type": "http.response.start",
                "status": resp.status_code,
                "headers": resp_headers,
            })

            async for chunk in resp.aiter_raw():
                await send({
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": True,
                })

            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })

async def proxy_with_raw_socket(scope, receive, send):
    method = scope["method"]
    path = scope.get("path", "/")
    query = scope.get("query_string", b"").decode("latin1")
    full_path = path + (f"?{query}" if query else "")

    # Lire le corps de la requête
    body_chunks = []
    while True:
        msg = await receive()
        if msg["type"] == "http.request":
            body_chunks.append(msg.get("body", b""))
            if not msg.get("more_body", False):
                break
    req_body = b"".join(body_chunks)

    reader, writer = await asyncio.open_connection("127.0.0.1", INTERNAL_NODE_PORT)
    try:
        # En-tête HTTP
        header_lines = [f"{method} {full_path} HTTP/1.1", f"Host: 127.0.0.1:{INTERNAL_NODE_PORT}"]
        for k, v in scope.get("headers", []):
            name = k.decode("latin1")
            if name.lower() not in ("host", "connection"):
                header_lines.append(f"{name}: {v.decode('latin1')}")
        header_lines.append("Connection: close")
        if req_body:
            header_lines.append(f"Content-Length: {len(req_body)}")

        req_bytes = "\r\n".join(header_lines).encode("latin1") + b"\r\n\r\n" + req_body
        writer.write(req_bytes)
        await writer.drain()

        # Lecture de la réponse HTTP
        header_data = b""
        while b"\r\n\r\n" not in header_data:
            chunk = await reader.read(4096)
            if not chunk:
                break
            header_data += chunk

        if not header_data:
            await send({"type": "http.response.start", "status": 502, "headers": []})
            await send({"type": "http.response.body", "body": b"Bad Gateway", "more_body": False})
            return

        headers_part, first_body_part = header_data.split(b"\r\n\r\n", 1)
        lines = headers_part.split(b"\r\n")
        status_line = lines[0].decode("latin1")
        status_code = int(status_line.split(" ")[1])

        resp_headers = []
        for line in lines[1:]:
            if b":" in line:
                name, val = line.split(b":", 1)
                resp_headers.append([name.strip().lower(), val.strip()])

        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": resp_headers,
        })

        if first_body_part:
            await send({
                "type": "http.response.body",
                "body": first_body_part,
                "more_body": True,
            })

        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            await send({
                "type": "http.response.body",
                "body": chunk,
                "more_body": True,
            })

        await send({
            "type": "http.response.body",
            "body": b"",
            "more_body": False,
        })
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

# L'application ASGI cible appelée par "uvicorn main:app"
async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                start_node_server()
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                stop_node_server()
                await send({"type": "lifespan.shutdown.complete"})
                return

    elif scope["type"] == "http":
        # Assurer que Node est démarré même si lifespan a été ignoré
        if not node_process or node_process.poll() is not None:
            start_node_server()

        try:
            import httpx
            await proxy_with_httpx(scope, receive, send, httpx)
        except ImportError:
            await proxy_with_raw_socket(scope, receive, send)

    else:
        # Types non supportés
        pass

if __name__ == "__main__":
    # Si exécuté directement : python main.py
    ensure_build()
    print("[NLSbox] Exécution directe de Node.js...")
    os.execvp("node", ["node", "dist/server.cjs"])
