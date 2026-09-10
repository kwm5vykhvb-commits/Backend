# app.py redirige vers main.py
from main import app, start_node_server, stop_node_server

if __name__ == "__main__":
    import os
    import subprocess
    if not os.path.exists("dist/server.cjs"):
        subprocess.run(["npm", "run", "build"], check=False)
    os.execvp("node", ["node", "dist/server.cjs"])
