import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

IMAGE = open(sys.argv[1] if len(sys.argv) > 1 else "test.jpg", "rb").read()
state = {"pending": False, "frame_id": 0}


def make_photos():
    while True:
        time.sleep(8)
        state["frame_id"] = (state["frame_id"] + 1) & 0xFF
        state["pending"] = True
        print(f"[mock] new photo ready (frame {state['frame_id']})")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence per-request logging

    def do_GET(self):
        if self.path != "/frame":
            self.send_error(404)
            return
        if not state["pending"]:
            self.send_response(204)
            self.end_headers()
            return
        state["pending"] = False
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("X-Frame-Id", str(state["frame_id"]))
        self.send_header("Content-Length", str(len(IMAGE)))
        self.end_headers()
        self.wfile.write(IMAGE)

    def do_POST(self):
        if self.path != "/result":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        msg_type, frame_id, w, h, count = struct.unpack_from("<BBHHB", body, 0)
        boxes = [struct.unpack_from("<HHHH", body, 7 + i * 8) for i in range(count)]
        print(f"[mock] result: type={msg_type} frame={frame_id} {w}x{h} boxes={boxes}")
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


threading.Thread(target=make_photos, daemon=True).start()
print("[mock] listening on port 8080")
ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
