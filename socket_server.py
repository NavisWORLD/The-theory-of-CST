import socket
import threading
import json
from cst_engine import CSTEngine
import time
import traceback

def handle_client(client_socket, engine):
    client_addr = "unknown"
    try:
        client_addr = client_socket.getpeername()
        print(f"[+] Connection from {client_addr}")
        buffer = ""
        active_threads = threading.active_count()
        client_socket.settimeout(15.0)
        client_socket.sendall(b'{"status":"connected"}\n')
        print(f"[Server] Sent handshake response to {client_addr}")

        while True:
            try:
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    print(f"[Client {client_addr}] Disconnected")
                    break
                buffer += data
                print(f"[Server] Received raw data from {client_addr}: {data[:100]}...")
                if len(buffer) > 10 * 1024 * 1024:
                    print(f"[Server] Buffer overflow for {client_addr}, clearing old data")
                    buffer = buffer[-1024:]
                while "\n" in buffer:
                    cmd_line, buffer = buffer.split("\n", 1)
                    cmd_line = cmd_line.strip()
                    if not cmd_line:
                        continue
                    parts = cmd_line.split(' ', 1)
                    cmd = parts[0].lower()
                    payload = parts[1] if len(parts) > 1 else ""
                    print(f"[Server] Received command from {client_addr}: '{cmd}', Payload: {payload[:50]}..., Active threads: {active_threads}")
                    if cmd == "update":
                        try:
                            state = engine.update(dt=0.01)
                            client_socket.sendall((state + '\n').encode('utf-8'))
                            print(f"[Server] Sent {len(state)} bytes to {client_addr}")
                        except Exception as e:
                            print(f"[SEND ERROR] Failed to send state to {client_addr}: {e}")
                            print(traceback.format_exc())
                            client_socket.sendall(b'[]\n')
                    elif cmd == "ping":
                        try:
                            response = engine.ping()
                            client_socket.sendall((response + '\n').encode('utf-8'))
                            print(f"[Server] Sent ping response to {client_addr}")
                        except Exception as e:
                            print(f"[SEND ERROR] Failed to send ping response to {client_addr}: {e}")
                            print(traceback.format_exc())
                            client_socket.sendall(b'{"status":"error"}\n')
                    elif cmd == "audio":
                        try:
                            audio_data = json.loads(payload)
                            if not isinstance(audio_data, dict) or 'rms' not in audio_data or 'pitch' not in audio_data:
                                raise ValueError("Invalid audio data format")
                            engine.process_audio(audio_data)
                            client_socket.sendall(b'{"status":"ok"}\n')
                            print(f"[Server] Processed audio data from {client_addr}: RMS={audio_data.get('rms', 0):.3f}, Pitch={audio_data.get('pitch', 0):.1f}Hz")
                        except Exception as e:
                            print(f"[SEND ERROR] Failed to process audio data from {client_addr}: {e}")
                            print(traceback.format_exc())
                            client_socket.sendall(b'{"status":"error"}\n')
                    else:
                        print(f"[?] Unknown command from {client_addr}: '{cmd}'")
                        client_socket.sendall(b'[]\n')
            except socket.timeout:
                print(f"[Client {client_addr}] Receive timeout, continuing...")
                continue
            except Exception as e:
                print(f"[!] Client error from {client_addr}: {e}")
                print(traceback.format_exc())
                break
    except Exception as e:
        print(f"[!] Unhandled error in handle_client for {client_addr}: {e}")
        print(traceback.format_exc())
    finally:
        client_socket.close()
        print(f"[-] Disconnected from {client_addr}")

def start_server(host='127.0.0.1', port=5555):
    print(f"[Server] Starting server on {host}:{port}...")
    try:
        engine = CSTEngine()
        print("[Server] CSTEngine initialized successfully")
    except Exception as e:
        print(f"[Server] Failed to initialize CSTEngine: {e}")
        print(traceback.format_exc())
        return

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((host, port))
        print(f"[Server] Successfully bound to {host}:{port}")
        server.listen(10)
        print(f"[Server] Listening on {host}:{port}")
        while True:
            print(f"[Server] Heartbeat: Server alive at {time.strftime('%Y-%m-%d %H:%M:%S')}, Active clients: {threading.active_count() - 1}")
            try:
                client_socket, addr = server.accept()
                print(f"[Server] Accepted connection from {addr}, Socket state: {client_socket}")
                client_thread = threading.Thread(target=handle_client, args=(client_socket, engine), daemon=True)
                client_thread.start()
            except Exception as e:
                print(f"[Server Accept Error] Failed to accept client: {e}")
                print(traceback.format_exc())
                # Continue running the server despite the error
                time.sleep(1)
    except KeyboardInterrupt:
        print("[Server] Shutting down")
    except Exception as e:
        print(f"[Server Error] {e}")
        print(traceback.format_exc())
    finally:
        server.close()
        engine.cleanup()
        print("[Server] Closed")

if __name__ == "__main__":
    start_server()