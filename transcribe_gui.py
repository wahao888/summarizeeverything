import asyncio
import base64
import json
import websockets
import pyaudio
import requests
import numpy as np
import tkinter as tk
import datetime
import threading
from dotenv import load_dotenv
import os
import ssl
import certifi
from opencc import OpenCC  # 簡體→繁體轉換

# 讀取 .env 檔案中的環境變數
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_TRANSCRIBE")
TEXT_LOG_FILE = "transcript_log.txt"

# 初始化 opencc 轉換器
cc = OpenCC("s2t")

def get_client_secret():
    url = "https://api.openai.com/v1/realtime/transcription_sessions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }
    payload = {
        "input_audio_format": "pcm16",
        "input_audio_transcription": {
            "model": "gpt-4o-transcribe",
            "prompt": ""
        },
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,  # 調整門檻值
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500  # 縮短靜默時間
        },
        "input_audio_noise_reduction": {
            "type": "near_field"
        }
    }
    resp = requests.post(url, headers=headers, json=payload, verify=certifi.where())
    if resp.status_code != 200:
        raise Exception(f"API 請求失敗: {resp.status_code} - {resp.text}")
    return resp.json()["client_secret"]["value"]

class SubtitleWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎤 語音轉文字字幕")
        self.root.attributes('-topmost', True)
        self.root.geometry("600x150+50+50")  # 調高視窗高度
        self.root.configure(bg="black")

        self.label = tk.Label(
            self.root,
            text="",
            font=("Helvetica", 16),
            fg="lime",
            bg="black",
            justify="left",   # 🆕 文字靠左
            anchor="nw",      # 🆕 由左上開始顯示
            wraplength=580
        )
        self.label.pack(expand=True, fill="both")

        self.full_text = ""  # 🆕 用來累積所有文字

    def update_text(self, text):
        # 🆕 累積並用 after 排到主緒更新 GUI
        self.full_text += text
        # 若要換行，可在 text 前後自行加 "\n"
        self.root.after(0, lambda: self.label.config(text=self.full_text))

    def run(self):
        self.root.mainloop()

def save_text_log(text):
    try:
        with open(TEXT_LOG_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            f.write(f"{timestamp} {text}\n")
    except Exception as e:
        print(f"儲存文字記錄時發生錯誤: {e}")

async def transcribe_realtime(update_gui_callback):
    client_secret = get_client_secret()

    uri = "wss://api.openai.com/v1/realtime"
    headers = [
        ("Authorization", f"Bearer {client_secret}"),
        ("OpenAI-Beta", "realtime=v1")
    ]
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    print("🔄 正在建立 WebSocket 連線...")
    async with websockets.connect(uri, extra_headers=headers, ssl=ssl_ctx) as ws:
        print("✅ WebSocket 連線成功")

        await ws.send(json.dumps({
            "type": "transcription_session.update",
            "session": {
                "input_audio_transcription": {
                    "model": "gpt-4o-transcribe",
                    "prompt": ""
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,  # 調整門檻值
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500  # 縮短靜默時間
                },
                "input_audio_noise_reduction": {
                    "type": "near_field"
                },
                "include": []
            }
        }))
        print("✅ 初始化設定發送成功")

        RATE, CHUNK = 16000, 1024
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("🎙️ 錄音中... Ctrl+C 停止")

        async def read_audio():
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                b64 = base64.b64encode(data).decode("utf-8")
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
                await asyncio.sleep(CHUNK / RATE)

        async def receive_text():
            async for msg in ws:
                resp = json.loads(msg)
                t = resp.get("type")
                if t == "conversation.item.input_audio_transcription.delta":
                    delta = resp.get("delta", "")
                    if delta:
                        trad = cc.convert(delta)
                        print(f"🔤 部分轉錄（繁體）: {trad}")
                        update_gui_callback(trad)
                        save_text_log(trad)
                elif t == "conversation.item.input_audio_transcription.completed":
                    txt = resp.get("transcript", "")
                    if txt:
                        trad_full = cc.convert(txt)
                        print(f"🏁 最終轉錄（繁體）: {trad_full}")
                        update_gui_callback(trad_full)
                        save_text_log(trad_full)
                elif t == "error":
                    print(f"❌ 錯誤: {resp.get('error')}")

        await asyncio.gather(read_audio(), receive_text())

def main():
    win = SubtitleWindow()
    threading.Thread(target=lambda: asyncio.run(transcribe_realtime(win.update_text)), daemon=True).start()
    win.run()

if __name__ == "__main__":
    main()
