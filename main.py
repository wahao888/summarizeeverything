import os
import subprocess
from datetime import timedelta
from dotenv import load_dotenv
from openai import OpenAI
import json

# 讀取 .env 文件中的環境變數
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 初始化 OpenAI 客戶端
client = OpenAI(api_key=api_key)

#-------------------------------------
# 使用者自訂參數
input_dir = "input_media"        # 輸入資料夾名稱
output_dir = "output_result"     # 輸出資料夾名稱
processed_files_log = "processed_files.json"  # 記錄已處理檔案的日誌
segment_length = 10 * 60         # 影片分段長度（10分鐘，單位：秒）
model_name = "gpt-4o"            # 最新模型 GPT-4o
whisper_model = "whisper-1"      # Whisper 語音轉文字模型
#-------------------------------------

# 建立輸出資料夾
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 載入已處理檔案記錄
if os.path.exists(processed_files_log):
    with open(processed_files_log, "r", encoding="utf-8") as f:
        processed_files = json.load(f)
else:
    processed_files = []

#-------------------------------------
# 定義函式：使用FFmpeg分割影音檔
def split_media_file(file_path, segment_length):
    print(f"正在分割檔案：{file_path}")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_duration_str = result.stdout.strip()
    if not total_duration_str:
        print("無法讀取檔案時長，跳過該檔案。")
        return []
    total_duration = float(total_duration_str)

    num_segments = int((total_duration // segment_length) + (1 if total_duration % segment_length != 0 else 0))
    segment_paths = []

    for i in range(num_segments):
        start_time = i * segment_length
        output_segment = f"{os.path.splitext(file_path)[0]}_part{i}{os.path.splitext(file_path)[1]}"
        cmd_segment = [
            "ffmpeg",
            "-i", file_path,
            "-ss", str(timedelta(seconds=start_time)),
            "-t", str(segment_length),
            "-c", "copy",
            output_segment,
            "-y"
        ]
        subprocess.run(cmd_segment, capture_output=True)
        segment_paths.append(output_segment)
    print(f"分割完成，共生成 {len(segment_paths)} 個片段。")
    return segment_paths

#-------------------------------------
# (新增) 定義函式：將 .mov 轉成音訊檔 (mp3)
# (修改處)
def convert_to_audio(input_file, output_audio):
    """
    使用 ffmpeg 將輸入的 mov 檔轉成 mp3 檔。
    如果想進一步壓縮，可調整比特率(b:a)或取樣率(ar)等參數。
    """
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vn",                # 移除影像，只保留音訊
        "-acodec", "libmp3lame",
        "-b:a", "64k",        # 這裡設定 64k 做壓縮
        output_audio,
        "-y"                  # 覆蓋輸出檔案
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return output_audio

#-------------------------------------
# 定義函式：呼叫 Whisper 語音轉文字
def transcribe_audio(file_path, model="whisper-1"):
    print(f"正在轉文字：{file_path}")
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file
        )
    print(f"轉文字完成：{file_path}")
    return response.text

#-------------------------------------
# 定義函式：呼叫 GPT-4o API 生成摘要
def summarize_text(content, model="gpt-4o"):
    print("正在生成摘要...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一個專業的摘要、評論專家，請以繁體中文輸出重點摘要結果，如果有不同人發言，請根據每個人做摘要，最後給我一段點評或心得。"},
            {"role": "user", "content": f"以下是完整文本，請幫我摘要重點及點評或心得：\n{content}"}
        ]
    )
    print("摘要生成完成。")
    return response.choices[0].message.content.strip()

#-------------------------------------
# 處理流程開始
print("掃描輸入資料夾...")
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)

    # 檢查是否已處理過
    if filename in processed_files:
        print(f"已處理過檔案，跳過：{filename}")
        continue

    if os.path.isfile(file_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".mov", ".mp4", ".m4a", ".mp3"]:
            all_texts = []
            segments = split_media_file(file_path, segment_length)
            for seg_file in segments:
                # (修改處) 如果是 mov 檔，則先轉成 mp3 再進行語音轉文字
                seg_ext = os.path.splitext(seg_file)[1].lower()
                if seg_ext == ".mov":
                    # 轉檔成 mp3
                    audio_file = seg_file.replace(".mov", "_audio.mp3")
                    convert_to_audio(seg_file, audio_file)  # 轉成音訊檔
                    text = transcribe_audio(audio_file, model=whisper_model)
                    all_texts.append(text)
                    # 清理暫存檔案
                    os.remove(seg_file)
                    os.remove(audio_file)
                else:
                    # 其他格式直接做語音轉文字
                    text = transcribe_audio(seg_file, model=whisper_model)
                    all_texts.append(text)
                    os.remove(seg_file)

            # 合併所有文字，並加入換行方便閱讀
            full_text = "\n\n".join(all_texts)

            # 輸出完整文字檔
            print(f"正在輸出完整文字檔：{filename}")
            full_text_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_full_text.txt")
            with open(full_text_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            # 使用 GPT-4o 產生摘要
            summary_text = summarize_text(full_text, model=model_name)

            # 每段摘要加換行處理
            formatted_summary = "\n\n".join(summary_text.split("\n"))

            # 輸出摘要文字檔
            print(f"正在輸出摘要文字檔：{filename}")
            summary_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(formatted_summary)

            # 更新已處理檔案記錄
            processed_files.append(filename)
        elif ext == ".txt":
            print(f"正在處理文字檔：{filename}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            processed_files.append(filename)

# 更新處理記錄檔
with open(processed_files_log, "w", encoding="utf-8") as f:
    json.dump(processed_files, f)

print("所有檔案處理完成！")
