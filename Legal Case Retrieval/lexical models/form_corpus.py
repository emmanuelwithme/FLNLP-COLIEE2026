import json
import os
import re
import sys
from pathlib import Path
from tqdm import tqdm
import stat

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir

TASK1_DIR = get_task1_dir()

# 定義路徑
raw_path = Path(f"{TASK1_DIR}/processed")
file_dir = sorted(
    [p for p in raw_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
)

# 定義目標資料夾
output_dir = f"{TASK1_DIR}/lht_process/BM25/corpus"

# 確保目標資料夾存在，如果不存在則自動創建
os.makedirs(output_dir, exist_ok=True)
# 設置資料夾的權限，確保所有用戶都有讀寫權限
os.chmod(output_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
print(f"✅ 目標資料夾 {output_dir} 已建立並設定適當權限")


outfile = open(f'{output_dir}/corpus.json','w', encoding='utf-8')

# for a_file in file_dir:
#     pid = a_file.split('.')[0]
#     path = f'{raw_path}/{a_file}'
#     text_ = ''
    
#     with open(path, encoding='utf-8') as fin:
#         lines = fin.readlines()
#         for line in lines:
#             line = line.replace("\n", "")
#             text_ = text_ + line
  
#     save_dict = {}
#     save_dict['id'] = pid
#     save_dict['contents'] = text_
#     # print(save_dict)
#     outline = json.dumps(save_dict,ensure_ascii=False)+'\n'
#     outfile.write(outline)
#     # break
save_dict = {}
written_count = 0
skipped_count = 0

for file_path in tqdm(file_dir):
    pid = file_path.stem
    text_ = ''

    try:
        with open(file_path, encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.replace("\n", "")
                text_ = text_ + line
    except UnicodeDecodeError as exc:
        skipped_count += 1
        print(f"⚠️ 跳過無法以 UTF-8 讀取的檔案: {file_path.name} ({exc})")
        continue

    save_dict = {}
    save_dict['id'] = pid
    save_dict['contents'] = text_
    # print(save_dict)
    outline = json.dumps(save_dict, ensure_ascii=False) + '\n'
    outfile.write(outline)
    written_count += 1

    # save_dict[str(pid)] = text_
    # save_dict['contents'] = text_
    # print(save_dict)
    # outline = json.dumps(save_dict,ensure_ascii=False)+'\n'
    # outfile.write(outline)
    # # break
# with open(f"{TASK1_DIR}/corpus_all.json", "w", encoding="utf-8") as fp:
#     json.dump(save_dict,fp,ensure_ascii=False)
print(f"✅ JSON 檔案已成功寫入至 {outfile.name}")
print(f"寫入文件數: {written_count}，跳過文件數: {skipped_count}")
