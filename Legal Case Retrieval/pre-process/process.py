import os
import re
import sys
import argparse
from langdetect import detect
from langdetect import detect_langs
from langdetect import DetectorFactory
from tqdm import tqdm
import multiprocessing
from functools import partial
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_env_int, get_task1_dir, get_task1_year, resolve_repo_path

DetectorFactory.seed = 0

def is_sentence(s):
    return s == "" or s.strip().endswith(('.', ':', ';'))

def remove(match):
    result = match.group()
    return result.replace("[", "").replace("]", "").replace(" ", "")

def remove2(match):
    result = match.group()
    return result.replace("[", "").replace("]", "")

def rep(match):
    result = match.group()
    return result.replace("[", "{").replace("]", "}")

def rep2(match):
    result = match.group()
    return result.replace("{}", "[").replace("}", "]")

def process_file(name, input_dir, summary_dir, output_dir, have_sum):
    last_lang = "en"
    
    with open(f"{input_dir}/{name}", "r", encoding="utf-8") as f:
        t = f.read()
        idx_ = t.find("[1]")
        if idx_ != -1:
            t = t[idx_:]
        lines = t.splitlines()
        lines = [line.strip() for line in lines]
        sentence_list = []
        flag = True
        
        for l in lines:
            if flag and (
                "<FRAGMENT_SUPPRESSED>" in l
                or " FRAGMENT_SUPPRESSED" in l
                or l == ""
            ):
                continue
            flag = False
            l1 = l.replace("<FRAGMENT_SUPPRESSED>", "").replace("FRAGMENT_SUPPRESSED", "").strip()
            l2 = re.sub(r'\[\d{1,3}\]', "", l1).strip()
            if (
                (len(l2) == 1 or
                    (
                        l2 != ""
                        and l2[0] != "("
                        and len(l2) > 1
                        and l2[1] != ")"
                        and not l2[0].isdigit()
                    ))
                and sentence_list
                and not is_sentence(sentence_list[-1])
            ):
                sentence_list[-1] += f" {l2}"
            else:
                sentence_list.append(l2)
    txt = "\n".join(sentence_list)

    txt = re.sub(r"\. *(\. *)+", "", txt)
    txt = re.sub(r"[A-Z]*_SUPPRESSED", "", txt)
    
    need_to_removed = ["[translation]", "[Translation]", "[sic]", "[ sic ]", "[Emphasis added.]",
                       "[emphasis added]", 
                       "[End of document]", "*", "[  ]", "[]", "[ ]",
                        "[DATE_SUPPRESSED]", "[TRANSLATION]", 
                       "[English language version follows French language version]", 
                       "[La version anglaise vient à la suite de la version française]", 
                       "[Diagram omitted - see printed version]", 
                       "[French language version follows English language version]",
                       "[La version française vient à la suite de la version anglaise]", 
                       "[Traduction]"]
    for token in need_to_removed:
        txt = txt.replace(token, "")


    txt = re.sub(r"\[[A-Z][A-Z]+\]", rep, txt)
    txt = re.sub(r"[^a-zA-Z]\[[b-zB-Z]\] ", remove, txt)
    txt = re.sub(r"\[[a-zA-Z][a-zA-Z \.']*\]", remove2, txt)
    txt = re.sub(r"\{[A-Z][A-Z]+\}", rep2, txt)
    txt = re.sub(r"\n\n+", "\n\n", txt)
    txt = re.sub(r"\.\.+", ".", txt)
    txt = re.sub(r"\n\.\n", "\n\n", txt)
    
    new_lines = txt.split("\n")
    for i in range(len(new_lines)):
        if len(new_lines[i]) > 0:
            try:
                lang = detect(new_lines[i])
            except:
                if last_lang == "fr":
                    new_lines[i] = ""
                   
            if lang == "fr":
                last_lang = "fr"
                new_lines[i] = ""
            elif lang != "en":
                if last_lang == "fr":
                    new_lines[i] = ""
            else:
                last_lang = "en"
    
    txt = "\n".join(new_lines)     
    txt = re.sub(r"\n\n+", "\n\n", txt)
    
    if "Summary:" not in txt and name in have_sum:
        with open(f"{summary_dir}/{name}", "r", encoding="utf-8") as f:
            sum_ = f.read()
            txt = f"Summary:\n{sum_}\n{txt}"
    with open(f"{output_dir}/{name}", "w+", encoding="utf-8") as f:
        f.write(txt)

def parse_args() -> argparse.Namespace:
    task1_dir = Path(get_task1_dir())
    task1_year = get_task1_year()
    input_dir = resolve_repo_path(os.getenv("TASK1_TRAIN_RAW_DIR")) or (
        task1_dir / f"task1_train_files_{task1_year}"
    )
    summary_dir = resolve_repo_path(os.getenv("TASK1_SUMMARY_DIR")) or (task1_dir / "summary")
    output_dir = resolve_repo_path(os.getenv("TASK1_PROCESSED_DIR")) or (task1_dir / "processed")
    parser = argparse.ArgumentParser(description="Clean Task 1 raw cases into processed corpus.")
    parser.add_argument("--input-dir", type=Path, default=input_dir)
    parser.add_argument("--summary-dir", type=Path, default=summary_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=get_env_int("TASK1_PROCESS_NUM_WORKERS", 0),
        help="0 => use multiprocessing.cpu_count().",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir.resolve()
    summary_dir = args.summary_dir.resolve()
    output_dir = args.output_dir.resolve()

    os.makedirs(output_dir, exist_ok=True)

    names = os.listdir(input_dir)
    have_sum = os.listdir(summary_dir)

    num_cores = args.num_workers if args.num_workers > 0 else multiprocessing.cpu_count()
    print(f"使用 {num_cores} 个CPU核心进行并行处理")

    process_func = partial(
        process_file,
        input_dir=input_dir,
        summary_dir=summary_dir,
        output_dir=output_dir,
        have_sum=have_sum,
    )

    with multiprocessing.Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap(process_func, names), total=len(names), desc="处理文件"))
