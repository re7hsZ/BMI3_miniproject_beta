from Bio import SeqIO
from collections import defaultdict, namedtuple
import re
import random
import os

CDS_FILE = "./data/raw/host_cds_from_genomic.fna"  # 输入：宿主所有CDS（FASTA格式）
OUTPUT_CORE_CDS = "./data/processed/host/host_core_cds_sequences.fasta"  # 输出：最终筛选出的CDS结果
OUTPUT_CORE_IDS = "./data/processed/host/host_core_ids_clean.txt"  # 输出：筛选出CDS的ID列表
TEMP_MOBILE_SITES = "./data/processed/host/mobile_element_sites.txt"  # 临时文件：移动元件位点
OUTPUT_TRAIN = "./data/processed/host/host_core_train.fasta"
OUTPUT_TEST = "./data/processed/host/host_core_test_empty.fasta"

MIN_CDS_LENGTH = 150  # 最小CDS长度（bp）
N_EVAL_THRESHOLD = 0.05  # N碱基占比阈值（>5%则剔除）
NEIGHBORHOOD_RANGE = 5000  # 移动元件邻域排除范围（±5000 bp）

# 用于筛选的关键词（移动元件相关，匹配即剔除）
mobile_keywords = [
    # ① 转座子/逆转录转座子相关
        "transposase", "insertion sequence", "IS element", "insertion element",
        "mobile element", "integrase", "recombinase", "resolvase",
        "excisionase", "invertase", "tyrosine recombinase", "serine recombinase",
        # ② 噬菌体/原噬菌体相关
        "phage", "prophage", "phage integrase", "phage tail", "tail protein",
        "capsid", "portal protein", "terminase", "lysin", "holin", "lysis",
        "head protein", "baseplate",
        # ③ 质粒相关/共轭转移系统
        "plasmid", "conjugation", "conjugal", "tra gene", "trb gene",
        "mobA", "mobB", "mobC", "relaxase", "type IV secretion", "T4SS", "Pilus",
        # ④ 抗性岛/基因岛/可移动岛屿结构
        "genomic island", "island", "integrative", "ICE", "IME", "cargo gene",
        "att site", "attachment site", "acquired", "horizontal transfer",
        "HGT", "mobile region", "MGI",
        # ⑤ CRISPR/Cas相关
        "CRISPR", "Cas"
]

VALID_NUCLEOTIDES = {"A", "T", "G", "C", "a", "t", "g", "c"}

# 拆分相关参数（3:1训练集+空白测试集）
TRAIN_TEST_RATIO = 3  # 训练集:测试集 = 3:1
RANDOM_SEED = 42  # 固定种子，确保拆分结果可复现

# -------------------------------------------------------------------

# 存储CDS的位置信息
CDS_Location = namedtuple("CDS_Location", ["id", "start", "end", "record"])

def extract_location_from_header(header):
    """
    从FASTA header中提取CDS的位置信息（[location=xxx]字段）
    输入：header字符串（如 "[location=654..2012]"）
    输出：(start, end)整数元组，提取失败返回(None, None)
    """
    location_pattern = r"\[location=([^\]]+)\]"
    match = re.search(location_pattern, header)
    if not match:
        return None, None
    
    location_str = match.group(1).lower()

    if "complement" in location_str or "join" in location_str:
        num_match = re.search(r"(\d+)\.\.(\d+)", location_str)
    else:
        num_match = re.search(r"(\d+)\.\.(\d+)", location_str)
    
    if not num_match:
        return None, None
    
    start = int(num_match.group(1))
    end = int(num_match.group(2))
    return (start, end) if start <= end else (end, start)

def split_train_test(core_records):
    """
    仅按3:1比例拆分训练集和测试集（不做任何额外操作，均保留完整CDS）
    core_records：筛选后的核心CDS序列列表（SeqRecord对象）
    """
    total_count = len(core_records)
    print(f"Total number of core CDS: {total_count}")

    random.seed(RANDOM_SEED)
    random.shuffle(core_records)
    
    train_count = int(total_count * TRAIN_TEST_RATIO / (TRAIN_TEST_RATIO + 1))
    test_count = total_count - train_count
    
    train_records = core_records[:train_count]
    test_records = core_records[train_count:]

    SeqIO.write(train_records, OUTPUT_TRAIN, "fasta")
    SeqIO.write(test_records, OUTPUT_TEST, "fasta")

    print(f"Train(3/4): {len(train_records)} CDS → {OUTPUT_TRAIN}")
    print(f"Test(1/4): {len(test_records)} CDS→ {OUTPUT_TEST}")

def main():
    try:
        # 1. 读取宿主CDS文件并提取位置信息
        all_cds_locations = []
        for record in SeqIO.parse(CDS_FILE, "fasta"):
            header = record.description
            start, end = extract_location_from_header(header)
            all_cds_locations.append(CDS_Location(
                id=record.id,
                start=start,
                end=end,
                record=record
            ))
        print(f"{len(all_cds_locations)} CDS total read from {CDS_FILE}")

        # 2. 剔除短于150bp的CDS 
        filtered_by_length = []
        for cds in all_cds_locations:
            length = len(cds.record.seq)
            if length >= MIN_CDS_LENGTH:
                filtered_by_length.append(cds)
        print(f"{len(filtered_by_length)} CDS remain after length filtering (≥{MIN_CDS_LENGTH} bp)")

        # 3. 剔除含非法字符或N碱基占比过高的CDS
        filtered_by_seq_validity = []
        for cds in filtered_by_length:
            seq = cds.record.seq
            length = len(seq)
            
            invalid_chars = [char for char in seq if char not in VALID_NUCLEOTIDES]
            if invalid_chars:
                continue
            
            n_count = seq.upper().count("N")
            n_ratio = n_count / length if length > 0 else 0.0
            if n_ratio > N_EVAL_THRESHOLD:
                continue
            
            filtered_by_seq_validity.append(cds)
        print(f"{len(filtered_by_seq_validity)} CDS remain after sequence validity filtering with N ratio ≤ {N_EVAL_THRESHOLD*100}% ")

        # 4. 剔除移动元件相关CDS（关键词筛选）
        filtered_by_mobile = []
        mobile_cds_ids = set()
        mobile_cds_locations = []
        
        for cds in filtered_by_seq_validity:
            description = cds.record.description.lower()

            if any(keyword.lower() in description for keyword in mobile_keywords):
                mobile_cds_ids.add(cds.id)
                if cds.start and cds.end:
                    mobile_cds_locations.append(cds)
            else:
                filtered_by_mobile.append(cds)
        print(f"{len(filtered_by_mobile)} CDS remain after mobile element filtering, removed {len(mobile_cds_ids)} mobile-related CDS")

        with open(TEMP_MOBILE_SITES, "w", encoding="utf-8") as f:
            f.write("CDS_ID\tStart\tEnd\n")
            for cds in mobile_cds_locations:
                f.write(f"{cds.id}\t{cds.start}\t{cds.end}\n")

        # 5. 邻域排除：剔除位于移动元件±5000bp范围内的CDS
        if not mobile_cds_locations:
            print("No mobile element CDS found; skipping neighborhood exclusion step.")
            filtered_by_neighborhood = filtered_by_mobile
        else:
            filtered_by_neighborhood = []
            for target_cds in filtered_by_mobile:
                if not target_cds.start or not target_cds.end:
                    filtered_by_neighborhood.append(target_cds)
                    continue

                is_excluded = False
                target_mid = (target_cds.start + target_cds.end) / 2 # location midpoint
                
                for mobile_cds in mobile_cds_locations:
                    mobile_start = mobile_cds.start - NEIGHBORHOOD_RANGE
                    mobile_end = mobile_cds.end + NEIGHBORHOOD_RANGE

                    if mobile_start <= target_mid <= mobile_end:
                        is_excluded = True
                        break
                
                if not is_excluded:
                    filtered_by_neighborhood.append(target_cds)
        
        print(f"{len(filtered_by_neighborhood)} CDS remain after neighborhood exclusion around mobile elements")

        # 6. 单拷贝基因筛选：仅保留单拷贝基因，或多拷贝中最长的一个
        gene_groups = defaultdict(list)
        for cds in filtered_by_neighborhood:
            header = cds.record.description
            gene_pattern = r"\[gene=([^\]]+)\]"
            gene_match = re.search(gene_pattern, header)
            if gene_match:
                gene_name = gene_match.group(1).strip()
            else:
                gene_name = f"NO_GENE_{cds.id}"
            gene_groups[gene_name].append(cds)
        
        single_copy_cds = []
        for gene_name, cds_list in gene_groups.items():
            if len(cds_list) == 1:
                single_copy_cds.append(cds_list[0])
            else:
                longest_cds = max(cds_list, key=lambda x: len(x.record.seq))
                single_copy_cds.append(longest_cds)
        
        final_core_records = [cds.record for cds in single_copy_cds]
        final_core_ids = [cds.id for cds in single_copy_cds]
        print(f"{len(final_core_records)} CDS remain after single-copy gene filtering")

        # 7. 保存最终筛选结果
        SeqIO.write(final_core_records, OUTPUT_CORE_CDS, "fasta")
        with open(OUTPUT_CORE_IDS, "w", encoding="utf-8") as f:
            f.write("\n".join(final_core_ids))

        # 8. 拆分训练集和测试集
        split_train_test(final_core_records)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()