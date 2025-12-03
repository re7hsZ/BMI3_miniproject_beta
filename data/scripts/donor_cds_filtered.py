from Bio import SeqIO
import random
import csv

CDS_FILE = "./data/raw/near_cds_from_genomic.fna"  # 输入：外源CDS文件（FASTA格式）
HOST_TEST_FILE = "./data/processed/host/host_core_test_empty.fasta"  # 宿主空白测试集
OUTPUT_FILTERED = "./data/processed/donor/foreign_filtered_cds.fasta"  # 输出：预处理后完整外源CDS结果
OUTPUT_TRAIN = "./data/processed/donor/foreign_train.fasta"  # 输出：外源训练集
OUTPUT_TEST = "./data/processed/donor/foreign_test.fasta"    # 输出：外源测试集
OUTPUT_IDS = "./data/processed/donor/foreign_filtered_ids.txt"  # 输出：预处理后外源CDS的ID列表
OUTPUT_HOST_HGT = "./data/processed/host_test_with_hgt.fasta"  # 插入HGT后的宿主测试集
OUTPUT_METADATA = "./data/processed/hgt_insert_metadata.csv"  # 插入信息metadata表格
OUTPUT_TRUTH = "./data/processed/hgt_truth.tsv"  # 标签TSV：GeneID / TrueState(0=host,2=foreign)

MIN_CDS_LENGTH = 150  # 最小CDS长度（bp）
N_EVAL_THRESHOLD = 0.05  # N碱基占比阈值（>5%则剔除）
VALID_NUCLEOTIDES = {"A", "T", "G", "C", "a", "t", "g", "c"}

mobile_keywords = [
    "transposase", "insertion sequence", "IS element", "insertion element",
    "mobile element", "integrase", "recombinase", "resolvase",
    "excisionase", "invertase", "tyrosine recombinase", "serine recombinase",
    "phage", "prophage", "phage integrase", "phage tail", "tail protein",
    "capsid", "portal protein", "terminase", "lysin", "holin", "lysis",
    "head protein", "baseplate",
    "plasmid", "conjugation", "conjugal", "tra gene", "trb gene",
    "mobA", "mobB", "mobC", "relaxase", "type IV secretion", "T4SS", "Pilus",
    "genomic island", "island", "integrative", "ICE", "IME", "cargo gene",
    "att site", "attachment site", "acquired", "horizontal transfer",
    "HGT", "mobile region", "MGI", "CRISPR", "Cas"
]

TRAIN_TEST_RATIO = 3 # 训练：测试=3: 1
RANDOM_SEED = 42  # 固定种子，确保拆分结果可复现

INSERT_LENGTH_MIN = 50  # 插入片段最小长度（bp）
INSERT_LENGTH_MAX = 300  # 插入片段最大长度（bp）
FRAME_SHIFT_RATIO = 0.15  # 15%的插入片段为非3倍数（模拟移码突变）
INSERT_PROB = 0.5  # 50%的宿主测试集CDS插入HGT（HGT概率）
AVOID_REGION = 50  # 避开起始密码子后/终止密码子前50bp

# -------------------------------------------------------------------

def split_train_test(filtered_records):
    """
    3:1比例拆分训练集和测试集，均保留完整CDS
    filtered_records：预处理后的外源CDS序列列表（SeqRecord对象）
    """

    total_count = len(filtered_records)
    print(f"Total number of core CDS: {total_count}")

    random.seed(RANDOM_SEED)
    random.shuffle(filtered_records)

    train_count = int(total_count * TRAIN_TEST_RATIO / (TRAIN_TEST_RATIO + 1))
    test_count = total_count - train_count
    
    train_records = filtered_records[:train_count]
    test_records = filtered_records[train_count:]

    SeqIO.write(train_records, OUTPUT_TRAIN, "fasta")
    SeqIO.write(test_records, OUTPUT_TEST, "fasta")

    print(f"Train(3/4): {len(train_records)} CDS → {OUTPUT_TRAIN}")
    print(f"Test(1/4): {len(test_records)} CDS→ {OUTPUT_TEST}")

def generate_insert_fragments(foreign_test_file):
    """
    从外源测试集生成插入片段（遵循规则：50-300bp，15%非3倍数）
    返回：插入片段列表（含片段序列、来源ID、长度、是否移码）
    """
    foreign_records = list(SeqIO.parse(foreign_test_file, "fasta"))
    insert_fragments = []
    random.seed(RANDOM_SEED)

    for foreign_rec in foreign_records:
        foreign_seq = foreign_rec.seq
        foreign_len = len(foreign_seq)
        
        if foreign_len < INSERT_LENGTH_MIN:
            continue
        
        if random.random() < FRAME_SHIFT_RATIO:
            max_possible_len = min(INSERT_LENGTH_MAX, foreign_len)
            insert_len = random.randint(INSERT_LENGTH_MIN, max_possible_len)
            while insert_len % 3 == 0:
                insert_len = random.randint(INSERT_LENGTH_MIN, max_possible_len)
        else:
            max_possible_len = min(INSERT_LENGTH_MAX, foreign_len)
            max_3x_len = (max_possible_len // 3) * 3
            min_3x_len = (INSERT_LENGTH_MIN // 3) * 3
            if min_3x_len < INSERT_LENGTH_MIN:
                min_3x_len += 3
            insert_len = random.randint(min_3x_len // 3, max_3x_len // 3) * 3

        max_start = foreign_len - insert_len
        start_pos = random.randint(0, max_start)
        insert_seq = foreign_seq[start_pos:start_pos+insert_len]

        insert_fragments.append({
            "seq": insert_seq,
            "source_id": foreign_rec.id,
            "insert_len": insert_len,
            "is_frame_shift": insert_len % 3 != 0,
            "start_in_foreign": start_pos,
            "foreign_original_len": foreign_len
        })
    
    print(f"Generated {len(insert_fragments)} insert fragments (15% frame-shift)")
    return insert_fragments

def insert_hgt_into_host(host_test_file, insert_fragments):
    """
    将外源片段插入宿主测试集
    返回：插入后的宿主序列列表 + metadata信息列表
    """
    host_records = list(SeqIO.parse(host_test_file, "fasta"))
    inserted_host_records = []
    metadata_list = []
    random.seed(RANDOM_SEED)

    for host_rec in host_records:
        host_id = host_rec.id
        host_seq = host_rec.seq
        host_len = len(host_seq)
        has_hgt = "no"
        reason = "randomly_not_inserted"
        foreign_source_id = ""
        insert_length = ""
        is_frame_shift = ""
        insert_position = ""
        host_final_length = host_len

        if random.random() < INSERT_PROB:
            fragment = random.choice(insert_fragments)
            insert_len = fragment["insert_len"]
            
            required_min_len = AVOID_REGION * 2 + insert_len
            if host_len < required_min_len:
                reason = "host_cds_too_short_for_insert"
            else:
                min_insert_pos = AVOID_REGION
                max_insert_pos = host_len - AVOID_REGION - insert_len
                if max_insert_pos < min_insert_pos:
                    reason = "insert_length_too_long_for_host"
                else:
                    insert_pos = random.randint(min_insert_pos, max_insert_pos)
                    new_host_seq = host_seq[:insert_pos] + fragment["seq"] + host_seq[insert_pos:]

                    has_hgt = "yes"
                    reason = ""
                    foreign_source_id = fragment["source_id"]
                    insert_length = insert_len
                    is_frame_shift = fragment["is_frame_shift"]
                    insert_position = insert_pos
                    host_final_length = len(new_host_seq)

                    new_host_rec = host_rec[:]
                    new_host_rec.id = f"{host_id}_hgt_inserted"
                    new_host_rec.description = (
                        f"{host_rec.description} "
                        f"[hgt=yes] [foreign_source={foreign_source_id}] "
                        f"[insert_length={insert_length}] [is_frame_shift={is_frame_shift}] "
                        f"[insert_position={insert_position}]"
                    ).strip()
                    new_host_rec.seq = new_host_seq
                    host_rec = new_host_rec

        inserted_host_records.append(host_rec)
        metadata_list.append({
            "host_id": host_id,
            "has_hgt": has_hgt,
            "reason": reason,
            "foreign_source_id": foreign_source_id,
            "insert_length": insert_length,
            "is_frame_shift": is_frame_shift,
            "insert_position": insert_position,
            "host_original_length": host_len,
            "host_final_length": host_final_length,
            "foreign_original_len": fragment["foreign_original_len"] if has_hgt == "yes" else ""
        })
    
    inserted_count = sum(1 for meta in metadata_list if meta["has_hgt"] == "yes")
    total_host = len(host_records)
    frame_shift_count = sum(1 for meta in metadata_list if meta["is_frame_shift"] is True)
    print(f"Inserted HGT into {inserted_count}/{total_host} host CDS")
    print(f"Frame-shift insertions: {frame_shift_count}/{inserted_count} (15% target)")
    return inserted_host_records, metadata_list

def write_metadata(metadata_list, output_file):
    """将插入信息写入metadata表格（CSV格式）"""
    headers = [
        "host_id", "has_hgt", "reason", "foreign_source_id",
        "insert_length", "is_frame_shift", "insert_position",
        "host_original_length", "host_final_length", "foreign_original_len"
    ]
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(metadata_list)
    print(f"Metadata saved to: {output_file}")


def write_truth_tsv(metadata_list, output_file):
    """
    保存GeneID -> TrueState(0=host, 2=foreign)的标签文件。
    如果插入了HGT，则实际序列ID会添加 _hgt_inserted 后缀。
    """
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.write("GeneID\tTrueState\n")
        for meta in metadata_list:
            gene_id = meta["host_id"] + ("_hgt_inserted" if meta["has_hgt"] == "yes" else "")
            true_state = 2 if meta["has_hgt"] == "yes" else 0
            f.write(f"{gene_id}\t{true_state}\n")
    print(f"Truth labels saved to: {output_file}")

def main():
    try:
        # 1. 读取外源CDS文件
        all_records = list(SeqIO.parse(CDS_FILE, "fasta"))
        print(f"{len(all_records)} total CDS loaded from {CDS_FILE}")

        # 2. 长度筛选
        filtered_by_length = []
        for record in all_records:
            if len(record.seq) >= MIN_CDS_LENGTH:
                filtered_by_length.append(record)
        print(f"{len(filtered_by_length)} CDS remain after length filtering (≥{MIN_CDS_LENGTH} bp)")

        # 3. 序列合法性筛选（仅保留A/T/G/C，N占比≤5%）
        filtered_by_seq_validity = []
        for record in filtered_by_length:
            seq = record.seq
            length = len(seq)

            invalid_chars = [char for char in seq if char not in VALID_NUCLEOTIDES]
            if invalid_chars:
                continue
            
            n_count = seq.upper().count("N")
            n_ratio = n_count / length if length > 0 else 0.0
            if n_ratio > N_EVAL_THRESHOLD:
                continue
            
            filtered_by_seq_validity.append(record)
        print(f"{len(filtered_by_seq_validity)} CDS remain after sequence validity filtering (N ratio ≤{N_EVAL_THRESHOLD*100}%)")

        # 4. 剔除含移动元件相关关键词的CDS
        filtered_by_mobile = []
        mobile_count = 0
        for record in filtered_by_seq_validity:
            description = record.description.lower()
            if any(keyword.lower() in description for keyword in mobile_keywords):
                mobile_count += 1
            else:
                filtered_by_mobile.append(record)
        print(f"{len(filtered_by_mobile)} CDS remain after removing {mobile_count} mobile element-related CDS")

        
        final_filtered_records = filtered_by_mobile
        final_filtered_ids = [record.id for record in final_filtered_records]

        # 5. 保存结果
        SeqIO.write(final_filtered_records, OUTPUT_FILTERED, "fasta")
        with open(OUTPUT_IDS, "w", encoding="utf-8") as f:
            f.write("\n".join(final_filtered_ids))

        # 6. 拆分训练集和测试集
        split_train_test(final_filtered_records)

        # 7. 模拟HGT插入
        insert_fragments = generate_insert_fragments(OUTPUT_TEST)
        if not insert_fragments:
            raise ValueError("No valid insert fragments generated (check foreign test set length)")
        
        inserted_host_records, metadata_list = insert_hgt_into_host(HOST_TEST_FILE, insert_fragments)
        
        SeqIO.write(inserted_host_records, OUTPUT_HOST_HGT, "fasta")
        print(f"Host test set with HGT saved to: {OUTPUT_HOST_HGT}")

        write_metadata(metadata_list, OUTPUT_METADATA)
        write_truth_tsv(metadata_list, OUTPUT_TRUTH)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
