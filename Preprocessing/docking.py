#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量分子对接脚本 (纯Vina引擎) - 稳定版
固定随机种子 + 动态超时 + 动态Worker分配
自动将SDF转换为PDBQT (Open Babel)
动态盒子大小 (根据配体大小)
"""

import os, sys, re, glob, subprocess, logging
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============== 日志配置（绝对路径） ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, 'docking_vina_only.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============== 配置 ==============
VINA_EXE = r"D:\AutoDock-Vina\build\linux\release\vina.exe"   # Windows版 Vina 1.2.7
PROTEIN_PDBQT_DIR = r"D:\TumorCNN\dataset\Dock\Protein_pdbqt"
LIGAND_DIR = r"D:\TumorCNN\dataset\3D_SDF"
POCKET_DIR = os.path.join(SCRIPT_DIR, "pocket_cleaned")
OUTPUT_DIR = SCRIPT_DIR

LOCAL_BOX_SIZE = (20, 20, 20)          # 备用
GLOBAL_BOX_SIZE = (100, 100, 100)      # 无口袋时仍用大盒子
BOX_PADDING = 4.0                      # 配体外扩距离 (Å)
MAX_LOCAL_POCKETS = 5
VINA_EXHAUSTIVENESS = 16
VINA_NUM_MODES = 5
VINA_ENERGY_RANGE = 3
VINA_SEED = 42
MAX_WORKERS = 8
LARGE_PROTEIN_THRESHOLD_MB = 10
TIMEOUT_PER_TASK = 1000
# ==================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024) if os.path.isfile(filepath) else 0

def get_workers_for_protein(size_mb):
    if size_mb > LARGE_PROTEIN_THRESHOLD_MB:
        return 2
    ratio = min(size_mb / LARGE_PROTEIN_THRESHOLD_MB, 1.0)
    workers = int(8 - ratio * 4)
    return max(4, min(8, workers))

def read_pocket_csv(pocket_file):
    if not os.path.exists(pocket_file): return []
    try:
        df = pd.read_csv(pocket_file, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        if 'score' in df.columns:
            df = df.sort_values('score', ascending=False)
        pockets = []
        for _, row in df.iterrows():
            if len(pockets) >= MAX_LOCAL_POCKETS: break
            try:
                pockets.append((float(row['center_x']), float(row['center_y']), float(row['center_z'])))
            except: continue
        return pockets
    except Exception as e:
        logger.warning(f"Parse pocket file failed: {e}")
        return []

def compute_protein_center_from_pdbqt(pdbqt_file):
    x, y, z = [], [], []
    try:
        with open(pdbqt_file) as f:
            for line in f:
                if line.startswith(("ATOM","HETATM")):
                    try:
                        x.append(float(line[30:38])); y.append(float(line[38:46])); z.append(float(line[46:54]))
                    except: pass
    except: pass
    if not x: return (0,0,0)
    return (sum(x)/len(x), sum(y)/len(y), sum(z)/len(z))

def detect_metal_in_ligand(ligand_pdbqt):
    metals = {'PT','FE','ZN','CU','MG','MN','CO','NI','MO','RU','RH','PD','AG','CD','IR','AU','HG'}
    try:
        with open(ligand_pdbqt) as f:
            for line in f:
                if line.startswith(("ATOM","HETATM")):
                    parts = line.split()
                    if len(parts)>=12:
                        elem = ''.join(filter(str.isalpha, parts[-1].upper()))
                        if elem in metals: return True
        return False
    except: return False

def replace_metal_to_zn_in_pdbqt(ligand_pdbqt, output_dir):
    base = os.path.splitext(os.path.basename(ligand_pdbqt))[0]
    new_file = os.path.join(output_dir, base+"_tozn.pdbqt")
    if os.path.exists(new_file): return new_file
    metals = {'PT','FE','ZN','CU','MG','MN','CO','NI','MO','RU','RH','PD','AG','CD','IR','AU','HG'}
    try:
        with open(ligand_pdbqt) as f: lines = f.readlines()
        new_lines = []
        for line in lines:
            if line.startswith(("ATOM","HETATM")):
                parts = line.split()
                if len(parts)>=12:
                    elem = ''.join(filter(str.isalpha, parts[-1].upper()))
                    if elem in metals:
                        parts[-1] = "Zn"
                        line = " ".join(parts)+"\n"
            new_lines.append(line)
        with open(new_file,'w') as f: f.writelines(new_lines)
        return new_file
    except Exception as e:
        logger.error(f"Replace metal failed {ligand_pdbqt}: {e}")
        return None

def count_rotatable_bonds(pdbqt_file):
    count = 0
    try:
        with open(pdbqt_file) as f:
            for line in f:
                if line.startswith("BRANCH"):
                    count += 1
    except Exception as e:
        logger.warning(f"Could not count rotatable bonds in {pdbqt_file}: {e}")
    return count

# ---------- 根据配体PDBQT计算动态盒子大小 ----------
def compute_ligand_box_size(ligand_pdbqt, padding=BOX_PADDING):
    xs, ys, zs = [], [], []
    try:
        with open(ligand_pdbqt) as f:
            for line in f:
                if line.startswith(("ATOM","HETATM")):
                    try:
                        xs.append(float(line[30:38]))
                        ys.append(float(line[38:46]))
                        zs.append(float(line[46:54]))
                    except:
                        continue
        if not xs:
            return LOCAL_BOX_SIZE
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)
        size_x = (x_max - x_min) + 2 * padding
        size_y = (y_max - y_min) + 2 * padding
        size_z = (z_max - z_min) + 2 * padding
        size_x = max(size_x, 10)
        size_y = max(size_y, 10)
        size_z = max(size_z, 10)
        return (size_x, size_y, size_z)
    except Exception as e:
        logger.warning(f"Could not compute box size for {ligand_pdbqt}: {e}")
        return LOCAL_BOX_SIZE
# --------------------------------------------------

def run_vina(receptor_pdbqt, ligand_pdbqt, center, box_size, output_prefix):
    rotors = count_rotatable_bonds(ligand_pdbqt)
    if rotors >= 10:
        local_exhaust = 8
        local_timeout = 1500
    elif rotors >= 6:
        local_exhaust = 12
        local_timeout = 1200
    else:
        local_exhaust = VINA_EXHAUSTIVENESS
        local_timeout = TIMEOUT_PER_TASK

    try:
        cx, cy, cz = center; sx, sy, sz = box_size
        out_pdbqt = output_prefix+"_out.pdbqt"
        cmd = [VINA_EXE, "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt,
               "--out", out_pdbqt,
               "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
               "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
               "--exhaustiveness", str(local_exhaust),
               "--num_modes", str(VINA_NUM_MODES),
               "--energy_range", str(VINA_ENERGY_RANGE),
               "--seed", str(VINA_SEED),
               "--cpu", "1"]
        logger.debug(f"Running Vina: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=local_timeout)
        if result.returncode != 0:
            logger.error(f"  Vina non-zero exit ({result.returncode}) for {output_prefix}")
            logger.error(f"  stderr: {result.stderr[:500]}")
            return None
        if not os.path.isfile(out_pdbqt) or os.path.getsize(out_pdbqt)<100:
            logger.warning(f"  Output file missing or too small for {output_prefix}")
            return None
        best = None
        with open(out_pdbqt) as f:
            for line in f:
                m = re.search(r'REMARK VINA RESULT:\s*(-?\d+\.?\d*)', line)
                if m:
                    v = float(m.group(1))
                    if best is None or v<best: best=v
        return best
    except subprocess.TimeoutExpired:
        logger.error(f"  Vina timeout ({local_timeout}s, rotors={rotors}): {output_prefix}")
        return None
    except Exception as e:
        stderr = ""
        if hasattr(e, 'stderr') and e.stderr:
            stderr = e.stderr[:500] if isinstance(e.stderr, str) else e.stderr.decode(errors='ignore')[:500]
        logger.error(f"  Vina exception for {output_prefix}: {e}")
        if stderr:
            logger.error(f"  stderr: {stderr}")
        return None

# ---------- SDF → PDBQT 转换函数 (Open Babel) ----------
def convert_sdf_to_pdbqt(sdf_dir, pdbqt_dir):
    OBABEL_EXE = r"D:\Anaconda\docking_env\Library\bin\obabel.exe"
    if not os.path.isfile(OBABEL_EXE):
        logger.error(f"Open Babel executable not found at {OBABEL_EXE}")
        return 0

    sdf_files = glob.glob(os.path.join(sdf_dir, "*.sdf"))
    if not sdf_files:
        logger.warning(f"No .sdf files found in {sdf_dir}")
        return 0

    converted = 0
    for sdf_path in sdf_files:
        base = os.path.splitext(os.path.basename(sdf_path))[0]
        pdbqt_path = os.path.join(pdbqt_dir, base + ".pdbqt")
        if os.path.exists(pdbqt_path):
            logger.debug(f"  {base}.pdbqt already exists, skipping.")
            converted += 1
            continue

        cmd = [OBABEL_EXE, sdf_path, "-O", pdbqt_path, "-xh"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.isfile(pdbqt_path):
                logger.info(f"  Converted {base}.sdf → {base}.pdbqt")
                converted += 1
            else:
                logger.warning(f"  Failed to convert {base}.sdf: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            logger.warning(f"  Timeout converting {base}.sdf")
        except Exception as e:
            logger.warning(f"  Error converting {base}.sdf: {e}")
    return converted
# -------------------------------------------------

# ---------- process_one_protein ----------
def process_one_protein(protein_name, receptor_pdbqt, ligand_pdbqt_list, centers, tmp_dir):
    results = []
    for lig_pdbqt in ligand_pdbqt_list:
        lig_name = os.path.splitext(os.path.basename(lig_pdbqt))[0]
        best_aff = float('inf')
        if detect_metal_in_ligand(lig_pdbqt):
            lig_zn = replace_metal_to_zn_in_pdbqt(lig_pdbqt, tmp_dir)
            if lig_zn is None:
                logger.warning(f"    Replace metal failed, skip {lig_name} on {protein_name}")
                continue
            used_lig = lig_zn
        else:
            used_lig = lig_pdbqt

        dynamic_box_size = compute_ligand_box_size(used_lig)

        for idx, center in enumerate(centers):
            prefix = os.path.join(tmp_dir, f"{protein_name}_{lig_name}_box{idx}")
            aff = run_vina(receptor_pdbqt, used_lig, center, dynamic_box_size, prefix)
            if aff is not None and aff < best_aff: best_aff = aff
        if best_aff != float('inf'):
            logger.info(f"  {lig_name} -> {protein_name}: {best_aff:.2f}")
            results.append((lig_name, best_aff))
        else:
            logger.warning(f"  Warning: All docking failed for {lig_name} on {protein_name}")
    return results
# -------------------------------------------------

def main():
    # 检查 Vina 是否可执行
    if not os.path.isfile(VINA_EXE):
        logger.error(f"Vina executable not found at {VINA_EXE}")
        sys.exit(1)
    try:
        test = subprocess.run([VINA_EXE, "--help"], capture_output=True, text=True, timeout=10)
        if test.returncode != 0:
            logger.error(f"Vina test failed: {test.stderr[:200]}")
            sys.exit(1)
        else:
            logger.info("Vina executable verified successfully.")
    except FileNotFoundError:
        logger.error(f"Vina executable not found at {VINA_EXE}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Vina test raised exception: {e}")
        sys.exit(1)

    tmp_dir = os.path.join(OUTPUT_DIR, "tmp")
    ensure_dir(tmp_dir)

    # 1. 扫描蛋白质
    logger.info("="*60); logger.info("Step 1: Scan protein PDBQT files from directory")
    if not os.path.isdir(PROTEIN_PDBQT_DIR):
        sys.exit(f"Protein PDBQT directory not exist: {PROTEIN_PDBQT_DIR}")
    protein_files = glob.glob(os.path.join(PROTEIN_PDBQT_DIR, "*.pdbqt"))
    if not protein_files:
        sys.exit("No protein pdbqt files found")
    PROTEIN_LIST = sorted([os.path.splitext(os.path.basename(f))[0] for f in protein_files])
    logger.info(f"Found {len(PROTEIN_LIST)} proteins: {', '.join(PROTEIN_LIST[:5])}{'...' if len(PROTEIN_LIST)>5 else ''}")

    protein_pdbqt_map = {}
    for prot in PROTEIN_LIST:
        pdbqt_file = os.path.join(PROTEIN_PDBQT_DIR, f"{prot}.pdbqt")
        if os.path.isfile(pdbqt_file):
            protein_pdbqt_map[prot] = pdbqt_file
    logger.info(f"Total usable proteins: {len(protein_pdbqt_map)}")

    # 2. 配体准备
    if not os.path.isdir(LIGAND_DIR):
        sys.exit(f"Ligand directory not exist: {LIGAND_DIR}")

    ligands = glob.glob(os.path.join(LIGAND_DIR, "*.pdbqt"))
    if not ligands:
        logger.info("No .pdbqt files found in ligand directory. Attempting SDF → PDBQT conversion...")
        converted = convert_sdf_to_pdbqt(LIGAND_DIR, LIGAND_DIR)
        if converted == 0:
            sys.exit("No .sdf files found or conversion failed. Please provide .pdbqt or .sdf files in the ligand directory.")
        ligands = glob.glob(os.path.join(LIGAND_DIR, "*.pdbqt"))
        if not ligands:
            sys.exit("Conversion produced no .pdbqt files. Check your SDF files.")
    logger.info(f"Found {len(ligands)} ligand PDBQT files")

    # 3. 预计算中心点
    protein_centers = {}
    for prot in PROTEIN_LIST:
        if prot not in protein_pdbqt_map: continue
        rec = protein_pdbqt_map[prot]
        pockets = read_pocket_csv(os.path.join(POCKET_DIR, f"{prot}.pdb_predictions.csv"))
        if pockets:
            centers = pockets
        else:
            center = compute_protein_center_from_pdbqt(rec)
            centers = [center]
        protein_centers[prot] = centers

    # 4. 对接
    all_data = []
    for prot in PROTEIN_LIST:
        if prot not in protein_pdbqt_map: continue
        rec = protein_pdbqt_map[prot]; centers = protein_centers[prot]
        size_mb = get_file_size_mb(rec)
        n_workers = get_workers_for_protein(size_mb)
        logger.info(f"\n=== {prot} (size={size_mb:.1f}MB, workers={n_workers}) ===")
        chunk_size = max(1, len(ligands)//n_workers)
        chunks = [ligands[i:i+chunk_size] for i in range(0,len(ligands),chunk_size)]
        prot_res = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(process_one_protein, prot, rec, chk, centers, tmp_dir): chk for chk in chunks}
            for f in as_completed(futures):
                try: prot_res.extend(f.result())
                except Exception as e: logger.error(f"  Worker error: {e}")
        for lig, aff in prot_res:
            all_data.append({'Protein':prot,'Ligand':lig,'Affinity':aff})
        if all_data:
            pd.DataFrame(all_data).to_csv(os.path.join(OUTPUT_DIR,f"intermediate_{prot}.csv"), index=False)

    # 5. 热力图
    if all_data:
        df = pd.DataFrame(all_data).drop_duplicates(subset=['Protein','Ligand'])
        heatmap = df.pivot_table(index='Ligand', columns='Protein', values='Affinity', aggfunc='min')
        heatmap.to_csv(os.path.join(OUTPUT_DIR,"1_Final_Docking_Heatmap_Vina.csv"))
        logger.info(f"\nDone! Heatmap shape: {heatmap.shape}")
    else:
        logger.warning("No results generated.")

if __name__=="__main__":
    main()