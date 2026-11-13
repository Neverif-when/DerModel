import os
import subprocess
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

# 设置Gaussian环境变量
os.environ['GAUSS_EXEDIR'] = r'D:\Gaussian\G16W'
os.environ['g16root'] = r'D:\Gaussian'

# 可执行程序路径
G16_EXE = r"D:\Gaussian\G16W\g16.exe"
FORMCHK_EXE = r"D:\Gaussian\G16W\formchk.exe"
CUBEGEN_EXE = r"D:\Gaussian\G16W\cubegen.exe"

# 工作目录
WORK_DIR = r"D:\TumorCNN\dataset"
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)

# SDF文件夹路径
SDF_DIR = r"D:\TumorCNN\dataset\3D_SDF"

# Cube文件夹路径（用于存放已有的cube文件）
CUBE_DIR = r"D:\TumorCNN\dataset\Cube"
os.makedirs(CUBE_DIR, exist_ok=True)


def create_input(sdf_path, mol_name, method="B3LYP", basis="6-31G*", charge=0, mult=1):
    """创建Gaussian输入文件"""
    try:
        mols = list(Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False))
        if not mols or mols[0] is None:
            print(f"  ✗ 无法读取SDF文件: {sdf_path}")
            return None

        mol = mols[0]

        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()

        with open(f"{mol_name}.com", "w") as f:
            f.write(f"%chk={mol_name}.chk\n")
            f.write("%mem=8GB\n")
            f.write("%nprocshared=8\n")
            f.write(f"#p {method}/{basis} density=current scf=tight\n")
            f.write(f"\n{mol_name} calculation\n\n")
            f.write(f"{charge} {mult}\n")

            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                f.write(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")
            f.write("\n")

        print(f"  ✓ 已生成输入文件: {mol_name}.com")
        return mol

    except Exception as e:
        print(f"  ✗ 创建输入文件失败: {e}")
        return None


def run_calculation(mol_name):
    """执行计算流程"""
    if not os.path.exists(f"{mol_name}.com"):
        return False

    print(f"  运行Gaussian计算...")
    try:
        with open(f"{mol_name}.log", "w") as log_file:
            result = subprocess.run(
                f'"{G16_EXE}" < "{mol_name}.com"',
                shell=True,
                stdout=log_file,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"  ✗ Gaussian计算失败，返回码: {result.returncode}")
                if result.stderr:
                    error_msg = result.stderr[:200]
                    if "Error termination" in error_msg:
                        print(f"  错误信息: {error_msg}")
                return False

    except Exception as e:
        print(f"  ✗ 运行Gaussian时出错: {e}")
        return False

    if not os.path.exists(f"{mol_name}.chk"):
        print(f"  ✗ 未生成chk文件，计算可能失败")
        return False

    print(f"  转换文件格式...")
    try:
        result = subprocess.run(
            [FORMCHK_EXE, f"{mol_name}.chk", f"{mol_name}.fchk"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  ✗ formchk转换失败: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ 运行formchk时出错: {e}")
        return False

    print(f"  生成电子密度文件...")
    try:
        result = subprocess.run(
            [CUBEGEN_EXE, "0", "density=scf", f"{mol_name}.fchk",
             f"{mol_name}_density.cube", "0", "h"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  ✗ cubegen生成失败: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ 运行cubegen时出错: {e}")
        return False

    print(f"  ✓ 完成！电子密度文件: {mol_name}_density.cube")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("批量计算电子密度 - 单点能计算（快速）")
    print("=" * 70)

    # ---------- 预扫描阶段 ----------
    all_sdf = sorted([f for f in os.listdir(SDF_DIR) if f.lower().endswith(".sdf")])
    total = len(all_sdf)

    existing_mols = []      # 已有cube的分子名
    need_calc_mols = []     # 需要计算的分子名

    for filename in all_sdf:
        mol_name = os.path.splitext(filename)[0]
        cube_path = os.path.join(CUBE_DIR, f"{mol_name}_density.cube")
        if os.path.exists(cube_path):
            existing_mols.append(mol_name)
        else:
            need_calc_mols.append(mol_name)

    print(f"SDF文件夹: {SDF_DIR}")
    print(f"Cube文件夹: {CUBE_DIR}")
    print(f"输出目录: {WORK_DIR}")
    print(f"共发现 {total} 个SDF文件")
    print(f"已有Cube文件: {len(existing_mols)} 个")
    print(f"需要计算: {len(need_calc_mols)} 个分子")
    print("=" * 70)

    if len(need_calc_mols) == 0:
        print("所有分子均已计算完毕，无需执行新计算。")
        sys.exit(0)

    # 可选：列出前几个需要计算的分子（防止输出太长）
    if len(need_calc_mols) <= 20:
        print("需要计算的分子列表:")
        for name in need_calc_mols:
            print(f"  - {name}")
    else:
        print(f"需要计算的分子较多（共{len(need_calc_mols)}个），仅显示前10个:")
        for name in need_calc_mols[:10]:
            print(f"  - {name}")
        print("  ... (剩余略)")
    print("=" * 70)

    # ---------- 开始计算 ----------
    success_count = 0
    fail_count = 0
    skip_count = len(existing_mols)   # 跳过的分子数即为已有cube的分子数
    error_files = []

    for i, mol_name in enumerate(need_calc_mols, 1):
        filename = mol_name + ".sdf"
        sdf_path = os.path.join(SDF_DIR, filename)

        print(f"\n[{i}/{len(need_calc_mols)}] 处理分子: {mol_name}")
        print("-" * 50)

        try:
            mol = create_input(sdf_path, mol_name,
                               method="B3LYP",
                               basis="6-31G*",
                               charge=0, mult=1)

            if mol is None:
                fail_count += 1
                error_files.append(f"{filename}: 无法创建输入文件")
                continue

            try:
                num_atoms = mol.GetNumAtoms()
                formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
                print(f"  原子数: {num_atoms}, 分子式: {formula}")
            except Exception as e:
                print(f"  ⚠ 无法获取完整分子信息: {e}")
                print(f"  继续计算...")

            if run_calculation(mol_name):
                success_count += 1
                print(f"  ✓ {mol_name} 计算成功！")
            else:
                fail_count += 1
                error_files.append(f"{filename}: 计算失败")
                print(f"  ✗ {mol_name} 计算失败！")

        except KeyboardInterrupt:
            print(f"\n⚠ 用户中断计算")
            print(f"已完成 {success_count + fail_count} / {len(need_calc_mols)} 个需计算分子，成功 {success_count} 个，跳过 {skip_count} 个")
            sys.exit(0)

        except Exception as e:
            fail_count += 1
            error_files.append(f"{filename}: 未预期的错误 - {str(e)[:100]}")
            print(f"  ✗ 处理 {mol_name} 时发生错误: {e}")
            print(f"  跳过此文件，继续处理下一个...")
            continue

    print(f"\n" + "=" * 70)
    print(f"计算完成！")
    print(f"成功: {success_count} 个分子")
    print(f"失败: {fail_count} 个分子")
    print(f"跳过（已有Cube）: {skip_count} 个分子")
    print(f"总计处理: {success_count + fail_count + skip_count} / {total} 个SDF文件")
    print(f"文件保存在: {WORK_DIR}")

    if error_files:
        print(f"\n出错的文件列表:")
        for error in error_files:
            print(f"  - {error}")

    print("=" * 70)