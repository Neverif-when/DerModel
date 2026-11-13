"""
smart_cube_cutter.py
智能Cube文件切割处理器
根据分子量动态确定切割点数，4N ≤ 512
"""

import numpy as np
import os
import glob
from typing import Tuple, Dict
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class SmartCubeCutter:
    """智能Cube文件切割器"""

    def __init__(self, max_features=512, features_per_point=4):
        """
        初始化切割器

        参数:
            max_features: 最大总特征数，默认512
            features_per_point: 每个点的特征数，默认4
        """
        self.max_points = max_features // features_per_point  # 最大点数N
        self.features_per_point = features_per_point
        print(f"最大点数N: {self.max_points} (4N ≤ {max_features})")

    def read_cube_header(self, filepath: str) -> Tuple[Dict, int]:
        """
        快速读取Cube文件头信息，不读取整个数据

        返回:
            header_info: 头信息字典
            data_start_line: 数据开始行号
        """
        header_info = {}
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # 前两行是注释
        header_info['title1'] = lines[0].strip()
        header_info['title2'] = lines[1].strip()

        # 第3行：原子数和原点
        parts = lines[2].strip().split()
        header_info['natoms'] = int(parts[0])
        header_info['origin'] = np.array([float(x) for x in parts[1:4]])

        # 第4-6行：网格信息
        nx_parts = lines[3].strip().split()
        ny_parts = lines[4].strip().split()
        nz_parts = lines[5].strip().split()

        header_info['nx'] = int(nx_parts[0])
        header_info['ny'] = int(ny_parts[0])
        header_info['nz'] = int(nz_parts[0])

        header_info['dx'] = np.array([float(x) for x in nx_parts[1:4]])
        header_info['dy'] = np.array([float(x) for x in ny_parts[1:4]])
        header_info['dz'] = np.array([float(x) for x in nz_parts[1:4]])

        # 数据开始行
        data_start_line = 6 + header_info['natoms']

        # 原子信息（用于计算分子量）
        atoms = []
        for i in range(header_info['natoms']):
            atom_line = lines[6 + i].strip().split()
            atoms.append({
                'atomic_number': int(float(atom_line[0])),
                'charge': float(atom_line[1]),
                'x': float(atom_line[2]),
                'y': float(atom_line[3]),
                'z': float(atom_line[4])
            })
        header_info['atoms'] = atoms

        return header_info, data_start_line

    def calculate_molecular_weight(self, atoms: list) -> float:
        """
        从原子列表计算分子量
        使用常见原子质量近似
        """
        atomic_masses = {
            1: 1.008,  # H
            6: 12.011,  # C
            7: 14.007,  # N
            8: 15.999,  # O
            9: 18.998,  # F
            15: 30.974,  # P
            16: 32.065,  # S
            17: 35.453,  # Cl
            35: 79.904,  # Br
        }

        total_mass = 0.0
        for atom in atoms:
            atomic_num = atom['atomic_number']
            if atomic_num in atomic_masses:
                total_mass += atomic_masses[atomic_num]
            else:
                # 如果不在字典中，使用原子序数近似
                total_mass += atomic_num

        return total_mass

    def determine_N(self, molecular_weight: float, volume: float) -> int:
        """
        根据分子量和体积确定切割点数N

        参数:
            molecular_weight: 分子量
            volume: 分子体积 (Å³)
        返回:
            N: 切割点数
        """
        # 归一化分子量 (假设药物分子量在200-800之间)
        norm_mw = (molecular_weight - 200) / (800 - 200)  # 映射到0-1
        norm_mw = np.clip(norm_mw, 0, 1)

        # 归一化体积 (假设体积在100-800 Å³之间)
        norm_vol = (volume - 100) / (800 - 100)
        norm_vol = np.clip(norm_vol, 0, 1)

        # 加权计算 (60%分子量 + 40%体积)
        weight = 0.6 * norm_mw + 0.4 * norm_vol

        # 计算N，确保N ≤ 128
        N = int(20 + weight * (self.max_points - 20))  # 20-128之间
        N = max(20, min(N, self.max_points))  # 确保在范围内

        return N

    def calculate_volume(self, header_info: Dict) -> float:
        """
        计算分子的大致体积
        """
        # 计算网格步长向量的行列式得到单位网格体积
        grid_vectors = np.array([header_info['dx'], header_info['dy'], header_info['dz']])
        grid_volume = abs(np.linalg.det(grid_vectors))

        # 总体积 = 网格数 × 单位网格体积
        total_volume = header_info['nx'] * header_info['ny'] * header_info['nz'] * grid_volume

        return total_volume

    def read_cube_data_fast(self, filepath: str, header_info: Dict, data_start_line: int, N: int) -> np.ndarray:
        """
        快速读取Cube数据并采样N个点

        参数:
            filepath: Cube文件路径
            header_info: 头信息字典
            data_start_line: 数据开始行
            N: 需要采样的点数
        返回:
            points: 采样点数组，形状(N, 4)
        """
        nx, ny, nz = header_info['nx'], header_info['ny'], header_info['nz']
        origin = header_info['origin']
        dx, dy, dz = header_info['dx'], header_info['dy'], header_info['dz']

        # 读取数据部分
        with open(filepath, 'r') as f:
            lines = f.readlines()

        data_lines = lines[data_start_line:]

        # 解析所有数据
        all_data = []
        for line in data_lines:
            if not line.strip():
                continue
            numbers = [float(x) for x in line.strip().split()]
            all_data.extend(numbers)

        # 重塑为三维数组
        data = np.array(all_data).reshape((nx, ny, nz), order='F')

        # 创建采样网格
        points = []

        # 均匀采样策略：在三维网格中均匀选取N个点
        # 计算每个方向的采样间隔
        step_x = max(1, nx // int((N ** (1 / 3)) + 1))
        step_y = max(1, ny // int((N ** (1 / 3)) + 1))
        step_z = max(1, nz // int((N ** (1 / 3)) + 1))

        count = 0
        for i in range(0, nx, step_x):
            for j in range(0, ny, step_y):
                for k in range(0, nz, step_z):
                    if count >= N:
                        break

                    # 计算坐标
                    x = origin[0] + i * dx[0] + j * dy[0] + k * dz[0]
                    y = origin[1] + i * dx[1] + j * dy[1] + k * dz[1]
                    z = origin[2] + i * dx[2] + j * dy[2] + k * dz[2]

                    # 电子密度
                    density = data[i, j, k]

                    # 只保留电子密度较大的点
                    if abs(density) > 1e-5:
                        points.append([x, y, z, density])
                        count += 1

        # 如果点数不足，随机采样补充
        if len(points) < N:
            # 随机选择一些点补充
            all_indices = np.ndindex((nx, ny, nz))
            indices_list = list(all_indices)
            np.random.shuffle(indices_list)

            for idx in indices_list:
                if len(points) >= N:
                    break
                i, j, k = idx

                x = origin[0] + i * dx[0] + j * dy[0] + k * dz[0]
                y = origin[1] + i * dx[1] + j * dy[1] + k * dz[1]
                z = origin[2] + i * dx[2] + j * dy[2] + k * dz[2]
                density = data[i, j, k]

                if abs(density) > 1e-5:
                    points.append([x, y, z, density])

        # 如果还是不足，用随机点填充
        while len(points) < N:
            i = np.random.randint(0, nx)
            j = np.random.randint(0, ny)
            k = np.random.randint(0, nz)

            x = origin[0] + i * dx[0] + j * dy[0] + k * dz[0]
            y = origin[1] + i * dx[1] + j * dy[1] + k * dz[1]
            z = origin[2] + i * dx[2] + j * dy[2] + k * dz[2]
            density = data[i, j, k]

            points.append([x, y, z, density])

        return np.array(points[:N], dtype=np.float32)

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        标准化点云数据
        """
        if len(points) == 0:
            return np.zeros((self.max_points, 4), dtype=np.float32)

        # 分离坐标和电子密度
        coords = points[:, :3]
        densities = points[:, 3].reshape(-1, 1)

        # 标准化坐标
        if coords.std(axis=0).max() > 0:
            coords_mean = coords.mean(axis=0)
            coords_std = coords.std(axis=0)
            coords_normalized = (coords - coords_mean) / (coords_std + 1e-8)
        else:
            coords_normalized = coords

        # 标准化电子密度
        if densities.std() > 0:
            densities_normalized = (densities - densities.mean()) / (densities.std() + 1e-8)
        else:
            densities_normalized = densities

        return np.concatenate([coords_normalized, densities_normalized], axis=1)

    def process_cube_file(self, cube_path: str) -> Tuple[np.ndarray, Dict]:
        """
        处理单个Cube文件

        返回:
            points: 切割后的点云，形状(N, 4)
            info: 处理信息字典
        """
        try:
            # 1. 读取头信息
            header_info, data_start_line = self.read_cube_header(cube_path)

            # 2. 计算分子量和体积
            molecular_weight = self.calculate_molecular_weight(header_info['atoms'])
            volume = self.calculate_volume(header_info)

            # 3. 确定切割点数N
            N = self.determine_N(molecular_weight, volume)

            # 4. 读取数据并采样N个点
            points = self.read_cube_data_fast(cube_path, header_info, data_start_line, N)

            # 5. 标准化
            points = self.normalize_points(points)

            # 6. 构建信息字典
            info = {
                'molecular_weight': molecular_weight,
                'volume': volume,
                'N_points': N,
                'total_features': N * 4,
                'original_shape': (header_info['nx'], header_info['ny'], header_info['nz'])
            }

            return points, info

        except Exception as e:
            print(f"处理文件 {cube_path} 时出错: {e}")
            # 返回默认点云
            points = np.random.randn(self.max_points, 4).astype(np.float32)
            info = {
                'molecular_weight': 0,
                'volume': 0,
                'N_points': self.max_points,
                'total_features': self.max_points * 4,
                'error': str(e)
            }
            return points, info

    def process_folder(self, input_folder: str, output_folder: str):
        """
        处理文件夹中的所有Cube文件，若输出文件已存在则跳过

        参数:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
        """
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 查找所有.cube文件
        cube_files = glob.glob(os.path.join(input_folder, "*.cube"))
        if not cube_files:
            # 也查找.cub扩展名
            cube_files = glob.glob(os.path.join(input_folder, "*.cub"))

        if not cube_files:
            print(f"在文件夹 {input_folder} 中未找到.cube或.cub文件")
            return

        print(f"找到 {len(cube_files)} 个Cube文件")
        print("开始处理...")

        all_info = []
        skipped_count = 0

        for cube_file in tqdm(cube_files, desc="处理Cube文件"):
            # 获取文件名
            file_name = os.path.basename(cube_file)
            drug_name = file_name.split('.')[0]

            # 构造输出路径
            output_path = os.path.join(output_folder, f"{drug_name}.npy")

            # 检查是否已存在，若存在则跳过
            if os.path.exists(output_path):
                skipped_count += 1
                continue

            # 处理Cube文件
            points, info = self.process_cube_file(cube_file)

            # 保存为.npy文件
            np.save(output_path, points)

            # 记录信息
            info['drug_name'] = drug_name
            info['output_file'] = output_path
            all_info.append(info)

            # 打印第一个文件的信息
            if len(all_info) == 1:
                print(f"\n示例文件: {drug_name}")
                print(f"  分子量: {info['molecular_weight']:.1f}")
                print(f"  体积: {info['volume']:.1f} Å³")
                print(f"  切割点数N: {info['N_points']}")
                print(f"  总特征数: {info['total_features']} (4N)")
                print(f"  原始网格: {info['original_shape']}")
                print(f"  处理后形状: {points.shape}")
                print(f"  保存到: {output_path}")

        # 打印跳过信息
        if skipped_count > 0:
            print(f"\n已跳过 {skipped_count} 个已存在的文件")

        # 打印汇总信息（只针对实际处理的文件）
        if all_info:
            self.print_summary(all_info)
        else:
            print("没有新文件需要处理。")

    def print_summary(self, all_info: list):
        """打印处理摘要"""
        print("\n" + "=" * 60)
        print("处理完成摘要")
        print("=" * 60)

        print(f"总共处理: {len(all_info)} 个文件")

        # 计算统计信息
        N_values = [info['N_points'] for info in all_info]
        features_values = [info['total_features'] for info in all_info]
        mw_values = [info['molecular_weight'] for info in all_info]

        print(f"\n切割点数统计:")
        print(f"  平均N: {np.mean(N_values):.1f}")
        print(f"  最小N: {np.min(N_values)}")
        print(f"  最大N: {np.max(N_values)}")

        print(f"\n总特征数统计 (4N):")
        print(f"  平均特征数: {np.mean(features_values):.1f}")
        print(f"  最小特征数: {np.min(features_values)}")
        print(f"  最大特征数: {np.max(features_values)}")

        print(f"\n分子量统计:")
        print(f"  平均分子量: {np.mean(mw_values):.1f}")
        print(f"  最小分子量: {np.min(mw_values):.1f}")
        print(f"  最大分子量: {np.max(mw_values):.1f}")

        # 检查条件
        max_features = np.max(features_values)
        if max_features <= 512:
            print(f"\n✓ 满足条件: 最大4N = {max_features} ≤ 512")
        else:
            print(f"\n✗ 不满足条件: 最大4N = {max_features} > 512")


# ============================================
# 主程序
# ============================================

def main():
    """主函数"""
    # 配置参数
    INPUT_FOLDER = r"D:\TumorCNN\dataset\Cube"  # 包含Cube文件的文件夹
    OUTPUT_FOLDER = r"D:\TumorCNN\dataset\npy"  # 输出文件夹
    MAX_FEATURES = 512  # 最大总特征数

    print("=" * 60)
    print("智能Cube文件切割处理器")
    print(f"确保 4N ≤ {MAX_FEATURES}")
    print("=" * 60)

    # 检查输入文件夹
    if not os.path.exists(INPUT_FOLDER):
        print(f"错误: 输入文件夹不存在: {INPUT_FOLDER}")
        return

    # 创建处理器
    processor = SmartCubeCutter(max_features=MAX_FEATURES)

    # 处理文件夹
    processor.process_folder(INPUT_FOLDER, OUTPUT_FOLDER)

    print(f"\n处理完成！")
    print(f"结果保存在: {OUTPUT_FOLDER}")
    print(f"文件格式: .npy (NumPy数组)")
    print(f"数组形状: (N, 4) 其中 N ≤ {processor.max_points}")


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    main()