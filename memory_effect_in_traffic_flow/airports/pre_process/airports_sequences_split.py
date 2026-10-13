import re
import os
from tqdm import tqdm


def split_airport_sequences(file_paths):
    """
    对多个输入文件内容，按每行的机场数量分割机场代码并保存到对应年月文件夹下的文件中
    :param file_paths: 输入文件路径列表
    """
    for input_file_path in file_paths:
        # 获取文件中的年月信息用于创建文件夹
        year_month = input_file_path.split('.')[2]
        # 构建输出目录路径
        output_dir = os.path.join(os.path.dirname(r'../data\sequences_split'),"sequences_split", year_month)
        # 如果输出目录不存在，则创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 定义正则表达式模式来匹配三个大写字母的机场代码
        airport_code_pattern = re.compile(r'[A-Z]{3}')
        # 统计文件总行数用于进度条
        total_lines = sum(1 for _ in open(input_file_path, 'r'))
        # 打开输入文件进行读取
        with open(input_file_path, 'r') as file:
            # 创建进度条对象
            pbar = tqdm(total=total_lines, desc=f"Processing {input_file_path}")
            # 逐行读取文件内容
            for line in file:
                # 查找行中的所有机场代码
                airport_codes = airport_code_pattern.findall(line)
                # 计算机场代码的数量
                num_airports = len(airport_codes)
                # 构建输出文件名
                file_name = f"{year_month}_{num_airports}.txt"
                # 构建输出文件路径
                output_file_path = os.path.join(output_dir, file_name)
                # 打开输出文件进行写入
                with open(output_file_path, 'a') as output_file:
                    # 将机场代码写入输出文件
                    output_file.write(' '.join(airport_codes) + '\n')
                # 更新进度条
                pbar.update(1)
            # 关闭进度条
            pbar.close()


if __name__ == "__main__":
    file_paths = [
        # r'../data\original_data\db1b.public.201103.asc',
        # r'../data\original_data\db1b.public.201106.asc',
        # r'../data\original_data\db1b.public.201109.asc',
        # r'../data\original_data\db1b.public.201112.asc',
        # r'../data\original_data\db1b.public.201203.asc',
        r'../data\original_data\db1b.public.202412.asc',
        # 可以继续添加更多文件路径

    ]
    split_airport_sequences(file_paths)