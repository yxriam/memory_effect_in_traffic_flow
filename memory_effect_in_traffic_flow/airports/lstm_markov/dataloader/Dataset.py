from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class AirportDataset(Dataset):
    def __init__(self, file_paths, airport_to_index, pattern, seq_length):
        self.data = []
        self.seq_length = seq_length
        self.airport_to_index = airport_to_index
        self.pattern = pattern
        self._load_data(file_paths)

    def _load_data(self, file_paths):
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in tqdm(lines, desc=f"Processing {file_path}"):
                    airports = self.pattern.findall(line)
                    # print("airports:", airports)
                    airport_indices = [self.airport_to_index[airport] for airport in airports if
                                       airport in self.airport_to_index]
                    # print("airport_indices:", airport_indices)
                    if len(airport_indices) < self.seq_length:
                        continue  # 如果长度不够则丢弃
                    else:
                        for i in range(len(airport_indices) - self.seq_length + 1):
                            self.data.append(airport_indices[i:i + self.seq_length])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)



class OrderSequenceDataset(Dataset):
    def __init__(self, sequences, airport_to_index):
        """
        参数:
        - sequences: List[str]，每个元素是空格分隔的机场三字码序列，例如 'ABC DEF XYZ'
        - airport_to_index: dict，将机场三字码映射为索引
        """
        self.data = []
        self.targets = []

        for seq_str in sequences:
            # print("Processing sequence:", seq_str)
            # tokens = seq_str.strip().split()
            if len(seq_str) < 2:
                continue  # 至少要有一个输入和一个目标
            indexed_seq = [airport_to_index[a] for a in seq_str if a in airport_to_index]
            if len(indexed_seq) == len(seq_str):
                self.data.append(indexed_seq[:-1])   # 前 n-1 作为输入
                self.targets.append(indexed_seq[-1])  # 最后一个作为预测目标

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )

