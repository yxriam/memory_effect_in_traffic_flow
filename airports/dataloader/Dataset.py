import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class AirportDataset(Dataset):
    """Sliding windows of `seq_length` airport indices cut from every path in `file_paths`.

    Each line of a sequence file is one DB1B itinerary written as space-separated IATA codes
    (see airports_sequences_split.py). Codes missing from `airport_to_index` are dropped and
    paths shorter than `seq_length` are skipped.
    """

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
                idx = [self.airport_to_index[a] for a in airports if a in self.airport_to_index]
                if len(idx) < self.seq_length:
                    continue
                for i in range(len(idx) - self.seq_length + 1):
                    self.data.append(idx[i:i + self.seq_length])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)
