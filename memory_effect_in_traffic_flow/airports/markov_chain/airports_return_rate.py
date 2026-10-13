from tqdm import tqdm


def read_sequences(file_path, num_lines):
    sequences = []
    with tqdm(total=num_lines, desc="Reading sequences") as pbar:
        with open(file_path, 'r') as file:
            lines_read = 0
            while lines_read < num_lines:
                line = next(file).strip()
                pbar.update(1)
                lines_read += 1
                airports = line.split()
                sequences.append(airports)
    return sequences


def calculate_return_rates(sequences):
    two_step_returns = 0
    three_step_returns = 0

    for i in range(len(sequences) - 2):
        if sequences[i] == sequences[i + 2]:
            two_step_returns += 1
        if i < len(sequences) - 3 and sequences[i] == sequences[i + 3]:
            three_step_returns += 1

    return two_step_returns, three_step_returns



file_path_order1 = '../data/predict1.txt'
file_path_order2 = '../data/predict2.txt'
num_lines = 10000

sequences1 = read_sequences(file_path_order1, num_lines)
two_step_returns1, three_step_returns1 = calculate_return_rates(sequences1)

sequences2 = read_sequences(file_path_order2, num_lines)
two_step_returns2, three_step_returns2 = calculate_return_rates(sequences2)

print("Order1 Two-step return rates:", two_step_returns1)
print("Order1 Three-step return rates:", three_step_returns1)
print("Order2 Two-step return rates:", two_step_returns2)
print("Order2 Three-step return rates:", three_step_returns2)
