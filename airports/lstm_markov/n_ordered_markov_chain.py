import os
import pickle
import re
from collections import defaultdict
import random


class MarkovChain:
    def __init__(self, order):
        self.order = order
        self.transitions = {}
        self.state_counts = defaultdict(int)  # 用于存储每个状态出现的次数
        self.alpha = 0.1

    def add_transition(self, *states):
        if len(states) != self.order + 1:
            raise ValueError(f"Expected {self.order + 1} states, got {len(states)}")

        for state in states:
            self.state_counts[state] += 1

        key = states[:-1]
        next_state = states[-1]

        if key not in self.transitions:
            self.transitions[key] = {}
        if next_state not in self.transitions[key]:
            self.transitions[key][next_state] = 0
        self.transitions[key][next_state] += 1

    def get_probabilities(self):
        probabilities = {}
        for key, transitions in self.transitions.items():
            total = sum(transitions.values())
            probabilities[key] = {state: count / total for state, count in transitions.items()}
        return probabilities

    def transition(self, current_states):
        if len(current_states) != self.order:
            raise ValueError(f"Expected {self.order} states, got {len(current_states)}")

        key = tuple(current_states)
        probabilities = self.get_probabilities().get(key, {})

        if random.random() < self.alpha or not probabilities:
            return random.choices(list(self.state_counts), weights=list(self.state_counts.values()))[0]
        return random.choices(list(probabilities.keys()), weights=list(probabilities.values()))[0]

    def predict(self, num_predictions):
        predictions = []
        current_states = self._get_most_frequent_states()
        for _ in range(num_predictions):
            next_state = self.transition(current_states)
            predictions.append(next_state)
            current_states = current_states[1:] + (next_state,)
        return predictions

    def calculate_accuracy(self, test_data):
        correct_predictions = 0
        total_predictions = 0
        for sequence in test_data:
            if len(sequence) > self.order:
                predicted_next_state = self.transition(sequence[-self.order:])
                if predicted_next_state == sequence[-1]:
                    correct_predictions += 1
                total_predictions += 1
        return correct_predictions / total_predictions if total_predictions > 0 else 0

    def _get_most_frequent_states(self):
        """
        辅助函数，用于找到出现次数最多的状态序列。
        """
        max_count = 0
        most_frequent_states = None
        for key in self.transitions:
            count = sum(self.transitions[key].values())
            if count > max_count:
                max_count = count
                most_frequent_states = key
        return most_frequent_states


def create_markov_chain(order):
    """
    工厂函数，根据指定的阶数创建相应的马尔可夫链类。
    """

    class DynamicMarkovChain(MarkovChain):
        def __init__(self):
            super().__init__(order)

    return DynamicMarkovChain


def example():
    for x in range(2, 13):
        order = x-1
        file_path = [
            f'../data/sequences_split/202409/202409_{x}.txt',
            # 添加更多文件路径
        ]
        model_output_dir = f'./output/markov'
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        MarkovChainClass = create_markov_chain(order)
        markov_chain = MarkovChainClass()
        for file_path in file_path:
            with open(file_path, 'r') as file:
                for line in file:
                    airports = re.findall(r'\b[A-Z]{3}\b', line)
                    markov_chain.add_transition(*airports)
        with open(os.path.join(model_output_dir, f'markov{order}.pkl'), 'wb') as f:
            pickle.dump({
                'order': markov_chain.order,
                'transitions': markov_chain.transitions,
                'state_counts': markov_chain.state_counts,
            }, f)



def example3():
    # 使用示例
    order = 3  # 指定阶数
    MarkovChainClass = create_markov_chain(order)
    markov_chain = MarkovChainClass()

    # 添加一些转换
    markov_chain.add_transition('A', 'B', 'C', 'E')
    markov_chain.add_transition('B', 'C', 'D', 'E')
    markov_chain.add_transition('C', 'D', 'E', 'E')

    # 预测
    print(markov_chain.predict(5))


if __name__ == "__main__":
    example()
