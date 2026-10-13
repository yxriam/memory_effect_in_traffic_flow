import re
from collections import defaultdict
import pandas as pd
from collections import defaultdict

# 假设predict1和predict2是两个列表，每个列表中的元素都是字符串形式的三元组，例如："(A, B, C)"
predict1 = []
predict2 = []

# 读取predict1.txt文件
with open('../data/predict1.txt', 'r') as f:
    predict1 = [line.strip() for line in f.readlines()]

# 读取predict2.txt文件
with open('../data/predict2.txt', 'r') as f:
    predict2 = [line.strip() for line in f.readlines()]

# print(predict1)
# print(predict2)

def pack_into_triplets(sequence):
    triplets = []
    # 确保序列长度至少为3
    if len(sequence) >= 3:
        for i in range(len(sequence) - 2):
            # 创建三元组 (sequence[i], sequence[i+1], sequence[i+1])
            triplet = (sequence[i], sequence[i+1], sequence[i+2])
            triplets.append(triplet)
    return triplets

triplets1 = pack_into_triplets(predict1)
triplets2 = pack_into_triplets(predict2)

# 初始化计数器和字典
triplet_counts = defaultdict(int)
triplet_dict = defaultdict(list)  # 使用列表来存储三元组
triplet_dict1 = defaultdict(list)  # 使用列表来存储三元组
triplet_dict2 = defaultdict(list)  # 使用列表来存储三元组


# 合并两个列表并统计三元组
for triplet in triplets1 + triplets2:
    # 提取第二个元素作为键
    second_element = triplet[1]
    triplet_counts[second_element] += 1
    triplet_dict[second_element].append(triplet)  # 将三元组添加到列表中
    if triplet in triplets1:
        triplet_dict1[second_element].append(triplet)
    else:
        triplet_dict2[second_element].append(triplet)


# 现在 triplet_counts 包含了每个第二个元素的出现次数
# triplet_dict 包含了每个第二个元素对应的所有三元组


# 找出出现次数最多的三元组
most_common_triplets = sorted(triplet_counts.items(), key=lambda x: x[1], reverse=True)
# print('出现次数最多的三元组:',most_common_triplets)
triples_list1 = triplet_dict1[most_common_triplets[0][0]]
triples_list2 = triplet_dict2[most_common_triplets[0][0]]
# print('triples_list1:',triples_list1)
# print('triples_list2:',triples_list2)

#excel输出原始3元组
counts_order1 = defaultdict(int)
counts_order2 = defaultdict(int)
#修改这里，选择是一阶还是二阶
for first, second, third in triples_list1:
    counts_order1[(first, second,third)] += 1
for first, second, third in triples_list2:
    counts_order2[(first, second,third)] += 1

keys_order1=counts_order1.keys()
keys_order2=counts_order2.keys()
string_keys_order1=[str(key_order1) for key_order1 in keys_order1 ]
string_keys_order2=[str(key_order2) for key_order2 in keys_order2 ]

cleaned_strings_order1 = [item.replace("'", "").replace("(", "").replace(")", "") for item in string_keys_order1]
split_strings_order1 = [item.split(",") for item in cleaned_strings_order1]

cleaned_strings_order2 = [item.replace("'", "").replace("(", "").replace(")", "") for item in string_keys_order2]
split_strings_order2 = [item.split(",") for item in cleaned_strings_order2]

# print(split_strings_order1)
# print(split_strings_order2)

# 创建 DataFrame
data_order1 = {'First_Element': [item[0] for item in split_strings_order1], 'Second_Element': [item[1] for item in split_strings_order1],'Third_Element': [item[2] for item in split_strings_order1], 'Count': list(counts_order1.values())}
data_order2 = {'First_Element': [item[0] for item in split_strings_order2], 'Second_Element': [item[1] for item in split_strings_order2],'Third_Element': [item[2] for item in split_strings_order2], 'Count': list(counts_order2.values())}
print(data_order1)
print(data_order2)
df_order1 = pd.DataFrame(data_order1)
df_order2 = pd.DataFrame(data_order2)

#修改这里，存储1阶还是2阶
excel_path_order1 = '../data/Order1_element_counts.xlsx'
df_order1.to_excel(excel_path_order1, index=False)
excel_path_order2 = '../data/Order2_element_counts.xlsx'
df_order2.to_excel(excel_path_order2, index=False)



