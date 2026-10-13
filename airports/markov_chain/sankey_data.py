import pandas as pd
from collections import Counter
import chardet
import json

file_name=r'../data\prediction\predictions_second_order_city_filter.txt'

# 检测文件编码
with open(file_name, 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

# 读取文件内容
with open(file_name, 'r', encoding=encoding) as f:
    lines = [line.strip() for line in f.readlines()]

# 提取第一个机场名称和第三个机场名称
source_airports = []
target_airports = []

for line in lines:
    airports = line.split()
    if len(airports) == 3:
        source_airports.append(f"{airports[0]}1")
        target_airports.append(f"{airports[2]}3")

# 创建DataFrame并统计次数
df = pd.DataFrame({'source': source_airports, 'target': target_airports})
counter = Counter(zip(source_airports, target_airports))
df['value'] = df.apply(lambda row: counter[(row['source'], row['target'])], axis=1)

# 去重处理
df = df.drop_duplicates()

# 生成桑基图所需的节点和链接（ECharts格式）
all_nodes = list(set(source_airports + target_airports))
nodes = [{'name': node} for node in all_nodes]
links = [
    {
        'source': row['source'],
        'target': row['target'],
        'value': row['value']
    }
    for _, row in df.iterrows()
]

# 整合为ECharts所需的格式
echarts_data = {
    'nodes': nodes,
    'links': links
}

# 输出JSON格式的数据
print(json.dumps(echarts_data, ensure_ascii=False, indent=2))

# 也可以保存到文件
with open('sankey_data.json', 'w', encoding='utf-8') as f:
    json.dump(echarts_data, f, ensure_ascii=False, indent=2)
