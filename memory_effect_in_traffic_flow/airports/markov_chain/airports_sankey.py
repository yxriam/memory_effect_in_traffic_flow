import pandas as pd
from pyecharts.charts import Sankey
from pyecharts import options as opts
from collections import Counter
import chardet

#解决html无法渲染
from pyecharts.globals import CurrentConfig
CurrentConfig.ONLINE_HOST = "http://127.0.0.1:8000/assets/"

file_name='../data/predictions_first_order_city_filter.txt'

with open(file_name, 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

# 读取predictions_second_order.txt文件
with open(file_name, 'r',encoding=encoding) as f:
    lines = [line.strip() for line in f.readlines()]

# 提取第一个机场名称和第三个机场名称
source_airports = []
target_airports = []

for line in lines:
    airports = line.split()
    if len(airports) == 3:
        source_airports.append(f"{airports[0]}1")
        target_airports.append(f"{airports[2]}3")

# print(source_airports)
# print(target_airports)

# 创建DataFrame
df = pd.DataFrame({'source': source_airports, 'target': target_airports})

# 统计每一组source和target出现的次数
counter = Counter(zip(source_airports, target_airports))
df['value'] = df.apply(lambda row: counter[(row['source'], row['target'])], axis=1)

# 生成桑基图所需的节点和链接
all_nodes = list(set(source_airports + target_airports))
nodes = [{'name': node} for node in all_nodes]
# links = [{'source': row['source'], 'target': row['target'], 'value': row['value']} for _, row in df.iterrows()]
links = [
    {
        'source': row['source'],
        'target': row['target'],
        'value': row['value'],
        # 'lineStyle': opts.LineStyleOpts(color='red') if row['source'][:-1] == row['target'][:-1] else opts.LineStyleOpts()
        'lineStyle': opts.LineStyleOpts()
    }
    for _, row in df.iterrows()
]

# 生成桑基图
sankey = Sankey(init_opts=opts.InitOpts(width="800px", height="800px", theme="westeros"))

sankey.add(
    "",
    nodes=nodes,
    links=links,
    linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
    label_opts=opts.LabelOpts(position="right"),
).set_global_opts(title_opts=opts.TitleOpts(title="一阶流"))

sankey.render("../doc/airports/citys_sankey1.html")