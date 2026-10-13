"""City-level Markov chains on DB1B itineraries and the return-flow Sankey diagram (paper Fig. 3, top).

Pipeline
  1. map every airport of every itinerary to its city with airport_dict.json
     (airports within 50 km of each other share a city name);
  2. fit a first- and a second-order MarkovChain on the city sequences;
  3. optionally score next-city accuracy on a held-out sequence file;
  4. for every origin city O and the hub city H, draw `--n-rollout` samples of the next city
     from P(next | H) (first order) and P(next | O, H) (second order), keep the samples whose
     destination is one of the origin cities, and render one Sankey per order
     (left column = origin, right column = destination after the hub).

The first-order Sankey ignores where the traveller came from, so the flows out of the hub are
identical for every origin; the second-order Sankey shows the return-to-origin flow.

Usage (from airports/):
    python city_markov_sankey.py --train data/sequences_split/202401/*.txt \
        --hub 芝加哥 --origins 纽约 亚特兰大 旧金山 西雅图 \
        [--test data/sequences_split/202404/202404_3.txt] [--n-rollout 1000] [--out data/sankey]
"""
import argparse
import json
import os
import re
from collections import Counter

from markov_chain import MarkovChain

AIRPORT_CODE = re.compile(r'\b[A-Z]{3}\b')


def resolve_city(name, cities):
    """Accept a full dict value ('芝加哥 Chicago') or any unique part of it ('芝加哥', 'Chicago')."""
    if name in cities:
        return name
    hits = sorted(c for c in cities if name in c)
    if len(hits) != 1:
        raise SystemExit(f"city '{name}' matches {hits or 'nothing'} in airport_dict.json")
    return hits[0]


def load_city_sequences(paths, airport_to_city):
    """Yield one list of city names per itinerary; airports without a city are dropped."""
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                cities = [airport_to_city.get(a) for a in AIRPORT_CODE.findall(line)]
                cities = [c for c in cities if c is not None]
                if len(cities) >= 2:
                    yield cities


def fit_chains(train_paths, airport_to_city, seed):
    chain1, chain2 = MarkovChain(1, seed=seed), MarkovChain(2, seed=seed)
    n_paths = n_airports = 0
    for cities in load_city_sequences(train_paths, airport_to_city):
        n_paths += 1
        n_airports += len(cities)
        chain1.fit_sequence(cities)
        chain2.fit_sequence(cities)
    print(f"fitted on {n_paths} itineraries, mean length {n_airports / max(n_paths, 1):.2f}")
    return chain1, chain2


def held_out_accuracy(test_paths, airport_to_city, chain1, chain2):
    """Arg-max next-city accuracy of both chains on the same prediction positions (i >= 2)."""
    hit1 = hit2 = total = 0
    for cities in load_city_sequences(test_paths, airport_to_city):
        for i in range(2, len(cities)):
            total += 1
            hit1 += chain1.most_likely((cities[i - 1],)) == cities[i]
            hit2 += chain2.most_likely((cities[i - 2], cities[i - 1])) == cities[i]
    if total:
        print(f"held-out next-city accuracy on {total} positions: "
              f"1st order {hit1 / total:.4f}, 2nd order {hit2 / total:.4f}")


def rollout(chain1, chain2, origins, hub, n):
    """Return {order: Counter((origin, destination))} for destinations within `origins`."""
    flows = {1: Counter(), 2: Counter()}
    for origin in origins:
        for _ in range(n):
            d1 = chain1.transition((hub,))
            d2 = chain2.transition((origin, hub))
            if d1 in origins:
                flows[1][(origin, d1)] += 1
            if d2 in origins:
                flows[2][(origin, d2)] += 1
    return flows


def sankey_data(flow):
    """ECharts-style nodes/links; origin nodes get suffix 1, destination nodes suffix 3."""
    links = [{'source': f"{o}1", 'target': f"{d}3", 'value': v} for (o, d), v in sorted(flow.items())]
    names = sorted({l['source'] for l in links} | {l['target'] for l in links})
    return {'nodes': [{'name': n} for n in names], 'links': links}


def render_sankey(data, title, html_path):
    from pyecharts import options as opts
    from pyecharts.charts import Sankey
    sankey = Sankey(init_opts=opts.InitOpts(width="800px", height="800px"))
    sankey.add("", nodes=data['nodes'], links=data['links'],
               linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
               label_opts=opts.LabelOpts(position="right"))
    sankey.set_global_opts(title_opts=opts.TitleOpts(title=title))
    sankey.render(html_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', nargs='+', required=True, help='sequence files (or raw .asc) used to fit the chains')
    ap.add_argument('--test', nargs='*', default=[], help='held-out sequence files for next-city accuracy')
    ap.add_argument('--hub', required=True, help='hub city, e.g. 芝加哥 or Chicago (matched against airport_dict.json values)')
    ap.add_argument('--origins', nargs='+', required=True, help='origin cities shown on the left of the Sankey')
    ap.add_argument('--n-rollout', type=int, default=1000)
    ap.add_argument('--airport-dict', default='airport_dict.json')
    ap.add_argument('--out', default='data/sankey')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(args.airport_dict, 'r', encoding='utf-8') as f:
        airport_to_city = {a: c for a, c in json.load(f).items() if c and c != 'UnKnown'}
    cities = set(airport_to_city.values())
    hub = resolve_city(args.hub, cities)
    origins = [resolve_city(o, cities) for o in args.origins]

    chain1, chain2 = fit_chains(args.train, airport_to_city, args.seed)
    if args.test:
        held_out_accuracy(args.test, airport_to_city, chain1, chain2)

    flows = rollout(chain1, chain2, origins, hub, args.n_rollout)
    os.makedirs(args.out, exist_ok=True)
    for order, flow in flows.items():
        data = sankey_data(flow)
        stem = os.path.join(args.out, f"city_sankey_order{order}")
        with open(stem + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            render_sankey(data, f"{'一阶' if order == 1 else '二阶'}流 via {hub}", stem + '.html')
            print(f"order {order}: {sum(flow.values())} kept samples -> {stem}.json / .html")
        except ImportError:
            print(f"order {order}: {sum(flow.values())} kept samples -> {stem}.json (pip install pyecharts for the HTML)")


if __name__ == "__main__":
    main()
