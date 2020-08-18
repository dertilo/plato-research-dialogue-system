from collections import defaultdict
from pprint import pprint

from util import data_io

if __name__ == '__main__':

    jsonl = "plato_results/results.jsonl"
    stats = defaultdict(dict)
    for d in data_io.read_jsonl(jsonl):
        s2ac = d["scores"]["train"]["state_counts"]
        sc = {s:sum([v for v in ac.values()]) for s,ac in s2ac.items()}
        stats["num_states"][d["name"]]=len(sc.keys())
        stats["most_common"][d["name"]] = max(sc.items(),key=lambda x:x[1])

    pprint(stats)