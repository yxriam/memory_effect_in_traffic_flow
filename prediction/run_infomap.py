"""Chapter 6: run the official Infomap on the exported networks and compare with the in-house optimiser.
Install once:  pip install infomap
Run:           python run_infomap.py            (reads infomap_input/*.net, writes results/infomap/infomap_summary.csv)
Each network is run two-level, directed, with recorded link flows (no teleportation to unrecorded links is
needed because flows are empirical), 20 trials, seed 123."""
import glob, os, csv
import infomap
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'infomap')
os.makedirs(OUT, exist_ok=True); rows = []
for path in sorted(glob.glob(os.path.join(HERE, 'infomap_input', '*.net'))):
    tag = os.path.basename(path)[:-4]
    im = infomap.Infomap(f'--two-level --directed --flow-model rawdir --num-trials 20 --seed 123 --silent')
    im.read_file(path)
    im.run()
    # state id -> physical id from the *States section (first-order files: state = physical)
    st2ph, sect = {}, None
    for line in open(path):
        if line.startswith('*'): sect = line.split()[0].lower(); continue
        if sect == '*states':
            a, b = line.split()[:2]; st2ph[int(a)] = int(b)
    is_mem = bool(st2ph)
    mods = im.get_modules(states=True) if is_mem else im.get_modules()
    phys_mod = {}
    for nid, m in mods.items():
        phys_mod.setdefault(st2ph.get(nid, nid), set()).add(m)
    multi = sum(1 for s in phys_mod.values() if len(s) > 1)
    rows.append(dict(network=tag, codelength=im.codelength, one_level_codelength=im.one_level_codelength,
                     modules=im.num_top_modules, physical_nodes=len(phys_mod), physical_in_multiple_modules=multi))
    im.write_clu(os.path.join(OUT, f'{tag}.clu')); print(rows[-1], flush=True)
with open(os.path.join(OUT, 'infomap_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
