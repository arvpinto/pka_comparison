#!/usr/bin/env python3
"""
render_structure.py - picture of a protein with its titratable sites coloured
by the consensus pKa - pH from a pka_compare.py table (PyMOL, headless).

  python render_structure.py 4LZT_clean.pdb 4LZT_pka_table.csv -o 4LZT.png --width 2400 --height 1800 --crop 70
  python render_structure.py 4LZT_clean.pdb 4LZT_pka_table.csv -o card.png --transparent --crop 40

Side chains of the titratable residues are drawn as sticks and their titrating
group as a sphere, blue when the consensus pKa is above the pH (protonated) and
red when it is below (deprotonated), with the colour scale of the heatmap.
Disulfide-bonded cysteines are yellow. Needs pymol-open-source, pandas and
matplotlib (conda install -c conda-forge pymol-open-source pandas matplotlib).
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# atom that carries the titrating proton, per site type
GROUP_ATOM = {"ASP": "CG", "GLU": "CD", "HIS": "CE1", "LYS": "NZ", "ARG": "CZ",
              "TYR": "OH", "CYS": "SG", "NTR": "N", "CTR": "OXT"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb")
    p.add_argument("table", help="<name>_pka_table.csv from pka_compare.py")
    p.add_argument("-o", "--out", default="structure.png")
    p.add_argument("--ph", type=float, default=7.0)
    p.add_argument("--clip", type=float, default=4.0, help="colour scale limit for |pKa - pH|")
    p.add_argument("--width", type=int, default=1400)
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--view", default=None, help="PyMOL view (18 numbers, from get_view)")
    p.add_argument("--turn", default="", help="rotations before rendering, e.g. 'y:30,x:-20'")
    p.add_argument("--cartoon-color", default="0x8d96a8")
    p.add_argument("--transparent", action="store_true", help="transparent background")
    p.add_argument("--crop", type=int, default=None, metavar="PAD",
                   help="trim the empty background, leaving PAD pixels around the molecule")
    args = p.parse_args()

    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd

    cmap = LinearSegmentedColormap.from_list(
        "pka_div", ["#a32a2a", "#e34948", "#f0efec", "#2a78d6", "#104281"])
    norm = TwoSlopeNorm(vmin=-args.clip, vcenter=0.0, vmax=args.clip)

    cmd.load(args.pdb, "prot")
    cmd.remove("hydro or solvent")
    cmd.hide("everything")
    cmd.set("cartoon_transparency", 0.15)
    cmd.show("cartoon")
    cmd.color(args.cartoon_color, "prot")
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)

    tab = pd.read_csv(args.table)
    for _, r in tab.iterrows():
        site, chain, resi = r["site"], r["chain"], int(r["resnum"])
        res = f"prot and chain {chain} and resi {resi}"
        atom = GROUP_ATOM[site]
        if r.get("state_consensus") == "disulfide":
            cmd.show("sticks", f"{res} and sidechain")
            cmd.color("0xe8c547", f"{res} and sidechain")
            continue
        pka = r["pKa_consensus"]
        if not np.isfinite(pka):
            continue
        rgb = cmap(norm(np.clip(pka - args.ph, -args.clip, args.clip)))[:3]
        name = f"c_{chain}{resi}{site}"
        cmd.set_color(name, list(rgb))
        if site not in ("NTR", "CTR"):
            cmd.show("sticks", f"{res} and (sidechain or name CA)")
            cmd.color(name, f"{res} and sidechain and not name CA")
        sel = f"{res} and name {atom}" if site != "CTR" else f"{res} and name OXT+C"
        cmd.show("spheres", sel)
        cmd.set("sphere_scale", 0.75 if site != "CTR" else 0.5, sel)
        cmd.color(name, sel)
    cmd.set("stick_radius", 0.22)
    cmd.set("stick_ball", 0)

    cmd.orient("prot")
    for t in filter(None, args.turn.split(",")):
        ax, ang = t.split(":")
        cmd.turn(ax.strip(), float(ang))
    if args.view:
        cmd.set_view([float(x) for x in args.view.replace(",", " ").split()])
    cmd.zoom("prot", 1.0, complete=1)

    cmd.bg_color("black")
    cmd.set("ray_opaque_background", 0 if args.transparent else 1)
    cmd.set("ray_shadows", 0)
    cmd.set("ambient", 0.35)
    cmd.set("specular", 0.25)
    cmd.set("antialias", 2)
    cmd.set("depth_cue", 1)
    cmd.set("fog_start", 0.55)
    cmd.ray(args.width, args.height)
    cmd.png(args.out, dpi=300)
    if args.crop is not None:
        crop(args.out, args.crop, args.transparent)
    print("written", args.out)


def crop(path, pad, transparent):
    """Trim the empty background around the molecule, keeping pad px on each side."""
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    a = np.asarray(img)
    mask = a[..., 3] > 8 if transparent else a[..., :3].max(axis=2) > 8
    ys, xs = np.nonzero(mask)
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad + 1, img.width), min(ys.max() + pad + 1, img.height))
    img.crop(box).save(path)


if __name__ == "__main__":
    main()
