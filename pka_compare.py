#!/usr/bin/env python3
"""
pka_compare.py - comparative protein pKa prediction pipeline.

Runs several pKa predictors on the same (cleaned) protein structure, merges
their results into a single per-residue table for every residue with a
variable protonation state, and assigns protonation states at a target pH
(per method and by consensus), with force-field residue names ready for
system preparation.

Predictors
----------
  propka   PROPKA 3 (empirical)                       - run natively (pip: propka)
  pypka    PypKa (Poisson-Boltzmann + Monte Carlo)    - run natively (pip: pypka)
  delphipka DelPhiPKa (Gaussian-smoothed PB, DelPhi)  - runs the delphiPKa executable
  pkai+    pKAI+ (deep learning, biased to exp. data) - run natively (pip: pKAI)
  pkai     pKAI  (deep learning, PypKa surrogate)     - run natively (pip: pKAI)
  deepka   DeepKa (3D CNN trained on CpHMD pKa)       - imported CSV or external command
  mepka    ME-pKa (multimodal: 3D grid + ESM-2)       - imported CSV or external command

DelPhiPKa is compiled from source (github.com/delphi001/DelPhiPKa); the
pipeline writes its run.prm, runs it (through mpirun when more than one CPU is
requested) and reads pKa.csv and titra.csv. create_pka_env.sh builds it.

DeepKa and ME-pKa have no installable package: their reference code needs
CHARMM, UCSF Chimera and OpenBabel 2 for featurisation and uses hard-coded
paths. They enter the pipeline either from a results file (--deepka-csv,
--mepka-csv; e.g. the DeepKa web server output or the 'Predict pKa' CSV the
repositories write) or through a command template (--deepka-cmd,
--mepka-cmd) that you point to your local installation.

Outputs (prefix = <outdir>/<name>)
-------
  <name>_pka_table.csv      wide per-residue table (one row per titratable site)
  <name>_pka_table.xlsx     same, if openpyxl is available
  <name>_pka_long.csv       long format (one row per site per method)
  <name>_summary.txt        readable summary: flagged sites, net charges, method status
  <name>_heatmap[_partN].png  pKa - pH heatmap (600 dpi by default)
  <name>_<ff>.pdb           optional, cleaned structure with protonation-state
                            residue names (--write-pdb amber|charmm)
  <name>_clean.pdb          cleaned structure every method was run on
  logs/ and per-method work directories

Example
-------
  python pka_compare.py -i 4LZT.pdb --ph 7.0 -o lyso_pka \
      --methods propka,pypka,pkai+,delphipka,deepka \
      --deepka-csv deepka_server_results.csv --write-pdb amber
"""

import argparse
import contextlib
import datetime
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd

__version__ = "1.1.0"

# --------------------------------------------------------------------------- #
# Chemistry constants
# --------------------------------------------------------------------------- #

# Sites handled by the pipeline. NTR/CTR are the chain termini.
ALL_SITES = ["ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG", "NTR", "CTR"]
DEFAULT_SITES = ["ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG", "NTR", "CTR"]

# Acids are charged (-1) when deprotonated; bases are charged (+1) when protonated.
ACIDIC = {"ASP", "GLU", "CYS", "TYR", "CTR"}
BASIC = {"HIS", "LYS", "ARG", "NTR"}

# Reference ("null model") pKa values of the isolated groups.
# Thurlkill et al., Protein Sci. 2006, 15, 1214 (Ala-based pentapeptides),
# Arg from Fitch et al., Biophys. J. 2015, 109, 1016.
MODEL_PKA = {
    "ASP": 3.67, "GLU": 4.25, "HIS": 6.54, "CYS": 8.55, "TYR": 9.84,
    "LYS": 10.40, "ARG": 13.80, "NTR": 8.00, "CTR": 3.67,
}

# Residue-name normalisation for the names different programs and force fields use.
RESNAME_ALIASES = {
    "ASP": "ASP", "ASH": "ASP", "ASPP": "ASP", "ASPH": "ASP", "AS4": "ASP",
    "GLU": "GLU", "GLH": "GLU", "GLUP": "GLU", "GLUH": "GLU", "GL4": "GLU",
    "HIS": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS",
    "HSP": "HIS", "HISA": "HIS", "HISB": "HIS", "HISD": "HIS", "HISE": "HIS",
    "HISH": "HIS", "HIP+": "HIS",
    "CYS": "CYS", "CYM": "CYS", "CYX": "CYS", "CYS2": "CYS",
    "TYR": "TYR", "TYM": "TYR",
    "LYS": "LYS", "LYN": "LYS", "LSN": "LYS", "LYSH": "LYS",
    "ARG": "ARG", "ARN": "ARG",
    "NTR": "NTR", "NTERM": "NTR", "NTER": "NTR", "N+": "NTR", "NT": "NTR", "N-TERM": "NTR",
    "CTR": "CTR", "CTERM": "CTR", "CTER": "CTR", "C-": "CTR", "CT": "CTR", "C-TERM": "CTR",
}

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # protonation-state variants that may already be present in an input file
    "ASH", "GLH", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "LYN", "CYM", "CYX",
    "ASPP", "GLUP", "LSN", "HISA", "HISB", "HISD", "HISE", "HISH",
}
TO_STANDARD = {k: v for k, v in RESNAME_ALIASES.items() if v not in ("NTR", "CTR")}
METALS = {"ZN", "MG", "CA", "MN", "FE", "FE2", "FE3", "CU", "CU1", "CU2", "NI", "CO", "CD", "HG",
          "NA", "K", "MO", "W", "V"}
WATERS = {"HOH", "WAT", "H2O", "TIP3", "TIP", "SOL", "DOD", "T3P", "SPC"}

METHOD_LABELS = OrderedDict([
    ("propka", "PROPKA3"),
    ("pypka", "PypKa"),
    ("delphipka", "DelPhiPKa"),
    ("pkai+", "pKAI+"),
    ("pkai", "pKAI"),
    ("deepka", "DeepKa"),
    ("mepka", "ME-pKa"),
])
NATIVE_METHODS = ["propka", "pypka", "delphipka", "pkai+", "pkai"]

# Site types each method predicts (used for coverage notes and flags).
METHOD_SITES = {
    "propka": set(ALL_SITES),
    "pypka": {"ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "NTR", "CTR"},
    "delphipka": {"ASP", "GLU", "HIS", "LYS", "ARG"},   # + TYR, CYS with --delphipka-more-residues
    "pkai+": {"ASP", "GLU", "HIS", "CYS", "TYR", "LYS"},
    "pkai": {"ASP", "GLU", "HIS", "CYS", "TYR", "LYS"},
    "deepka": {"ASP", "GLU", "HIS", "LYS"},
    "mepka": {"ASP", "GLU", "HIS", "LYS"},
}
EXTERNAL_METHODS = ["deepka", "mepka"]

# Residue names for each site / state, per force-field family.
#   state P = protonated, D = deprotonated. HIS neutral tautomer resolved later.
FF_NAMES = {
    "amber": {
        "ASP": {"P": "ASH", "D": "ASP"},
        "GLU": {"P": "GLH", "D": "GLU"},
        "HIS": {"P": "HIP", "D": "HIE", "D_HID": "HID", "D_HIE": "HIE"},
        "LYS": {"P": "LYS", "D": "LYN"},
        "CYS": {"P": "CYS", "D": "CYM", "SS": "CYX"},
        "TYR": {"P": "TYR", "D": "TYR(-)*"},
        "ARG": {"P": "ARG", "D": "ARG(0)*"},
        "NTR": {"P": "NH3+", "D": "NH2*"},
        "CTR": {"P": "COOH*", "D": "COO-"},
    },
    "charmm": {
        "ASP": {"P": "ASPP", "D": "ASP"},
        "GLU": {"P": "GLUP", "D": "GLU"},
        "HIS": {"P": "HSP", "D": "HSE", "D_HID": "HSD", "D_HIE": "HSE"},
        "LYS": {"P": "LYS", "D": "LSN"},
        "CYS": {"P": "CYS", "D": "CYM", "SS": "CYS(DISU)"},
        "TYR": {"P": "TYR", "D": "TYR(-)*"},
        "ARG": {"P": "ARG", "D": "ARG(0)*"},
        "NTR": {"P": "NTER", "D": "NNEU*"},
        "CTR": {"P": "CNEU*", "D": "CTER"},
    },
}

LOG = logging.getLogger("pka_compare")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def working_directory(path):
    """Temporarily change the working directory (several predictors write to cwd)."""
    old = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def norm_site(resname):
    """Map any residue/group name a program reports to one of ALL_SITES (or None)."""
    if resname is None:
        return None
    r = str(resname).strip().upper()
    return RESNAME_ALIASES.get(r)


def frac_protonated(pka, ph):
    """Henderson-Hasselbalch fraction of the protonated form."""
    if pka is None or not np.isfinite(pka):
        return np.nan
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def charge_from_fprot(site, f):
    if not np.isfinite(f):
        return np.nan
    return -(1.0 - f) if site in ACIDIC else f


def parse_resnum(value):
    """Split '52', '52A', 'A52' style residue identifiers into (int, icode)."""
    s = str(value).strip()
    m = re.match(r"^(-?\d+)([A-Za-z]?)$", s)
    if m:
        return int(m.group(1)), m.group(2).upper()
    m = re.search(r"(-?\d+)([A-Za-z]?)$", s)
    if m:
        return int(m.group(1)), m.group(2).upper()
    f = float(s)
    return int(round(f)), ""


def site_label(chain, resnum, icode, site, resname=None):
    name = resname if site not in ("NTR", "CTR") else site
    return f"{chain}:{name}{resnum}{icode}"


# --------------------------------------------------------------------------- #
# Structure handling
# --------------------------------------------------------------------------- #

class Structure:
    """Minimal PDB representation: ordered residues per chain + atom lines."""

    def __init__(self):
        self.atom_lines = []          # cleaned ATOM/HETATM lines, PDB format
        self.chains = OrderedDict()   # chain -> list of (resnum, icode, resname)
        self.coords = {}              # (chain, resnum, icode) -> {atomname: xyz}
        self.disulfide = set()        # (chain, resnum, icode) of bonded CYS
        self.metals = []              # (resname, xyz) of metal ions in the input
        self.warnings = []

    def residue_name(self, chain, resnum, icode):
        for rn, ic, name in self.chains.get(chain, []):
            if rn == resnum and ic == icode:
                return name
        return None

    def titratable_sites(self, wanted):
        """All sites present in the structure, ordered by chain and sequence."""
        out = []
        for chain, residues in self.chains.items():
            prot = [r for r in residues if r[2] in STANDARD_AA]
            if not prot:
                continue
            first, last = prot[0], prot[-1]
            if "NTR" in wanted:
                out.append((chain, first[0], first[1], "NTR", first[2]))
            for rn, ic, name in prot:
                site = TO_STANDARD.get(name)
                if site in wanted and site in ALL_SITES:
                    out.append((chain, rn, ic, site, name))
            if "CTR" in wanted:
                out.append((chain, last[0], last[1], "CTR", last[2]))
        return out


def clean_pdb(in_path, out_path, keep_h=False, keep_hetatm=False, altloc="A",
              blank_chain="A", ss_cutoff=2.5):
    """
    Produce a single-model, single-altloc, water-free PDB that all predictors
    read identically. Returns a Structure.
    """
    st = Structure()
    seen_model = False
    # pass 1: collect atom records of the first model, keyed per atom
    atoms = OrderedDict()   # (chain, resnum, icode, resname, atomname) -> list of (alt, line)
    n_alt = 0
    with open(in_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "MODEL":
                if seen_model:
                    st.warnings.append("Multiple MODEL records: only the first model is used.")
                    break
                seen_model = True
                continue
            if rec == "ENDMDL":
                break
            if rec not in ("ATOM", "HETATM"):
                continue
            line = line.rstrip("\n").ljust(80)
            resname = line[17:21].strip()
            if resname in WATERS:
                continue
            is_protein = resname in STANDARD_AA
            if not is_protein and (resname in METALS or line[76:78].strip().upper() in METALS):
                try:
                    st.metals.append((resname, (float(line[30:38]), float(line[38:46]),
                                                float(line[46:54]))))
                except ValueError:
                    pass
            if rec == "HETATM" and not is_protein and not keep_hetatm:
                continue
            atomname = line[12:16].strip()
            element = line[76:78].strip().upper() or re.sub(r"[^A-Za-z]", "", atomname)[:1].upper()
            if element in ("H", "D") and not keep_h:
                continue
            chain = line[21].strip() or blank_chain
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip().upper()
            alt = line[16].strip()
            n_alt += bool(alt)
            atoms.setdefault((chain, resnum, icode, resname, atomname), []).append((alt, line))
    if n_alt:
        st.warnings.append(f"Alternate locations present: kept altloc '{altloc}' "
                           f"(first conformer where '{altloc}' is absent).")
    # pass 2: one conformer per atom, renumber, write
    lines_out = []
    serial = 0
    last_chain = None
    for (chain, resnum, icode, resname, atomname), confs in atoms.items():
        alts = [a for a, _ in confs]
        pick = confs[0][1]
        if altloc != "first" and altloc in alts:
            pick = confs[alts.index(altloc)][1]
        is_protein = resname in STANDARD_AA
        serial += 1
        if last_chain is not None and chain != last_chain:
            lines_out.append("TER")
        last_chain = chain
        new = ("ATOM  " if is_protein else "HETATM") + f"{serial % 100000:5d}" + pick[11:16] + " " + \
            pick[17:21] + chain + pick[22:]
        lines_out.append(new.rstrip())
        if is_protein:
            res_list = st.chains.setdefault(chain, [])
            if not res_list or (res_list[-1][0], res_list[-1][1]) != (resnum, icode):
                res_list.append((resnum, icode, resname))
            try:
                xyz = (float(pick[30:38]), float(pick[38:46]), float(pick[46:54]))
                st.coords.setdefault((chain, resnum, icode), {})[atomname] = xyz
            except ValueError:
                pass
    lines_out.append("TER")
    lines_out.append("END")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines_out) + "\n")
    st.atom_lines = lines_out

    # disulfides
    sg = [(k, v["SG"]) for k, v in st.coords.items()
          if "SG" in v and TO_STANDARD.get(st.residue_name(*k) or "") == "CYS"]
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            d = math.dist(sg[i][1], sg[j][1])
            if d <= ss_cutoff:
                st.disulfide.add(sg[i][0])
                st.disulfide.add(sg[j][0])
    return st


# --------------------------------------------------------------------------- #
# Predictor runners. Each returns a list of record dicts:
#   {chain, resnum, icode, site, pka, ...extra columns}
# --------------------------------------------------------------------------- #

def run_propka(pdb, workdir, args, st):
    import propka.run as prun
    optargs = ["--pH", str(args.ph), "--quiet"]
    if args.propka_args:
        optargs += shlex.split(args.propka_args)
    with working_directory(workdir):
        mol = prun.single(os.path.abspath(pdb), optargs=optargs, write_pka=True)
    conf = mol.conformations.get("AVR") or list(mol.conformations.values())[-1]
    records = []
    for g in conf.get_titratable_groups():
        rtype = g.residue_type.strip().upper()
        if rtype in ("N+",):
            site = "NTR"
        elif rtype in ("C-",):
            site = "CTR"
        else:
            site = norm_site(rtype)
            if site is None or g.atom.res_name.strip().upper() not in TO_STANDARD:
                continue  # ligand groups
        if site == "CYS" and getattr(g, "exclude_cys_from_results", False):
            continue
        icode = (getattr(g.atom, "icode", "") or "").strip().upper()
        records.append({
            "chain": g.atom.chain_id.strip() or args.blank_chain,
            "resnum": int(g.atom.res_num), "icode": icode, "site": site,
            "pka": float(g.pka_value),
            "propka_buried_pct": round(100.0 * float(g.buried), 0),
            "propka_coupled": bool(getattr(g, "coupled_titrating_group", False)),
        })
    return records


def _interp_curve(curve, ph):
    """Interpolate a PypKa titration curve {pH: <protonation>} at ph."""
    xs = np.array(sorted(curve.keys()), dtype=float)
    ys = np.array([curve[x] for x in sorted(curve.keys())], dtype=float)
    if ph < xs[0] or ph > xs[-1]:
        return np.nan
    return float(np.interp(ph, xs, ys))


def run_pypka(pdb, workdir, args, st):
    from pypka import Titration
    params = {
        "structure": os.path.abspath(pdb),
        "ncpus": args.ncpus,
        "epsin": args.pypka_epsin,
        "ionicstr": args.pypka_ionicstr,
        "pbc_dimensions": 0,
        "pH": f"{min(args.pypka_phmin, args.ph)}-{max(args.pypka_phmax, args.ph)}",
        "pHstep": args.pypka_phstep,
        "output": "pypka_pKas.out",
    }
    if args.pypka_params:
        with open(args.pypka_params) as fh:
            params.update(json.load(fh))
    with working_directory(workdir):
        with open("pypka_params.json", "w") as fh:
            json.dump(params, fh, indent=2)
        tit = Titration(params)
    grid = None
    records = []
    for s in tit:
        site = norm_site(s.res_name)
        if site is None:
            continue  # SER/THR etc.
        try:
            rn = s.getResNumber(correct_icode=True)
        except TypeError:
            rn = s.getResNumber()
        resnum, icode = parse_resnum(rn)
        chain = (getattr(s.molecule, "chain", "") or args.blank_chain).strip()
        pk = s.pK if s.pK is not None else np.nan
        curve = s.getTitrationCurve() or {}
        fprot = _interp_curve(curve, args.ph) if curve else np.nan
        his_taut = ""
        if site == "HIS" and curve:
            grid = np.array(sorted(curve.keys()), dtype=float)
            ph_near = float(grid[np.argmin(np.abs(grid - args.ph))])
            try:
                probs = s.getTautsProb(ph_near)  # [HID, HIE, HIP] for His (GROMOS order)
                his_taut = "HID" if probs[0] >= probs[1] else "HIE"
            except Exception:  # noqa: BLE001
                his_taut = ""
        records.append({
            "chain": chain, "resnum": resnum, "icode": icode, "site": site,
            "pka": float(pk), "pypka_fprot": fprot, "pypka_his_tautomer": his_taut,
        })
    return records


def run_pkai(pdb, workdir, args, st, model_name):
    try:
        from pkai.pKAI import pKAI          # pKAI >= 1.2 (lower-case package)
    except ImportError:
        from pKAI.pKAI import pKAI          # older releases
    import inspect
    kw = {}
    sig = inspect.signature(pKAI).parameters
    if "model_name" in sig:
        kw["model_name"] = model_name
    if "device" in sig:
        kw["device"] = args.pkai_device
    if "threads" in sig and args.ncpus and args.ncpus > 0:
        kw["threads"] = args.ncpus
    with working_directory(workdir):
        res = pKAI(os.path.abspath(pdb), **kw)
    records = []
    for chain, resnumb, resname, pk in res:
        site = norm_site(resname)
        if site is None:
            continue
        resnum, icode = parse_resnum(resnumb)
        records.append({"chain": (str(chain).strip() or args.blank_chain), "resnum": resnum,
                        "icode": icode, "site": site, "pka": float(pk)})
    with open(os.path.join(workdir, f"{model_name}_pkas.csv"), "w") as fh:
        fh.write("chain,resnum,site,pka\n")
        for r in records:
            fh.write(f"{r['chain']},{r['resnum']}{r['icode']},{r['site']},{r['pka']:.3f}\n")
    return records


DELPHIPKA_FF = {"amber": "amber", "charmm": "charmm22", "gromos": "gromos", "parse": "parseres"}


def locate_delphipka(args):
    """Find the delphiPKa executable and its parameter directory."""
    exe = args.delphipka_exe or shutil.which("delphiPKa") or shutil.which("delphipka")
    home = args.delphipka_home or os.environ.get("DELPHIPKA_HOME")
    if not exe and home:
        for cand in (os.path.join(home, "bin", "delphiPKa"),
                     os.path.join(home, "..", "..", "bin", "delphiPKa")):
            if os.path.isfile(cand):
                exe = cand
                break
    if not exe or not os.path.isfile(exe):
        raise ImportError("delphiPKa executable not found (use --delphipka-exe or put it on PATH)")
    exe = os.path.abspath(exe)
    if not home:
        bindir = os.path.dirname(exe)
        for cand in (os.path.join(bindir, "..", "share", "delphipka"),   # conda layout
                     os.path.join(bindir, "..")):                         # source tree layout
            if os.path.isdir(os.path.join(cand, "param")):
                home = cand
                break
    if not home or not os.path.isdir(os.path.join(home, "param")):
        raise RuntimeError("DelPhiPKa parameter directory not found (use --delphipka-home or "
                           "set DELPHIPKA_HOME to the folder that contains param/ and param2/)")
    return exe, os.path.abspath(home)


def _delphipka_runprm(args, pdb_name, param_dir):
    ff = DELPHIPKA_FF[args.delphipka_ff]
    tf = lambda b: "T" if b else "F"  # noqa: E731
    lines = [
        "## run.prm written by pka_compare.py",
        f"PDB file name                      : {pdb_name}",
        f"Charge parameter                   : {os.path.join(param_dir, ff + '.crg')}",
        f"Radius parameter                   : {os.path.join(param_dir, ff + '.siz')}",
        f"Topology parameter                 : {os.path.join(param_dir, 'topology')}",
        "Remove HETATM                      : T",
        "Remove water molecule              : T",
        "HETATM in PQR format               : F",
        "Do Protonation                     : T",
        "Do Energy Calculation              : T",
        "Do pKa's  Calculation              : T",
        f"Calculate More Residues            : {tf(args.delphipka_more_residues)}",
        # must stay T: DelPhiPKa 2.3 labels the ionizable residues inside the routine that
        # writes this PQR, so with F it titrates whatever the uninitialised flags say
        "Output PQR file (With Topology)    : T",
        "Output PQR file (With pKa result)  : T",
        f"At given pH Value                  : {args.ph:.2f}",
        "Gaussian surface                   : 1",
        "Surface cutoff                     : 20",
        f"Variance of Gaussian distribution  : {args.delphipka_sigma:.2f}",
        f"Internal Dielectric                : {args.delphipka_epsin:.1f}",
        "External Dielectric                : 80.0",
        f"Cluster Delimitation Threshold (A) : {args.delphipka_cluster:.1f}",
        "Kmean++ cluster number             : AUTO",
        "Hydrogen of GLU Attached to Atom   : OE1",
        "Hydrogen of ASP Attached to Atom   : OD1",
        "pH Initial Value                   : 0.0",
        "pH End Value                       : 14.0",
        f"pH Interval                        : {args.delphipka_phstep:.2f}",
        f"Salt Concentration                 : {args.delphipka_salt:.3f}",
    ]
    return "\n".join(lines) + "\n"


_DPK_LABEL = re.compile(r"^([A-Za-z]+?)(-?\d{4,})(\S?)$")


def _parse_delphipka_label(lab):
    m = _DPK_LABEL.match(lab.strip())
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2)), m.group(3)


def run_delphipka(pdb, workdir, args, st):
    exe, home = locate_delphipka(args)
    param_dir = os.path.join(home, "param2" if args.delphipka_more_residues else "param")
    os.makedirs(workdir, exist_ok=True)
    local_pdb = "input.pdb"
    shutil.copy(pdb, os.path.join(workdir, local_pdb))
    with open(os.path.join(workdir, "run.prm"), "w") as fh:
        fh.write(_delphipka_runprm(args, local_pdb, param_dir))
    nproc = args.delphipka_np if args.delphipka_np > 0 else args.ncpus
    if nproc > 1:
        cmd = shlex.split(args.delphipka_mpirun) + ["-np", str(nproc), exe, "run.prm"]
    else:
        cmd = [exe, "run.prm"]
    env = dict(os.environ)
    if hasattr(os, "geteuid") and os.geteuid() == 0:   # Open MPI refuses root without these
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    LOG.info("DelPhiPKa: %s (%s parameters, eps_in %.1f, sigma %.2f)", " ".join(cmd),
             args.delphipka_ff, args.delphipka_epsin, args.delphipka_sigma)
    with open(os.path.join(workdir, "delphipka.log"), "w") as log_fh:
        proc = subprocess.run(cmd, cwd=workdir, stdout=log_fh, stderr=subprocess.STDOUT,
                              env=env, timeout=args.ext_timeout)
    pka_file = os.path.join(workdir, "pKa.csv")
    titra_file = os.path.join(workdir, "titra.csv")
    if proc.returncode != 0 or not os.path.isfile(pka_file):
        raise RuntimeError(f"delphiPKa exited with code {proc.returncode} and/or wrote no pKa.csv "
                           f"(see {workdir}/delphipka.log)")

    # titration curves: probability of the CHARGED state per site
    curves = {}
    if os.path.isfile(titra_file):
        with open(titra_file) as fh:
            header = fh.readline().split()
            phs = np.array([float(x) for x in header[1:]])
            for line in fh:
                if line.startswith("-") or line.startswith("NetCharge"):
                    break
                tok = line.split()
                if len(tok) < 2:
                    continue
                curves[tok[0]] = np.array([float(x) for x in tok[1:1 + len(phs)]])
    records = []
    with open(pka_file) as fh:
        fh.readline()
        for line in fh:
            tok = line.split()
            if len(tok) < 2:
                continue
            parsed = _parse_delphipka_label(tok[0])
            if not parsed:
                continue
            resname, resnum, chain = parsed
            site = norm_site(resname)
            if site is None:
                continue  # SER/THR, nucleotides
            try:
                pk = float(tok[1])
            except ValueError:
                pk = np.nan        # 'undetermined': curve never crosses 0.5 in 0-14
            fprot = np.nan
            curve = curves.get(tok[0])
            if curve is not None and len(curve) == len(phs) and phs[0] <= args.ph <= phs[-1]:
                p_charged = float(np.interp(args.ph, phs, curve))
                fprot = p_charged if site in BASIC else 1.0 - p_charged
            records.append({"chain": chain or args.blank_chain, "resnum": resnum, "icode": "",
                            "site": site, "pka": pk, "delphipka_fprot": fprot})
    return records


COLUMN_ALIASES = {
    "chain": ["chain", "chain id", "chain_id", "chainid", "ch", "segid"],
    "resid": ["res id", "resid", "res_id", "resnum", "res num", "res_num", "resseq",
              "residue number", "residue_number", "res number", "res_number",
              "residue id", "residue_id", "number", "resi", "position"],
    "resname": ["res name", "resname", "res_name", "residue name", "residue_name",
                "restype", "res type", "residue type", "residue", "aa", "amino acid"],
    "pka": ["predict pka", "predicted pka", "pred pka", "pka_pred", "pred_pka",
            "pka pred", "pka", "pka value", "pka_value", "prediction", "predict_pka",
            "predicted_pka"],
}


def read_external_table(path, colmap_arg=None):
    """Read a pKa results table with flexible column names (CSV/TSV/whitespace)."""
    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception:  # noqa: BLE001
        df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#")
    cols = {c.strip().lower(): c for c in df.columns}
    mapping = {}
    if colmap_arg:
        for item in colmap_arg.split(","):
            k, v = item.split("=", 1)
            mapping[k.strip()] = v.strip()
    for key, aliases in COLUMN_ALIASES.items():
        if key in mapping:
            continue
        for a in aliases:
            if a in cols:
                mapping[key] = cols[a]
                break
    missing = [k for k in ("resid", "resname", "pka") if k not in mapping]
    if missing:
        raise ValueError(f"{path}: could not identify column(s) {missing}; columns are "
                         f"{list(df.columns)}. Use the --*-colmap option, e.g. "
                         f"'chain=Chain,resid=Res ID,resname=Res Name,pka=Predict pKa'.")
    out = []
    for _, row in df.iterrows():
        site = norm_site(row[mapping["resname"]])
        if site is None:
            continue
        try:
            resnum, icode = parse_resnum(row[mapping["resid"]])
            pk = float(row[mapping["pka"]])
        except (ValueError, TypeError):
            continue
        chain = str(row[mapping["chain"]]).strip() if "chain" in mapping else ""
        if chain.lower() in ("nan", "none", "-", "_"):
            chain = ""
        out.append({"chain": chain, "resnum": resnum, "icode": icode, "site": site, "pka": pk})
    return out


def resolve_chains(records, st, method):
    """Assign chains for records whose chain is missing or not in the structure."""
    by_res = defaultdict(list)
    for chain, residues in st.chains.items():
        for rn, ic, name in residues:
            by_res[(rn, ic)].append(chain)
    fixed, dropped = [], 0
    for r in records:
        if r["chain"] in st.chains:
            fixed.append(r)
            continue
        cands = by_res.get((r["resnum"], r["icode"]), [])
        if r["site"] not in ("NTR", "CTR"):
            cands = [c for c in cands
                     if TO_STANDARD.get(st.residue_name(c, r["resnum"], r["icode"]) or "") == r["site"]]
        if len(cands) == 1:
            r = dict(r, chain=cands[0])
            fixed.append(r)
        else:
            dropped += 1
    if dropped:
        LOG.warning("%s: %d record(s) could not be mapped unambiguously to a chain and were "
                    "dropped (give a chain column or use single-chain inputs).", method, dropped)
    return fixed


def run_external(method, pdb, workdir, args, st):
    csv_path = getattr(args, f"{method}_csv")
    cmd = getattr(args, f"{method}_cmd")
    colmap = getattr(args, f"{method}_colmap")
    if cmd:
        os.makedirs(workdir, exist_ok=True)
        out_csv = os.path.join(os.path.abspath(workdir), f"{method}_results.csv")
        fmt = {"pdb": os.path.abspath(pdb), "out": out_csv,
               "workdir": os.path.abspath(workdir), "ph": args.ph}
        command = cmd.format(**fmt)
        LOG.info("%s: running external command: %s", METHOD_LABELS[method], command)
        with open(os.path.join(workdir, f"{method}_cmd.log"), "w") as log_fh:
            proc = subprocess.run(command, shell=True, cwd=workdir, stdout=log_fh,
                                  stderr=subprocess.STDOUT, timeout=args.ext_timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"external command exited with code {proc.returncode} "
                               f"(see {workdir}/{method}_cmd.log)")
        if not os.path.isfile(out_csv):
            raise RuntimeError(f"external command did not write {out_csv}")
        csv_path = out_csv
    if not csv_path:
        raise RuntimeError(f"no --{method}-csv or --{method}-cmd given")
    recs = read_external_table(csv_path, colmap)
    if not recs:
        raise RuntimeError(f"no titratable residues parsed from {csv_path}")
    return resolve_chains(recs, st, METHOD_LABELS[method])


# --------------------------------------------------------------------------- #
# Merging and state assignment
# --------------------------------------------------------------------------- #

def state_from_fprot(f, ph_margin_frac):
    """P (protonated), D (deprotonated) or M (mixed) from a protonated fraction."""
    if not np.isfinite(f):
        return ""
    if f >= ph_margin_frac:
        return "P"
    if f <= 1.0 - ph_margin_frac:
        return "D"
    return "M"


def ff_name(site, state, ff, disulfide=False, his_taut=""):
    if site == "CYS" and disulfide:
        return FF_NAMES[ff]["CYS"]["SS"]
    if state in ("", None):
        return ""
    base = "P" if state in ("P", "P?") else "D"
    if site == "HIS" and base == "D":
        if his_taut == "HID":
            name = FF_NAMES[ff]["HIS"]["D_HID"]
        elif his_taut == "HIE":
            name = FF_NAMES[ff]["HIS"]["D_HIE"]
        else:
            name = FF_NAMES[ff]["HIS"]["D"] + "/" + FF_NAMES[ff]["HIS"]["D_HID"]
    else:
        name = FF_NAMES[ff][site][base]
    return name + ("?" if state.endswith("?") else "")


def build_tables(st, results, args):
    methods = [m for m in results if results[m]]
    wanted = set(args.sites)
    site_rows = OrderedDict()
    for chain, rn, ic, site, resname in st.titratable_sites(wanted):
        site_rows[(chain, rn, ic, site)] = {"resname": resname, "in_structure": True}

    long_rows = []
    extras = defaultdict(dict)
    unmatched = defaultdict(int)
    for m in methods:
        for r in results[m]:
            if r["site"] not in wanted:
                continue
            if args.chains and r["chain"] not in args.chains:
                continue
            key = (r["chain"], r["resnum"], r["icode"], r["site"])
            if key not in site_rows:
                # tolerate missing insertion codes from programs that drop them
                alt = [k for k in site_rows if k[0] == key[0] and k[1] == key[1] and k[3] == key[3]]
                if len(alt) == 1:
                    key = alt[0]
                else:
                    unmatched[m] += 1
                    rname = st.residue_name(key[0], key[1], key[2]) or key[3]
                    site_rows[key] = {"resname": rname, "in_structure": False}
            site_rows[key][m] = r["pka"]
            for k, v in r.items():
                if k not in ("chain", "resnum", "icode", "site", "pka"):
                    extras[key][k] = v
            long_rows.append({"chain": key[0], "resnum": key[1], "icode": key[2], "site": key[3],
                              "method": METHOD_LABELS[m], "pka": r["pka"],
                              **{k: v for k, v in r.items()
                                 if k not in ("chain", "resnum", "icode", "site", "pka")}})
    for m, n in unmatched.items():
        LOG.warning("%s reported %d site(s) not found in the cleaned structure; kept as extra rows.",
                    METHOD_LABELS[m], n)

    if args.chains:
        site_rows = OrderedDict((k, v) for k, v in site_rows.items() if k[0] in args.chains)

    frac_thr = 1.0 / (1.0 + 10.0 ** (-args.margin))  # f(protonated) at pKa = pH + margin
    rows = []
    for key, d in site_rows.items():
        chain, rn, ic, site = key
        ss = (chain, rn, ic) in st.disulfide and site == "CYS"
        row = OrderedDict()
        row["chain"] = chain
        row["resnum"] = rn
        row["icode"] = ic
        row["resname"] = d["resname"] if site not in ("NTR", "CTR") else site
        row["site"] = site
        row["label"] = site_label(chain, rn, ic, site, row["resname"])
        row["model_pKa"] = MODEL_PKA[site]
        vals = []
        states = []
        for m in methods:
            v = d.get(m, np.nan)
            v = np.nan if v is None else v
            row[f"pKa_{METHOD_LABELS[m]}"] = round(v, 2) if np.isfinite(v) else np.nan
            fkey = f"{m}_fprot"
            if m in ("pypka", "delphipka") and np.isfinite(extras[key].get(fkey, np.nan)):
                f = extras[key][fkey]  # titration-curve population, includes site coupling
            else:
                f = frac_protonated(v, args.ph)
            stt = state_from_fprot(f, frac_thr)
            row[f"state_{METHOD_LABELS[m]}"] = stt
            if np.isfinite(v):
                vals.append(v)
            if stt:
                states.append("P" if f >= 0.5 else "D")
        n = len(vals)
        row["n_methods"] = n
        if n and not ss:
            cons = float(np.median(vals) if args.consensus == "median" else np.mean(vals))
            row["pKa_consensus"] = round(cons, 2)
            row["pKa_SD"] = round(float(np.std(vals, ddof=1)), 2) if n > 1 else np.nan
            row["pKa_spread"] = round(float(np.max(vals) - np.min(vals)), 2) if n > 1 else np.nan
            row["shift_vs_model"] = round(cons - MODEL_PKA[site], 2)
            fc = frac_protonated(cons, args.ph)
        else:
            row["pKa_consensus"] = row["pKa_SD"] = row["pKa_spread"] = row["shift_vs_model"] = np.nan
            fc = np.nan
        row["f_protonated"] = round(fc, 3) if np.isfinite(fc) else np.nan
        row["charge"] = round(charge_from_fprot(site, fc), 3) if np.isfinite(fc) else np.nan
        if states:
            nP = states.count("P")
            maj = "P" if nP >= len(states) - nP else "D"
            row["agreement"] = f"{max(nP, len(states) - nP)}/{len(states)}"
        else:
            maj = ""
            row["agreement"] = ""
        cstate = state_from_fprot(fc, frac_thr)
        if cstate == "M":
            cstate = ("P" if fc >= 0.5 else "D") + "?"
        row["state_consensus"] = {"P": "protonated", "D": "deprotonated",
                                  "P?": "protonated?", "D?": "deprotonated?", "": ""}[cstate]
        if ss:
            row["state_consensus"] = "disulfide"
        # residue names
        taut = extras[key].get("pypka_his_tautomer", "") if site == "HIS" else ""
        row["his_tautomer_PypKa"] = taut if site == "HIS" else ""
        row["AMBER_name"] = ff_name(site, cstate, "amber", ss, taut)
        row["CHARMM_name"] = ff_name(site, cstate, "charmm", ss, taut)
        row["PROPKA_buried_pct"] = extras[key].get("propka_buried_pct", np.nan)
        row["PROPKA_coupled"] = extras[key].get("propka_coupled", "")
        # flags: 'check' items need a decision, 'note' items are informative
        check, note = [], []
        covering = [m for m in methods if site in METHOD_SITES[m]]
        missing = [METHOD_LABELS[m] for m in covering
                   if d.get(m) is None or not np.isfinite(d.get(m))]
        if ss:
            note.append("disulfide")
        elif n == 0:
            check.append("no prediction")
        else:
            if cstate.endswith("?"):
                check.append(f"|pKa-pH|<{args.margin:g}")
            if states and maj and len(states) > 1 and states.count(maj) < len(states):
                check.append("methods disagree")
            if n > 1 and row["pKa_spread"] >= args.spread_warn:
                check.append(f"spread>={args.spread_warn:g}")
            if n == 1 and len(covering) > 1:
                check.append("single method")
            if site not in ("NTR", "CTR") and st.metals:
                sc = [xyz for an, xyz in st.coords.get((chain, rn, ic), {}).items()
                      if an not in ("N", "CA", "C", "O", "OXT")]
                near = sorted({mn for mn, mx in st.metals for a in sc
                               if math.dist(a, mx) <= args.metal_cutoff})
                if near:
                    check.append("metal ligand (" + ",".join(near) + ")")
            if missing:
                note.append("missing: " + ",".join(missing))
            if row["PROPKA_coupled"] is True:
                note.append("PROPKA coupled")
            if site == "HIS" and cstate.startswith("D") and not taut:
                note.append("His tautomer unresolved")
        if not d.get("in_structure", True):
            check.append("not in structure")
        row["flag"] = "CHECK" if check else ("note" if note else "OK")
        row["flag_reasons"] = "; ".join(check + note)
        rows.append(row)
    wide = pd.DataFrame(rows)
    if not wide.empty:
        wide["_term"] = wide["site"].map(lambda s: 0 if s == "NTR" else (2 if s == "CTR" else 1))
        wide = wide.sort_values(["chain", "_term", "resnum", "icode"],
                                key=lambda c: c if c.name != "chain" else c.map(
                                    {ch: i for i, ch in enumerate(st.chains)}).fillna(999))
        wide = wide.drop(columns=["_term"]).reset_index(drop=True)
    if not wide.empty:
        # group columns: identity | pKa per method + consensus stats | states | names | flags
        ident = ["chain", "resnum", "icode", "resname", "site", "label", "model_pKa"]
        pkas = [c for c in wide.columns if c.startswith("pKa_") and c not in
                ("pKa_consensus", "pKa_SD", "pKa_spread")]
        stats = ["n_methods", "pKa_consensus", "pKa_SD", "pKa_spread", "shift_vs_model",
                 "f_protonated", "charge"]
        states = [c for c in wide.columns if c.startswith("state_") and c != "state_consensus"]
        rest = ["agreement", "state_consensus", "his_tautomer_PypKa", "AMBER_name", "CHARMM_name",
                "PROPKA_buried_pct", "PROPKA_coupled", "flag", "flag_reasons"]
        order = ident + pkas + stats + states + rest
        wide = wide[[c for c in order if c in wide.columns] +
                    [c for c in wide.columns if c not in order]]
    if not args.keep_icode_column and not wide.empty and (wide["icode"] == "").all():
        wide = wide.drop(columns=["icode"])
    long_df = pd.DataFrame(long_rows)
    return wide, long_df, methods


def net_charges(wide, methods, args):
    """
    Net charge of the titratable sites at pH, per method and for the consensus.
    Sites a method does not predict (e.g. Arg in PypKa) are filled with the
    consensus value so that totals are comparable between methods.
    """
    out = OrderedDict()
    use = wide[wide["state_consensus"] != "disulfide"]
    for m in methods:
        col = f"pKa_{METHOD_LABELS[m]}"
        q, n_own = 0.0, 0
        for _, r in use.iterrows():
            v = r[col]
            if np.isfinite(v):
                q += charge_from_fprot(r["site"], frac_protonated(v, args.ph))
                n_own += 1
            elif np.isfinite(r["charge"]):
                q += r["charge"]
        out[METHOD_LABELS[m]] = (q, n_own)
    q = use["charge"].dropna()
    out["Consensus"] = (float(q.sum()), int(q.shape[0]))
    return out


# --------------------------------------------------------------------------- #
# Output: PDB with state names, summary text, plot
# --------------------------------------------------------------------------- #

def write_state_pdb(st, wide, ff, out_path):
    names = {}
    for _, r in wide.iterrows():
        if r["site"] in ("NTR", "CTR"):
            continue
        n = str(r[f"{ff.upper()}_name"]).rstrip("?")
        if not n or "*" in n or "/" in n or "(" in n:
            n = {"HIS": "HIE" if ff == "amber" else "HSE"}.get(r["site"], r["site"]) \
                if "/" in n else r["resname"]
        icode = r["icode"] if "icode" in wide.columns else ""
        names[(r["chain"], int(r["resnum"]), icode)] = n
    with open(out_path, "w") as fh:
        fh.write(f"REMARK   pka_compare.py: residue names = consensus protonation states ({ff})\n")
        fh.write("REMARK   hydrogens removed; rebuild them with your preparation tool\n")
        for line in st.atom_lines:
            if line.startswith(("ATOM", "HETATM")):
                key = (line[21], int(line[22:26]), line[26].strip())
                if key in names:
                    nm = names[key]
                    line = line[:17] + f"{nm:<4s}"[:4] + line[21:]
            fh.write(line + "\n")


def write_summary(path, args, st, wide, methods, status, charges, timings):
    lines = []
    lines.append(f"pka_compare.py v{__version__}  -  {datetime.datetime.now():%Y-%m-%d %H:%M}")
    lines.append(f"Input structure : {os.path.abspath(args.input)}")
    lines.append(f"Target pH       : {args.ph:g}   (ambiguity margin +/- {args.margin:g} pH units)")
    lines.append(f"Consensus       : {args.consensus} of available predictions")
    lines.append(f"Chains          : {', '.join(st.chains.keys())}   "
                 f"disulfide CYS: {len(st.disulfide)}   metal ions: {len(st.metals)}")
    for w in st.warnings:
        lines.append(f"Warning         : {w}")
    lines.append("")
    lines.append("Method status")
    for m, s in status.items():
        t = timings.get(m)
        lines.append(f"  {METHOD_LABELS[m]:<9s} {s}" + (f"  ({t:.1f} s)" if t else ""))
    lines.append("")
    lines.append(f"Net charge of titratable sites at pH {args.ph:g} "
                 f"(sites predicted by the method; others filled with consensus)")
    for k, (q, n) in charges.items():
        lines.append(f"  {k:<10s} {q:+7.2f}   ({n})")
    lines.append("")
    if wide.empty:
        lines.append("No titratable sites.")
    else:
        mcols = [f"pKa_{METHOD_LABELS[m]}" for m in methods]
        show = ["label"] + mcols + ["pKa_consensus", "pKa_spread", "state_consensus",
                                    "AMBER_name", "flag", "flag_reasons"]
        fmt = wide[show].copy()
        fmt.columns = ["site"] + [METHOD_LABELS[m] for m in methods] + \
                      ["cons", "spread", "state", "AMBER", "flag", "reasons"]
        flagged = fmt[wide["flag"] == "CHECK"]
        lines.append(f"Sites needing attention: {len(flagged)} of {len(fmt)}")
        if len(flagged):
            lines.append(flagged.to_string(index=False, na_rep="-", float_format=lambda x: f"{x:.2f}"))
        lines.append("")
        lines.append("All sites")
        lines.append(fmt.to_string(index=False, na_rep="-", float_format=lambda x: f"{x:.2f}"))
        lines.append("")
        lines.append("Names ending in '?' are within the ambiguity margin of the target pH; "
                     "names with '*' have no standard residue in that force field family.")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def make_heatmaps(wide, methods, args, prefix):
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    if wide.empty:
        return []
    plot_df = wide
    if args.plot_only_flagged:
        plot_df = wide[wide["flag"] == "CHECK"]
        if plot_df.empty:
            LOG.info("No flagged sites: heatmap skipped (--plot-only-flagged).")
            return []
    cols = [f"pKa_{METHOD_LABELS[m]}" for m in methods] + ["pKa_consensus"]
    col_names = [METHOD_LABELS[m] for m in methods] + ["Consensus"]

    # Diverging blue <-> red with a neutral grey midpoint at pKa = pH.
    # Blue: pKa above pH (protonated); red: pKa below pH (deprotonated).
    cmap = LinearSegmentedColormap.from_list(
        "pka_div", ["#a32a2a", "#e34948", "#f0efec", "#2a78d6", "#104281"])
    cmap.set_bad("#e6e5e1")
    lim = args.plot_clip
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    ink, ink2, grid_c = "#0b0b0b", "#52514e", "#fcfcfb"

    chunks = [plot_df.iloc[i:i + args.rows_per_plot]
              for i in range(0, len(plot_df), args.rows_per_plot)]
    paths = []
    for ci, chunk in enumerate(chunks, 1):
        data = chunk[cols].to_numpy(dtype=float)
        data[(chunk["state_consensus"] == "disulfide").to_numpy(), :] = np.nan  # not titratable
        delta = data - args.ph
        nrow, ncol = data.shape
        fig_h = max(3.0, 0.24 * nrow + 1.6)
        fig_w = max(5.5, 0.85 * ncol + 3.2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(grid_c)
        ax.set_facecolor(grid_c)
        masked = np.ma.masked_invalid(delta)
        im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        # 2 px surface gap between cells
        ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
        ax.grid(which="minor", color=grid_c, linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        # consensus column separator
        ax.axvline(ncol - 1.5, color=ink2, linewidth=1.0)
        for i in range(nrow):
            for j in range(ncol):
                v = data[i, j]
                if np.isfinite(v):
                    dark = abs(delta[i, j]) > 0.62 * lim
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if dark else ink)
                else:
                    txt = "S-S" if chunk.iloc[i]["state_consensus"] == "disulfide" else "n/a"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=ink2)
        ylabels = []
        for _, r in chunk.iterrows():
            lab = r["label"]
            if r["flag"] == "CHECK":
                lab += " *"
            ylabels.append(lab)
        ax.set_yticks(range(nrow))
        ax.set_yticklabels(ylabels, fontsize=7, color=ink)
        for tl, (_, r) in zip(ax.get_yticklabels(), chunk.iterrows()):
            if r["flag"] == "CHECK":
                tl.set_fontweight("bold")
        ax.set_xticks(range(ncol))
        ax.set_xticklabels(col_names, fontsize=8, color=ink, rotation=30, ha="right")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        title = f"Predicted pKa vs pH {args.ph:g}"
        if len(chunks) > 1:
            title += f"  ({ci}/{len(chunks)})"
        ax.set_title(title, fontsize=10, color=ink, loc="left", pad=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
        cb.set_label("pKa - pH  (blue: protonated, red: deprotonated)", fontsize=7, color=ink2)
        cb.ax.tick_params(labelsize=7, colors=ink2)
        cb.outline.set_visible(False)
        fig.text(0.01, 0.005, "* flagged site (see table)   numbers: predicted pKa   n/a: not predicted   S-S: disulfide",
                 fontsize=6.5, color=ink2, ha="left", va="bottom")
        fig.tight_layout()
        suffix = f"_part{ci}" if len(chunks) > 1 else ""
        for fmt in args.plot_formats:
            p = f"{prefix}_heatmap{suffix}.{fmt}"
            fig.savefig(p, dpi=args.dpi, facecolor=fig.get_facecolor())
            paths.append(p)
        if not args.no_show:
            try:
                plt.show()
            except Exception:  # noqa: BLE001
                pass
        plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def quiet_third_party_logging():
    """Undo the DEBUG-level root logging some predictors switch on when imported."""
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("matplotlib", "PIL", "numba", "h5py"):
        logging.getLogger(name).setLevel(logging.WARNING)


def build_parser():
    p = argparse.ArgumentParser(
        prog="pka_compare.py",
        description="Comparative protein pKa prediction (PROPKA3, PypKa, pKAI/pKAI+, DeepKa, "
                    "ME-pKa) with per-residue protonation states at a target pH.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("input / output")
    g.add_argument("-i", "--input", required=True, help="input PDB file")
    g.add_argument("-o", "--outdir", default=None, help="output directory (default: pka_<input stem>)")
    g.add_argument("-n", "--name", default=None, help="output file prefix (default: input stem)")
    g.add_argument("--ph", type=float, default=7.0, help="target pH for protonation states")
    g.add_argument("--methods", default="propka,pypka,delphipka,pkai+,pkai,deepka,mepka",
                   help="comma-separated methods: " + ",".join(METHOD_LABELS) +
                        "; external methods without a --*-csv/--*-cmd are skipped")
    g.add_argument("--sites", default=",".join(DEFAULT_SITES),
                   help="site types to report (" + ",".join(ALL_SITES) + ")")
    g.add_argument("--chains", default=None, help="comma-separated chain IDs to report (default all)")
    g.add_argument("--write-pdb", choices=["none", "amber", "charmm"], default="none",
                   help="also write the cleaned PDB with consensus-state residue names")
    g.add_argument("--keep-icode-column", action="store_true",
                   help="keep the insertion-code column even when empty")
    g.add_argument("--force", action="store_true", help="overwrite an existing output directory")

    g = p.add_argument_group("structure cleaning")
    g.add_argument("--altloc", default="A", help="alternate location to keep ('first' = first seen)")
    g.add_argument("--keep-h", action="store_true", help="keep hydrogens from the input")
    g.add_argument("--keep-hetatm", action="store_true",
                   help="keep non-water HETATM records (PROPKA uses ligands; PypKa/pKAI ignore them)")
    g.add_argument("--blank-chain", default="A", help="chain ID given to atoms with a blank chain")
    g.add_argument("--ss-cutoff", type=float, default=2.5,
                   help="SG-SG distance (A) below which CYS pairs are treated as disulfides")
    g.add_argument("--metal-cutoff", type=float, default=3.0,
                   help="side-chain to metal-ion distance (A) for the 'metal ligand' flag")

    g = p.add_argument_group("consensus and flags")
    g.add_argument("--consensus", choices=["median", "mean"], default="median")
    g.add_argument("--margin", type=float, default=1.0,
                   help="|pKa - pH| below which a state is called ambiguous (pH units)")
    g.add_argument("--spread-warn", type=float, default=2.0,
                   help="flag sites whose max-min pKa across methods is at least this")

    g = p.add_argument_group("PROPKA3")
    g.add_argument("--propka-args", default="", help="extra PROPKA command-line options (quoted)")

    g = p.add_argument_group("PypKa")
    g.add_argument("--ncpus", type=int, default=-1, help="CPUs for PypKa/pKAI (-1 = all)")
    g.add_argument("--pypka-epsin", type=float, default=15.0, help="protein dielectric constant")
    g.add_argument("--pypka-ionicstr", type=float, default=0.1, help="ionic strength (M)")
    g.add_argument("--pypka-phmin", type=float, default=0.0, help="titration pH min")
    g.add_argument("--pypka-phmax", type=float, default=14.0, help="titration pH max")
    g.add_argument("--pypka-phstep", type=float, default=0.25, help="titration pH step")
    g.add_argument("--pypka-params", default=None,
                   help="JSON file with extra/overriding PypKa parameters")

    g = p.add_argument_group("DelPhiPKa")
    g.add_argument("--delphipka-exe", default=None,
                   help="delphiPKa executable (default: on PATH or under $DELPHIPKA_HOME)")
    g.add_argument("--delphipka-home", default=None,
                   help="folder with param/ and param2/ (default: $DELPHIPKA_HOME or next to the executable)")
    g.add_argument("--delphipka-ff", choices=sorted(DELPHIPKA_FF), default="amber",
                   help="charge/radius parameter set")
    g.add_argument("--delphipka-epsin", type=float, default=8.0, help="reference (internal) dielectric")
    g.add_argument("--delphipka-sigma", type=float, default=0.70,
                   help="variance of the Gaussian dielectric (0.90-0.95 for buried sites)")
    g.add_argument("--delphipka-salt", type=float, default=0.15, help="salt concentration (M)")
    g.add_argument("--delphipka-cluster", type=float, default=12.0,
                   help="cluster delimitation threshold (A)")
    g.add_argument("--delphipka-phstep", type=float, default=0.5, help="titration pH step")
    g.add_argument("--delphipka-more-residues", action="store_true",
                   help="also titrate Tyr and Cys (uses the param2 set; Ser/Thr are ignored here)")
    g.add_argument("--delphipka-np", type=int, default=0,
                   help="MPI processes (0 = --ncpus; 1 = run without mpirun)")
    g.add_argument("--delphipka-mpirun", default="mpirun", help="MPI launcher command")

    g = p.add_argument_group("pKAI")
    g.add_argument("--pkai-device", default="cpu", help="torch device for pKAI (cpu, cuda)")

    for m, lab in (("deepka", "DeepKa"), ("mepka", "ME-pKa")):
        g = p.add_argument_group(lab)
        g.add_argument(f"--{m}-csv", default=None,
                       help=f"{lab} results table (CSV/TSV) with chain, residue number, "
                            f"residue name and pKa columns")
        g.add_argument(f"--{m}-cmd", default=None,
                       help=f"shell command that runs {lab} and writes a results CSV; placeholders "
                            f"{{pdb}} {{out}} {{workdir}} {{ph}}")
        g.add_argument(f"--{m}-colmap", default=None,
                       help="explicit column map, e.g. 'chain=Chain,resid=Res ID,resname=Res Name,"
                            "pka=Predict pKa'")
    p.add_argument("--ext-timeout", type=int, default=24 * 3600,
                   help="timeout (s) for DelPhiPKa and external commands")

    g = p.add_argument_group("plot")
    g.add_argument("--no-plot", action="store_true", help="skip the heatmap")
    g.add_argument("--no-show", action="store_true", help="save the plot without displaying it")
    g.add_argument("--dpi", type=int, default=600, help="plot resolution")
    g.add_argument("--plot-formats", default="png", help="comma-separated formats (png,pdf,svg)")
    g.add_argument("--rows-per-plot", type=int, default=60, help="sites per heatmap page")
    g.add_argument("--plot-clip", type=float, default=4.0, help="colour scale limit for |pKa - pH|")
    g.add_argument("--plot-only-flagged", action="store_true", help="plot flagged sites only")

    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.input):
        sys.exit(f"Input file not found: {args.input}")
    stem = os.path.splitext(os.path.basename(args.input))[0]
    args.name = args.name or stem
    args.outdir = os.path.abspath(args.outdir or f"pka_{stem}")
    args.methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in args.methods if m not in METHOD_LABELS]
    if bad:
        sys.exit(f"Unknown method(s): {bad}. Choose from {list(METHOD_LABELS)}")
    args.sites = [s.strip().upper() for s in args.sites.split(",") if s.strip()]
    bad = [s for s in args.sites if s not in ALL_SITES]
    if bad:
        sys.exit(f"Unknown site type(s): {bad}. Choose from {ALL_SITES}")
    args.chains = [c.strip() for c in args.chains.split(",")] if args.chains else None
    if args.delphipka_more_residues:
        METHOD_SITES["delphipka"] |= {"TYR", "CYS"}
    args.plot_formats = [f.strip().lower() for f in args.plot_formats.split(",") if f.strip()]
    if args.ncpus == -1:
        args.ncpus = os.cpu_count() or 1
    if not 0 <= args.ph <= 14:
        sys.exit("--ph must be between 0 and 14")

    if os.path.isdir(args.outdir) and os.listdir(args.outdir) and not args.force:
        sys.exit(f"Output directory {args.outdir} exists and is not empty (use --force).")
    os.makedirs(os.path.join(args.outdir, "logs"), exist_ok=True)
    prefix = os.path.join(args.outdir, args.name)

    # own handlers, not the root logger: PypKa reconfigures the root logger to DEBUG on import
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    LOG.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(args.outdir, "logs", "pka_compare.log"), "w")):
        h.setFormatter(fmt)
        LOG.addHandler(h)
    LOG.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    LOG.propagate = False
    LOG.info("pka_compare.py v%s | input %s | pH %g", __version__, args.input, args.ph)
    with open(os.path.join(args.outdir, "logs", "arguments.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    clean = f"{prefix}_clean.pdb"
    st = clean_pdb(args.input, clean, keep_h=args.keep_h, keep_hetatm=args.keep_hetatm,
                   altloc=args.altloc, blank_chain=args.blank_chain, ss_cutoff=args.ss_cutoff)
    nres = sum(len(v) for v in st.chains.values())
    LOG.info("Cleaned structure: %d chain(s), %d residues, %d disulfide CYS -> %s",
             len(st.chains), nres, len(st.disulfide), os.path.basename(clean))
    if st.metals and not args.keep_hetatm:
        LOG.warning("%d metal ion(s) in the input were removed for the calculations; their "
                    "ligands are flagged. Treat those sites by hand.", len(st.metals))
    for w in st.warnings:
        LOG.warning(w)
    if nres == 0:
        sys.exit("No protein residues found in the input.")

    results, status, timings = OrderedDict(), OrderedDict(), {}
    for m in args.methods:
        lab = METHOD_LABELS[m]
        if m in EXTERNAL_METHODS and not (getattr(args, f"{m}_csv") or getattr(args, f"{m}_cmd")):
            status[m] = "skipped (no --%s-csv or --%s-cmd)" % (m, m)
            LOG.info("%s: skipped (no results file or command given)", lab)
            continue
        workdir = os.path.join(args.outdir, f"work_{m.replace('+', 'plus')}")
        t0 = time.time()
        LOG.info("%s: running ...", lab)
        try:
            if m == "propka":
                recs = run_propka(clean, workdir, args, st)
            elif m == "pypka":
                recs = run_pypka(clean, workdir, args, st)
            elif m == "delphipka":
                recs = run_delphipka(clean, workdir, args, st)
            elif m in ("pkai", "pkai+"):
                recs = run_pkai(clean, workdir, args, st, "pKAI+" if m == "pkai+" else "pKAI")
            else:
                recs = run_external(m, clean, workdir, args, st)
            results[m] = recs
            timings[m] = time.time() - t0
            status[m] = f"ok, {len(recs)} sites"
            LOG.info("%s: %d sites in %.1f s", lab, len(recs), timings[m])
        except ImportError as exc:
            status[m] = f"not installed ({exc})"
            LOG.error("%s: not installed (%s)", lab, exc)
        except Exception as exc:  # noqa: BLE001
            status[m] = f"failed: {exc}"
            LOG.error("%s failed: %s", lab, exc)
            with open(os.path.join(args.outdir, "logs", f"{m.replace('+', 'plus')}_error.log"), "w") as fh:
                fh.write(traceback.format_exc())

    quiet_third_party_logging()
    if not results:
        sys.exit("No method produced results; see the log.")

    wide, long_df, methods = build_tables(st, results, args)
    wide.to_csv(f"{prefix}_pka_table.csv", index=False)
    long_df.to_csv(f"{prefix}_pka_long.csv", index=False)
    try:
        with pd.ExcelWriter(f"{prefix}_pka_table.xlsx") as xw:
            wide.to_excel(xw, sheet_name="per_residue", index=False)
            long_df.to_excel(xw, sheet_name="long", index=False)
    except ImportError:
        LOG.info("openpyxl not available: .xlsx output skipped")

    charges = net_charges(wide, methods, args)
    text = write_summary(f"{prefix}_summary.txt", args, st, wide, methods, status, charges, timings)
    if args.write_pdb != "none":
        out = f"{prefix}_{args.write_pdb}.pdb"
        write_state_pdb(st, wide, args.write_pdb, out)
        LOG.info("Protonation-state PDB written: %s", os.path.basename(out))
    if not args.no_plot:
        paths = make_heatmaps(wide, methods, args, prefix)
        for pth in paths:
            LOG.info("Plot saved: %s", os.path.basename(pth))
    print()
    print(text)
    LOG.info("Done. Results in %s", args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
