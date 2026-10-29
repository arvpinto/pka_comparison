#!/usr/bin/env bash
# =============================================================================
# create_pka_env.sh - build a conda environment for pka_compare.py
#
# Installs, in one environment (Linux x86_64):
#   PROPKA 3, PypKa (+ DelPhi4py, pdbmender), pKAI/pKAI+ (PyTorch 1.13),
#   DelPhiPKa (compiled from source, MPI or serial), pandas/matplotlib/openpyxl
# plus the pieces PypKa needs at runtime: libgfortran.so.4, gawk and a private
# python2.7 (for its bundled pdb2pqr). pka_compare.py is installed as a command.
#
# Optional (--with-deepka): a second Python 3.7 environment with OpenBabel 2 and
# PyTorch for the DeepKa / ME-pKa reference code, and clones of both
# repositories. CHARMM and UCSF Chimera, which their featurisation also needs,
# are licensed downloads and are NOT installed.
#
# Works with conda, mamba or micromamba.
# =============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------- defaults
ENV_NAME="pka_compare"
ENV_PREFIX=""
CONDA_EXE_USER=""
TORCH_MODE="cpu"            # cpu | cuda | none
WITH_DELPHIPKA=1
DELPHIPKA_MODE="mpi"        # mpi | serial
DELPHIPKA_REPO="https://github.com/delphi001/DelPhiPKa.git"
DELPHIPKA_REF="master"
WITH_DEEPKA=0
DEEPKA_REPO="https://github.com/nonsensejoke/clone-deepKa-dev.git"
MEPKA_REPO="https://github.com/yzjyg215/ME-pKa.git"
JOBS="$(nproc 2>/dev/null || echo 2)"
PIPELINE_SCRIPT=""
RUN_TEST=0
ASSUME_YES=0
FORCE=0

PY_VERSION="3.10"           # PypKa's compiled DelPhi bindings stop at 3.10
PROPKA_VERSION="3.5.1"
PYPKA_VERSION="2.10.0"
PKAI_VERSION="1.2.0"
TORCH_VERSION="1.13.1"      # pKAI requires torch < 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
cat <<EOF
Usage: $(basename "$0") [options]

Environment
  -n, --name NAME           environment name (default: ${ENV_NAME})
  -p, --prefix DIR          create the environment at DIR instead of by name
      --conda EXE           conda, mamba or micromamba (default: auto-detect)
      --force               remove an existing environment with the same name/prefix
  -y, --yes                 do not ask for confirmation

Components
      --torch MODE          PyTorch for pKAI: cpu (default), cuda, none (skips pKAI)
      --no-delphipka        do not build DelPhiPKa
      --delphipka-serial    build DelPhiPKa without MPI (default: MPI build)
      --delphipka-repo URL  DelPhiPKa git repository (default: ${DELPHIPKA_REPO})
      --delphipka-ref REF   branch, tag or commit (default: ${DELPHIPKA_REF})
      --with-deepka         also create <name>-deepka (Python 3.7, OpenBabel 2, PyTorch)
                            and clone the DeepKa and ME-pKa repositories
  -j, --jobs N              parallel make jobs (default: ${JOBS})
      --pipeline FILE       pka_compare.py to install (default: next to this script)

Checks
      --test                run a smoke test with every installed predictor at the end
  -h, --help                show this help

Example
  bash $(basename "$0") -n pka_compare --torch cpu --test
  conda activate pka_compare
  pka_compare.py -i protein.pdb --ph 7 --methods propka,pypka,pkai+,delphipka
EOF
}

# ----------------------------------------------------------------------------- arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)            ENV_NAME="$2"; shift 2 ;;
        -p|--prefix)          ENV_PREFIX="$2"; shift 2 ;;
        --conda)              CONDA_EXE_USER="$2"; shift 2 ;;
        --force)              FORCE=1; shift ;;
        -y|--yes)             ASSUME_YES=1; shift ;;
        --torch)              TORCH_MODE="$2"; shift 2 ;;
        --no-delphipka)       WITH_DELPHIPKA=0; shift ;;
        --delphipka-serial)   DELPHIPKA_MODE="serial"; shift ;;
        --delphipka-repo)     DELPHIPKA_REPO="$2"; shift 2 ;;
        --delphipka-ref)      DELPHIPKA_REF="$2"; shift 2 ;;
        --with-deepka)        WITH_DEEPKA=1; shift ;;
        -j|--jobs)            JOBS="$2"; shift 2 ;;
        --pipeline)           PIPELINE_SCRIPT="$2"; shift 2 ;;
        --test)               RUN_TEST=1; shift ;;
        -h|--help)            usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

case "$TORCH_MODE" in cpu|cuda|none) ;; *) echo "--torch must be cpu, cuda or none" >&2; exit 1 ;; esac

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || \
    warn "Tested on Linux x86_64 only; PypKa ships Linux binaries, so other platforms will likely fail."

# ----------------------------------------------------------------------------- conda front end
if [[ -n "$CONDA_EXE_USER" ]]; then
    CONDA="$CONDA_EXE_USER"
elif command -v mamba >/dev/null 2>&1; then
    CONDA="mamba"
elif command -v conda >/dev/null 2>&1; then
    CONDA="conda"
elif command -v micromamba >/dev/null 2>&1; then
    CONDA="micromamba"
else
    die "No conda, mamba or micromamba found on PATH."
fi
command -v "$CONDA" >/dev/null 2>&1 || die "$CONDA not found."

# target selector (-n NAME or -p PREFIX)
if [[ -n "$ENV_PREFIX" ]]; then
    ENV_PREFIX="$(mkdir -p "$(dirname "$ENV_PREFIX")" && cd "$(dirname "$ENV_PREFIX")" && pwd)/$(basename "$ENV_PREFIX")"
    TARGET=(-p "$ENV_PREFIX")
    DEEPKA_TARGET=(-p "${ENV_PREFIX}-deepka")
    LABEL="$ENV_PREFIX"
else
    TARGET=(-n "$ENV_NAME")
    DEEPKA_TARGET=(-n "${ENV_NAME}-deepka")
    LABEL="$ENV_NAME"
fi

run_in_env() { "$CONDA" run "${TARGET[@]}" "$@"; }

[[ -z "$PIPELINE_SCRIPT" && -f "$SCRIPT_DIR/pka_compare.py" ]] && PIPELINE_SCRIPT="$SCRIPT_DIR/pka_compare.py"

cat <<EOF

pka_compare environment installer
  front end        : $CONDA
  environment      : $LABEL
  Python           : $PY_VERSION
  PyTorch / pKAI   : $TORCH_MODE
  DelPhiPKa        : $([[ $WITH_DELPHIPKA == 1 ]] && echo "build ($DELPHIPKA_MODE, $DELPHIPKA_REF)" || echo "skip")
  DeepKa / ME-pKa  : $([[ $WITH_DEEPKA == 1 ]] && echo "side environment + repositories" || echo "skip")
  pka_compare.py   : ${PIPELINE_SCRIPT:-not found (install it by hand)}
EOF
if [[ $ASSUME_YES -eq 0 ]]; then
    read -r -p "Proceed? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 0
fi

# ----------------------------------------------------------------------------- 1. main environment
PKGS=(
    "python=${PY_VERSION}" "numpy<2" pandas matplotlib openpyxl psutil pytest pip git
    gawk libgfortran4
)
if [[ $WITH_DELPHIPKA -eq 1 ]]; then
    PKGS+=(cxx-compiler make gsl "boost-cpp")
    [[ "$DELPHIPKA_MODE" == "mpi" ]] && PKGS+=(openmpi openmpi-mpicxx)
fi
case "$TORCH_MODE" in
    cpu)  PKGS+=("pytorch=${TORCH_VERSION}=cpu*") ;;
    cuda) PKGS+=("pytorch=${TORCH_VERSION}=cuda*") ;;
esac

if [[ $FORCE -eq 1 ]]; then
    log "Removing existing environment (if any)"
    "$CONDA" env remove -y "${TARGET[@]}" >/dev/null 2>&1 || true
fi

log "Creating environment $LABEL"
"$CONDA" create -y "${TARGET[@]}" --override-channels -c conda-forge "${PKGS[@]}"

PREFIX="$(run_in_env python -c 'import sys; print(sys.prefix)' | tail -n 1)"
[[ -d "$PREFIX" ]] || die "Could not resolve the environment prefix."
echo "Environment prefix: $PREFIX"

# ----------------------------------------------------------------------------- 2. python packages
log "Installing PROPKA ${PROPKA_VERSION}, PypKa ${PYPKA_VERSION}$([[ $TORCH_MODE != none ]] && echo ", pKAI ${PKAI_VERSION}")"
CONSTRAINTS="$(mktemp)"
printf 'numpy<2\n' > "$CONSTRAINTS"
[[ "$TORCH_MODE" != "none" ]] && printf 'torch==%s\n' "$TORCH_VERSION" >> "$CONSTRAINTS"
PIP_PKGS=("propka==${PROPKA_VERSION}" "pypka==${PYPKA_VERSION}")
[[ "$TORCH_MODE" != "none" ]] && PIP_PKGS+=("pkai==${PKAI_VERSION}")
# conda's pytorch is not visible to pip as "torch" metadata in every build: install pKAI's
# own dependency list without letting pip pull a second (CUDA) torch wheel
run_in_env python -m pip install --no-input -c "$CONSTRAINTS" "propka==${PROPKA_VERSION}" "pypka==${PYPKA_VERSION}"
if [[ "$TORCH_MODE" != "none" ]]; then
    if run_in_env python -c 'import importlib.metadata as m; m.version("torch")' >/dev/null 2>&1; then
        run_in_env python -m pip install --no-input -c "$CONSTRAINTS" "pkai==${PKAI_VERSION}"
    else
        run_in_env python -m pip install --no-input --no-deps "pkai==${PKAI_VERSION}"
    fi
fi
rm -f "$CONSTRAINTS"

# ----------------------------------------------------------------------------- 3. python2.7 for PypKa's pdb2pqr
log "Installing a private python2.7 (needed by PypKa's bundled pdb2pqr)"
PY27_PREFIX="$PREFIX/opt/python27"
if [[ ! -x "$PY27_PREFIX/bin/python2.7" ]]; then
    "$CONDA" create -y -p "$PY27_PREFIX" --override-channels -c conda-forge "python=2.7"
fi
ln -sf "$PY27_PREFIX/bin/python2.7" "$PREFIX/bin/python2.7"

# ----------------------------------------------------------------------------- 4. DelPhiPKa
if [[ $WITH_DELPHIPKA -eq 1 ]]; then
    log "Building DelPhiPKa ($DELPHIPKA_MODE) from $DELPHIPKA_REPO ($DELPHIPKA_REF)"
    SRC="$PREFIX/src/DelPhiPKa"
    rm -rf "$SRC"
    mkdir -p "$PREFIX/src"
    git clone --quiet "$DELPHIPKA_REPO" "$SRC"
    git -C "$SRC" checkout --quiet "$DELPHIPKA_REF"
    git -C "$SRC" rev-parse HEAD > "$SRC/.built_commit"

    # point the Makefile at the environment's headers/libraries and compiler
    if [[ "$DELPHIPKA_MODE" == "mpi" ]]; then
        CXX_LINE='CC=mpicxx'
        # DelPhiPKa uses the MPI C++ bindings (MPI::COMM_WORLD ...), which current
        # Open MPI (5.x) and MPICH (4.x) no longer provide: map them onto the C API
        cat > "$SRC/src/delphiPKa/mpi_cxx_compat.h" <<'HDR'
// mpi_cxx_compat.h - minimal replacement for the MPI C++ bindings (MPI::...) that
// DelPhiPKa uses. The C++ bindings were removed in MPI-3 and are absent from
// Open MPI 5 and MPICH 4; this header maps the calls DelPhiPKa makes onto the C API.
// Written by create_pka_env.sh (pka_compare).
#ifndef MPI_CXX_COMPAT_H_
#define MPI_CXX_COMPAT_H_

#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif
#include <mpi.h>

namespace MPI {

static const MPI_Datatype FLOAT  = MPI_FLOAT;
static const MPI_Datatype DOUBLE = MPI_DOUBLE;
static const MPI_Datatype INT    = MPI_INT;
static const MPI_Datatype CHAR   = MPI_CHAR;
static const MPI_Datatype BOOL   = MPI_CXX_BOOL;

class Intracomm {
  public:
    explicit Intracomm(MPI_Comm comm) : comm_(comm) {}
    int Get_rank() const { int r = 0; MPI_Comm_rank(comm_, &r); return r; }
    int Get_size() const { int s = 0; MPI_Comm_size(comm_, &s); return s; }
    void Barrier() const { MPI_Barrier(comm_); }
    void Bcast(void *buf, int count, MPI_Datatype type, int root) const {
        MPI_Bcast(buf, count, type, root, comm_);
    }
    void Gather(const void *sbuf, int scount, MPI_Datatype stype,
                void *rbuf, int rcount, MPI_Datatype rtype, int root) const {
        MPI_Gather(sbuf, scount, stype, rbuf, rcount, rtype, root, comm_);
    }
    void Gatherv(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                 const int rcounts[], const int displs[], MPI_Datatype rtype, int root) const {
        MPI_Gatherv(sbuf, scount, stype, rbuf, rcounts, displs, rtype, root, comm_);
    }
  private:
    MPI_Comm comm_;
};

static Intracomm COMM_WORLD(MPI_COMM_WORLD);

inline void Init() { MPI_Init(NULL, NULL); }
inline void Init(int &argc, char **&argv) { MPI_Init(&argc, &argv); }
inline void Finalize() { MPI_Finalize(); }

}  // namespace MPI

#endif  // MPI_CXX_COMPAT_H_
HDR
        sed -i 's|^#include <mpi.h>|#include "mpi_cxx_compat.h"|' "$SRC/src/delphiPKa/prime_environment.h"
    else
        CXX_LINE='CC=$(CXX)'
        sed -i -e 's|^#define MPI_PARALLEL|//#define MPI_PARALLEL|' \
               -e 's|^#include <mpi.h>|//#include <mpi.h>|' "$SRC/src/delphiPKa/prime_environment.h"
    fi
    sed -i -e "s|^CC=.*|${CXX_LINE}|" \
           -e "s|^CFLAGS=\(.*\)|CFLAGS=\1 -I${PREFIX}/include|" \
           -e "s|^LDFLAGS=\(.*\)|LDFLAGS=\1 -L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib|" \
           "$SRC/Makefile"
    run_in_env bash -c "cd '$SRC' && make -j ${JOBS}" > "$SRC/build.log" 2>&1 || {
        tail -n 40 "$SRC/build.log" >&2
        die "DelPhiPKa build failed (full log: $SRC/build.log)"
    }
    install -m 755 "$SRC/bin/delphiPKa" "$PREFIX/bin/delphiPKa"
    mkdir -p "$PREFIX/share/delphipka"
    cp -r "$SRC/param" "$SRC/param2" "$SRC/run.prm" "$SRC/sample.pdb" "$PREFIX/share/delphipka/"
    [[ -f "$SRC/UserManual.pdf" ]] && cp "$SRC/UserManual.pdf" "$PREFIX/share/delphipka/"
    # object files are large; keep sources and the build log only
    run_in_env bash -c "cd '$SRC' && make clean" >/dev/null 2>&1 || true
fi

# ----------------------------------------------------------------------------- 5. pka_compare + activation hooks
if [[ -n "$PIPELINE_SCRIPT" ]]; then
    log "Installing pka_compare.py"
    install -m 755 "$PIPELINE_SCRIPT" "$PREFIX/bin/pka_compare.py"
    sed -i "1s|.*|#!${PREFIX}/bin/python|" "$PREFIX/bin/pka_compare.py"
fi

log "Writing activation hooks"
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
cat > "$PREFIX/etc/conda/activate.d/zz_pka_compare.sh" <<'EOF'
# pka_compare: runtime settings for PypKa (libgfortran.so.4) and DelPhiPKa
export _PKA_COMPARE_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [ -d "${CONDA_PREFIX}/share/delphipka" ]; then
    export DELPHIPKA_HOME="${CONDA_PREFIX}/share/delphipka"
fi
EOF
cat > "$PREFIX/etc/conda/deactivate.d/zz_pka_compare.sh" <<'EOF'
if [ -n "${_PKA_COMPARE_OLD_LD_LIBRARY_PATH+x}" ]; then
    if [ -n "${_PKA_COMPARE_OLD_LD_LIBRARY_PATH}" ]; then
        export LD_LIBRARY_PATH="${_PKA_COMPARE_OLD_LD_LIBRARY_PATH}"
    else
        unset LD_LIBRARY_PATH
    fi
    unset _PKA_COMPARE_OLD_LD_LIBRARY_PATH
fi
unset DELPHIPKA_HOME
EOF

# ----------------------------------------------------------------------------- 6. DeepKa / ME-pKa side environment
if [[ $WITH_DEEPKA -eq 1 ]]; then
    log "Creating the DeepKa / ME-pKa side environment"
    if [[ $FORCE -eq 1 ]]; then
        "$CONDA" env remove -y "${DEEPKA_TARGET[@]}" >/dev/null 2>&1 || true
    fi
    "$CONDA" create -y "${DEEPKA_TARGET[@]}" --override-channels -c conda-forge \
        "python=3.7" "openbabel=2.4.1" "pytorch=1.*=cpu*" numpy pandas scipy h5py \
        matplotlib seaborn openpyxl requests tqdm tensorboardx einops pip
    "$CONDA" run "${DEEPKA_TARGET[@]}" python -m pip install --no-input fair-esm
    mkdir -p "$PREFIX/share/external"
    for pair in "deepka|$DEEPKA_REPO" "mepka|$MEPKA_REPO"; do
        name="${pair%%|*}"; url="${pair#*|}"
        rm -rf "$PREFIX/share/external/$name"
        git clone --quiet --depth 1 "$url" "$PREFIX/share/external/$name"
    done
    warn "DeepKa / ME-pKa also need CHARMM ('charmm' on PATH) and UCSF Chimera ('chimera' on PATH)."
    warn "Both repositories contain hard-coded paths; ME-pKa ships no trained weights."
    warn "Use their output with: pka_compare.py --deepka-csv / --mepka-csv (or --*-cmd)."
fi

# ----------------------------------------------------------------------------- 7. verification
log "Checking the installation"
CHECK_PY="$(mktemp --suffix=.py)"
cat > "$CHECK_PY" <<'PY'
import importlib
import logging
import shutil
import sys

logging.disable(logging.INFO)   # PypKa turns on DEBUG logging when imported

required = [("propka", "PROPKA3"), ("pypka", "PypKa"), ("pandas", "pandas"),
            ("matplotlib", "matplotlib"), ("openpyxl", "openpyxl")]
optional = [("pkai.pKAI", "pKAI")]
failed = False
for mod, label in required + optional:
    try:
        importlib.import_module(mod)
        print(f"  [ok]   {label}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label}: {exc}")
        failed = failed or (mod, label) in required
for exe in ("python2.7", "gawk", "delphiPKa", "mpirun", "pka_compare.py"):
    path = shutil.which(exe)
    status = "ok" if path else "--"
    print(f"  [{status}]   {exe}: {path or 'not found'}")
sys.exit(1 if failed else 0)
PY
run_in_env bash -c "export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"; python '$CHECK_PY'" \
    || warn "Some components failed the import check (see above)."
rm -f "$CHECK_PY"

if [[ $RUN_TEST -eq 1 ]]; then
    log "Smoke test on DelPhiPKa's sample.pdb"
    TEST_DIR="$(mktemp -d)"
    SAMPLE="$PREFIX/share/delphipka/sample.pdb"
    if [[ ! -f "$SAMPLE" ]]; then
        SAMPLE="$TEST_DIR/sample.pdb"
        run_in_env python - "$SAMPLE" <<'PY'
import sys, os
import pypka
root = os.path.dirname(pypka.__file__)
for dp, dn, fn in os.walk(root):
    for f in fn:
        if f.lower().endswith(".pdb"):
            import shutil; shutil.copy(os.path.join(dp, f), sys.argv[1]); sys.exit(0)
sys.exit("no test structure found")
PY
    fi
    METHODS="propka,pypka"
    [[ "$TORCH_MODE" != "none" ]] && METHODS+=",pkai+,pkai"
    [[ $WITH_DELPHIPKA -eq 1 ]] && METHODS+=",delphipka"
    run_in_env bash -c "export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"; \
        export DELPHIPKA_HOME=\"\$CONDA_PREFIX/share/delphipka\"; cd '$TEST_DIR' && \
        pka_compare.py -i '$SAMPLE' -o test_out --methods $METHODS --no-show --ncpus ${JOBS}" \
        | sed -n '/Method status/,/^$/p'
    echo "Smoke-test output kept in $TEST_DIR/test_out"
fi

log "Done"
cat <<EOF
Activate with:
  $( [[ "$CONDA" == micromamba ]] && echo "micromamba" || echo "conda" ) activate $LABEL
Run:
  pka_compare.py -i protein.pdb --ph 7.0 --methods propka,pypka,pkai+,pkai,delphipka
EOF
