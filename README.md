# G‑SchNet con QM9 →  OE62: instalación, entrenamiento, generación y exportación

> Esta guía reproduce el pipeline completo en Linux con GPU(s) NVIDIA. Incluye: entorno reproducible con conda, preparación de **QM9**, entrenamiento de **G‑SchNet**, generación/filtrado/exportación a **XYZ/CIF**, y cómo llevar el método a un **dataset propio (p.ej. OE62)**.

---

## 0) Requisitos
- Linux x86_64 con drivers NVIDIA correctos (`nvidia-smi` funciona).
- **CUDA a nivel sistema no es obligatorio**; usamos `pytorch-cuda` dentro del entorno.
- 1–2 GPUs (8 GB VRAM por GPU funcionan para QM9).  
- Git + conexión a internet para instalar dependencias.

---

## 1) Crear y activar el entorno `gschnet`

> Si ya tienes conda inicializado en tu shell, puedes saltarte el primer comando.

```bash
# Inicializa conda en la shell actual
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/.conda/etc/profile.d/conda.sh"
conda config --set channel_priority strict

# Crea el entorno (Python 3.10 + PyTorch CUDA 12.1 + Lightning + herramientas)
conda create -y -n gschnet -c pytorch -c nvidia -c conda-forge \
  python=3.10 pytorch=2.4.1 torchvision=0.19.1 pytorch-cuda=12.1 \
  pytorch-lightning=2.3.* hydra-core=1.3 \
  ase rdkit tqdm torchmetrics "numpy<2" "sympy<=1.12"

conda activate gschnet
```

Instala SchNetPack y G‑SchNet:
```bash
pip install schnetpack==2.1.1      # SchNetPack 2.x
git clone https://github.com/atomistic-machine-learning/schnetpack-gschnet.git "$HOME/schnetpack-gschnet"
pip install "$HOME/schnetpack-gschnet"
```

Sanity check rápido:
```bash
python - << 'PY'
import torch, torchvision, schnetpack, schnetpack_gschnet
from torchvision.ops import nms
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| GPU avail:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__, "| nms OK:", callable(nms))
print("schnetpack:", schnetpack.__version__, "| gschnet import OK")
PY
```

---

## 2) Copiar configs base (Hydra)

```bash
mkdir -p "$HOME/my_gschnet_configs"
cp -r "$HOME/schnetpack-gschnet/src/schnetpack_gschnet/configs/." "$HOME/my_gschnet_configs"
```

---

## 3) Preparar **QM9** (crear `qm9.db` con metadatos)

> **Importante:** Hydra **no** expande `~` en overrides. Usa `$HOME` entre comillas.

```bash
mkdir -p "$HOME/data"

gschnet_train --config-dir="$HOME/my_gschnet_configs" \
  experiment=gschnet_qm9 \
  "data.datapath=$HOME/data/qm9.db" \
  trainer.max_epochs=0
```

Esto **descarga**, **parsea** y crea un `qm9.db` correcto (con `distance_unit` y `_property_unit_dict`) que luego se usa para comparaciones (train/val/test/novel).

---

## 4) Entrenamiento en QM9

### Opción A — 1 GPU (recomendado para 8 GB VRAM)
```bash
gschnet_train --config-dir="$HOME/my_gschnet_configs" \
  experiment=gschnet_qm9 \
  trainer.accelerator=gpu trainer.precision=16 \
  trainer.max_epochs=220 \
  globals.lr=3e-4 \
  data.batch_size=32 data.num_workers=8 data.pin_memory=True \
  globals.draw_random_samples=12 \
  globals.model_cutoff=10 globals.prediction_cutoff=5 \
  callbacks.early_stopping.patience=30
```

### Opción B — 2 GPUs (DDP)
```bash
gschnet_train --config-dir="$HOME/my_gschnet_configs" \
  experiment=gschnet_qm9 \
  trainer.accelerator=gpu trainer.devices=2 trainer.strategy=ddp \
  trainer.precision=16 \
  trainer.max_epochs=220 \
  globals.lr=3e-4 \
  data.batch_size=16 data.num_workers=12 data.pin_memory=True \
  globals.draw_random_samples=12 \
  globals.model_cutoff=10 globals.prediction_cutoff=5 \
  callbacks.early_stopping.patience=30
```

El modelo queda en:
```
$HOME/models/qm9_no_conditions/<run-id>/best_model
```

---

## 5) Generación básica (1 GPU) + filtrado + export

Genera 1000 moléculas:
```bash
export MODELDIR="$HOME/models/qm9_no_conditions/<run-id>"
gschnet_generate modeldir="$MODELDIR" n_molecules=1000 batch_size=512 max_n_atoms=120
```

Filtrado y estadísticas de validez/únicos/novedad (usa tu QM9 para comparar):
```bash
python "$HOME/schnetpack-gschnet/src/scripts/check_validity.py" \
  "$MODELDIR/generated_molecules/1.db" \
  --compute_uniqueness \
  --compare_db_path "$HOME/data/qm9.db" \
  --compare_db_split_path "$MODELDIR/split.npz" \
  --ignore_enantiomers --timeout 2 \
  --results_db_path auto --results_db_flags unique
```

Exportar a **multi‑XYZ** (robusto en cualquier versión de ASE):
```bash
export FILTERED_DB="$(ls -1t "$MODELDIR"/generated_molecules/*_filtered.db | head -n1)"
export MULTI_XYZ="$MODELDIR/exports/$(basename "${FILTERED_DB%.db}").xyz"
mkdir -p "$(dirname "$MULTI_XYZ")"

python - << 'PY'
import os
from ase.db import connect
from ase.io import write
db = os.environ["FILTERED_DB"]; out = os.environ["MULTI_XYZ"]
rows = list(connect(db).select())
write(out, [r.toatoms() for r in rows], format="extxyz")
print(f"Wrote {len(rows)} structures to {out}")
PY
```

Exportar **por molécula** a XYZ/CIF:
```bash
export TS="$(basename "$FILTERED_DB" .db | sed 's/filtered_//')"
export EXP_DIR_XYZ="$MODELDIR/exports_xyz/$TS"
export EXP_DIR_CIF="$MODELDIR/exports_cif/$TS"
mkdir -p "$EXP_DIR_XYZ" "$EXP_DIR_CIF"

python - << 'PY'
import os
from ase.db import connect
from ase.io import write
db = connect(os.environ["FILTERED_DB"])
x = os.environ["EXP_DIR_XYZ"]; c = os.environ["EXP_DIR_CIF"]
n=0
for i,row in enumerate(db.select(), start=1):
    at=row.toatoms(); at.center(vacuum=8.0); at.set_pbc([False]*3)
    write(os.path.join(x,f"mol_{i:04d}.xyz"), at)
    write(os.path.join(c,f"mol_{i:04d}.cif"), at)
    n+=1
print(f"Exportadas {n} moléculas a:\n  - {x}\n  - {c}")
PY
```

---

## 6) Generación **paralela** (2+ GPUs) con merge (metadatos), filtrado y export

Usa el script listo para usar (**incluido en este zip**): `generate_parallel_merge_filter.sh`.

```bash
# Copia el script (si aún no lo tienes)
cp ./generate_parallel_merge_filter.sh "$HOME/generate_parallel_merge_filter.sh"
chmod +x "$HOME/generate_parallel_merge_filter.sh"

# Ejecuta en 2 GPUs
conda activate gschnet
export MODELDIR="$HOME/models/qm9_no_conditions/<run-id>"
bash "$HOME/generate_parallel_merge_filter.sh" \
  -m "$MODELDIR" -n 2000 -b 512 -a 120 \
  -d "$HOME/data/qm9.db" -G "0,1"
```

El script:
- Genera en paralelo por GPU con nombres únicos (timestamp),
- **Mergea** preservando metadatos (clave para SchNetPack),
- Filtra con `check_validity.py` (opciones recomendadas),
- Exporta **multi‑XYZ** y **XYZ/CIF** por molécula.

> Puedes abrir los `.xyz` con **ase gui**, **Avogadro**, **VMD**, etc.

---

## 7) Llevarlo a tu **dataset (OE62 u otro)**

Para usar datos propios, SchNetPack requiere un **ASE Atoms DB** (`.db`) con unidades y propiedades. Hay dos casos:

### Caso A — Ya tienes **3D** (p.ej. SDF con coordenadas)
```python
# guarda como: make_db_from_sdf.py
from ase import Atoms
from ase.db import connect
from rdkit import Chem

sdf_path = "OE62.sdf"           # <- ajusta
db_out   = "OE62.db"
distance_unit = "Angstrom"
prop_units = {"energy":"eV", "gap":"eV"}   # ajusta a tus propiedades

# Crear DB con metadatos
from schnetpack.data import ASEAtomsData
ASEAtomsData.create(db_out, distance_unit=distance_unit, property_unit_dict=prop_units)

suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
db = connect(db_out)
for mol in suppl:
    if mol is None: 
        continue
    conf = mol.GetConformer()
    Z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    pos = [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
           for i in range(mol.GetNumAtoms())]
    at = Atoms(numbers=Z, positions=pos)
    # Ejemplo de propiedades: intenta leer campos SDF si existen
    props = {}
    for k in ("energy","gap"):
        if mol.HasProp(k):
            try: props[k] = float(mol.GetProp(k))
            except: pass
    db.write(at, data={k: [v] for k,v in props.items()})
print("DB listo:", db_out)
```

### Caso B — Tienes **SMILES** y necesitas 3D
```python
# guarda como: make_db_from_smiles.py
from ase import Atoms
from ase.db import connect
from rdkit import Chem
from rdkit.Chem import AllChem

smiles_csv = "OE62.csv"   # columnas: smiles[, gap, energy, ...]
col_smiles = "smiles"
db_out = "OE62.db"
distance_unit = "Angstrom"
prop_units = {"gap":"eV"}  # ajusta a tus columnas

from schnetpack.data import ASEAtomsData
ASEAtomsData.create(db_out, distance_unit=distance_unit, property_unit_dict=prop_units)

import csv
db = connect(db_out)
with open(smiles_csv) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        smi = row[col_smiles].strip()
        if not smi: continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        mol = Chem.AddHs(mol)
        ok = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if ok != 0: continue
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        conf = mol.GetConformer()
        Z = [a.GetAtomicNum() for a in mol.GetAtoms()]
        pos = [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
               for i in range(mol.GetNumAtoms())]
        at = Atoms(numbers=Z, positions=pos)
        # Propiedades opcionales desde el CSV
        props = {}
        for k in ("gap","energy"):
            if k in row and row[k]:
                try: props[k] = float(row[k])
                except: pass
        db.write(at, data={k: [v] for k,v in props.items()})
print("DB listo:", db_out)
```

> **Tip:** Determina los elementos de tu dataset para `globals.atom_types`:
```python
from ase.db import connect
Z=set()
for r in connect("OE62.db").select():
    Z.update(r.numbers)
print(sorted(Z))   # e.g., [1,6,7,8,9,16,17,...]
```

### Entrenar con tu DB
Usa el experimento plantilla `gschnet_template` + `data=custom_data` que ya trae SchNetPack:

```bash
# Ajusta los tipos atómicos y cutoffs a tu dataset
gschnet_train --config-dir="$HOME/my_gschnet_configs" \
  experiment=gschnet_template \
  data.datapath="$HOME/data/OE62.db" \
  data.batch_size=10 data.num_train=50000 data.num_val=5000 \
  globals.name=custom_data globals.id=run1 \
  "globals.atom_types=[1,6,7,8,9,16,17]" \
  globals.model_cutoff=10 globals.prediction_cutoff=5 globals.placement_cutoff=1.7 \
  globals.draw_random_samples=10
```

> Para datasets **grandes**, usa `data=custom_data_cached` (cachea neighborlists):  
> añade `data=custom_data_cached` y define `globals.cache_workdir` (disco rápido del nodo).

### Modelos condicionados (opcional, p.ej. HOMO‑LUMO gap)
- Crea un conditioning YAML (`model/conditioning/gap.yaml`) como en los ejemplos de `my_gschnet_configs` (ver `gap_relenergy.yaml`).
- Entrena con `model/conditioning=gap` o define un experimento que ya lo incluya.
- En generación, pasa los objetivos: `-C "++conditions.gap=4.0"` (ajusta a tus unidades).

---

## 8) Consejos de rendimiento
- VRAM justa → baja `data.batch_size` y/o sube `globals.draw_random_samples` (5–16).
- `globals.prediction_cutoff=5`, `globals.model_cutoff=10` son buenos inicios.
- I/O lento → usa `data.num_workers` 8–12 y `data.pin_memory=True`. Considera `globals.data_workdir`/`globals.cache_workdir` a disco local.
- Activa Tensor Cores: al inicio de tu script Python, añade:  
  ```python
  import torch; torch.set_float32_matmul_precision('high')
  ```

---

## 9) FAQ (errores comunes)

- **Hydra error con `~`**  
  Usa `"data.datapath=$HOME/ruta.db"`; no uses `~` en overrides.

- **`torchvision::nms does not exist`**  
  Versiones desalineadas. Asegura `torch==2.4.1` y `torchvision==0.19.1` en el **mismo** entorno.

- **`KeyError: '_property_unit_dict'` al filtrar/mergear**  
  Tu `.db` no tiene metadatos. Crea `qm9.db` con `trainer.max_epochs=0` o usa nuestro script de merge que **preserva metadatos**.

- **Conda HTTP 000 / DNS**  
  Problemas de red. Reintenta; si persiste, configura proxy/DNS del sistema.

---

## 10) Hoja de trucos

```bash
# Activar
conda activate gschnet

# Dataset QM9 (una sola vez)
gschnet_train --config-dir="$HOME/my_gschnet_configs" \
  experiment=gschnet_qm9 "data.datapath=$HOME/data/qm9.db" trainer.max_epochs=0

# Entrenar (1 GPU)
gschnet_train --config-dir="$HOME/my_gschnet_configs" experiment=gschnet_qm9 \
  trainer.accelerator=gpu trainer.precision=16 trainer.max_epochs=220 \
  globals.lr=3e-4 data.batch_size=32 data.num_workers=8 data.pin_memory=True \
  globals.draw_random_samples=12 globals.model_cutoff=10 globals.prediction_cutoff=5 \
  callbacks.early_stopping.patience=30

# Generar+filtrar+exportar en 2 GPUs
export MODELDIR="$HOME/models/qm9_no_conditions/<run-id>"
bash "$HOME/generate_parallel_merge_filter.sh" -m "$MODELDIR" -n 2000 -b 512 -a 120 \
  -d "$HOME/data/qm9.db" -G "0,1"
```

---

## 11) Apéndice A — Script `generate_parallel_merge_filter.sh`

> También te lo dejo como archivo aparte listo para descargar.

```bash
#!/usr/bin/env bash
set -Eeuo pipefail; IFS=$'\n\t'
MODELDIR=""; N_MOLS=2000; BATCH=512; MAX_ATOMS=120
GRID_SPACING=0.05; TEMP_TERM=0.1; GPU_LIST="0,1"
TRAIN_DB=""; CONDITIONS=""; REPO_DIR="$HOME/schnetpack-gschnet"
usage(){ cat <<USAGE
Uso: bash $0 -m MODELDIR [-n N] [-b B] [-a A] [-G "0,1"] [-d TRAIN_DB] [-C "COND..."]
USAGE; exit 1; }
while getopts ":m:n:b:a:g:t:G:d:C:r:h" o; do case $o in
m) MODELDIR="$OPTARG";; n) N_MOLS="$OPTARG";; b) BATCH="$OPTARG";;
a) MAX_ATOMS="$OPTARG";; g) GRID_SPACING="$OPTARG";; t) TEMP_TERM="$OPTARG";;
G) GPU_LIST="$OPTARG";; d) TRAIN_DB="$OPTARG";; C) CONDITIONS="$OPTARG";;
r) REPO_DIR="$OPTARG";; h|*) usage;; esac; done
[[ -z "$MODELDIR" ]] && usage
CHECK_VALID="$REPO_DIR/src/scripts/check_validity.py"
TS="$(date +%Y%m%d_%H%M%S)"; GEN_DIR="$MODELDIR/generated_molecules"; mkdir -p "$GEN_DIR"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"; K="${#GPUS[@]}"; PER=$(( (N_MOLS + K - 1) / K ))
echo "[INFO] Generando $N_MOLS en ${K} GPUs (~$PER/GPU)"
PARTS=(); for g in "${GPUS[@]}"; do out="part_${TS}_gpu${g}.db"; PARTS+=("$out");
CUDA_VISIBLE_DEVICES="$g" gschnet_generate modeldir="$MODELDIR" \
  n_molecules="$PER" batch_size="$BATCH" max_n_atoms="$MAX_ATOMS" \
  outputfile="$out" grid_spacing="$GRID_SPACING" temperature_term="$TEMP_TERM" \
  ${CONDITIONS:-} & done; wait
MERGED_DB="${GEN_DIR}/merged_${TS}.db"; export MODELDIR MERGED_DB; export PARTS_CSV="$(IFS=:; echo "${PARTS[*]}")"
python - << 'PY'
import os, json, sqlite3
from ase.db import connect
from schnetpack.data import ASEAtomsData
md = os.environ["MODELDIR"]; merged=os.environ["MERGED_DB"]; parts=os.environ["PARTS_CSV"].split(':')
src0=None
for p in parts:
    sp=os.path.join(md,"generated_molecules",p)
    if os.path.exists(sp): src0=sp; break
dist="Ang"; pu={}
if src0:
    try:
        con=sqlite3.connect(src0); meta=dict(con.execute("SELECT key,value FROM metadata").fetchall())
        dist=meta.get("distance_unit","Ang")
        if "_property_unit_dict" in meta: pu=json.loads(meta["_property_unit_dict"])
    except Exception as e: print("[PY][WARN] meta:",e)
ASEAtomsData.create(merged, distance_unit=dist, property_unit_dict=pu)
out=connect(merged); tot=0
for p in parts:
    sp=os.path.join(md,"generated_molecules",p)
    if not os.path.exists(sp): print("[PY][WARN] falta",sp); continue
    src=connect(sp)
    for row in src.select():
        at=row.toatoms(); data=dict(getattr(row,"data",{})); kv=dict(getattr(row,"key_value_pairs",{}))
        out.write(at, data=data, key_value_pairs=kv); tot+=1
print(f"[PY] Merge completo: {tot} -> {merged}")
PY
FILTERED_DB="${GEN_DIR}/filtered_${TS}.db"
ARGS=( "$MERGED_DB" "--timeout" "2" "--results_db_path" "$FILTERED_DB" "--results_db_flags" "unique" )
[[ -n "$TRAIN_DB" && -f "$TRAIN_DB" ]] && ARGS+=( "--compute_uniqueness" "--compare_db_path" "$TRAIN_DB" "--compare_db_split_path" "${MODELDIR}/split.npz" "--ignore_enantiomers" )
python "$CHECK_VALID" "${ARGS[@]}"
MULTI_XYZ="${MODELDIR}/exports/all_filtered_${TS}.xyz"; mkdir -p "$(dirname "$MULTI_XYZ")"
export FILTERED_DB MULTI_XYZ
python - << 'PY'
import os
from ase.db import connect
from ase.io import write
db=os.environ["FILTERED_DB"]; out=os.environ["MULTI_XYZ"]
rows=list(connect(db).select())
write(out,[r.toatoms() for r in rows], format="extxyz")
print(f"[PY] Multi-XYZ: {out} ({len(rows)})")
PY
EXP_DIR_XYZ="${MODELDIR}/exports_xyz/${TS}"; EXP_DIR_CIF="${MODELDIR}/exports_cif/${TS}"
mkdir -p "$EXP_DIR_XYZ" "$EXP_DIR_CIF"; export FILTERED_DB EXP_DIR_XYZ EXP_DIR_CIF
python - << 'PY'
import os
from ase.db import connect
from ase.io import write
db=connect(os.environ["FILTERED_DB"]); x=os.environ["EXP_DIR_XYZ"]; c=os.environ["EXP_DIR_CIF"]; n=0
for i,row in enumerate(db.select(), start=1):
    at=row.toatoms(); at.center(vacuum=8.0); at.set_pbc([False]*3)
    write(os.path.join(x,f"mol_{i:04d}.xyz"), at)
    write(os.path.join(c,f"mol_{i:04d}.cif"), at)
    n+=1
print(f"[PY] Exportadas {n} ->\n  - {x}\n  - {c}")
PY
echo "[INFO] OK:"
echo "  Merge:     ${MERGED_DB}"
echo "  Filtrado:  ${FILTERED_DB}"
echo "  MultiXYZ:  ${MULTI_XYZ}"
echo "  XYZ dir:   ${EXP_DIR_XYZ}"
echo "  CIF dir:   ${EXP_DIR_CIF}"
```

---

## 12) Apéndice B — `environment.yml` (opcional)

> Si prefieres crear el entorno con un archivo YAML:

```yaml
name: gschnet
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.4.1
  - torchvision=0.19.1
  - pytorch-cuda=12.1
  - pytorch-lightning=2.3.*
  - hydra-core=1.3
  - ase
  - rdkit
  - tqdm
  - torchmetrics>=1.3,<1.5
  - "numpy<2"
  - "sympy<=1.12"
  - pip
  - pip:
      - schnetpack==2.1.1
```

Luego:
```bash
conda env create -f environment.yml
conda activate gschnet
git clone https://github.com/atomistic-machine-learning/schnetpack-gschnet.git "$HOME/schnetpack-gschnet"
pip install "$HOME/schnetpack-gschnet"
```

---

**¡Listo!** Con esto tu estudiante debería poder replicar el flujo de trabajo completo (QM9) y adaptarlo a **OE62** o cualquier dataset propio.
