#!/usr/bin/env bash
# builds the patched DeepSeek-V2-Lite modelling package that
# scripts/bench_deepseek.py imports as `dsmod`.
#
#   scripts/setup_dsmod.sh            # goes to ~/dsmod
#   scripts/setup_dsmod.sh /path/to   # or wherever
#
# why this is needed: the checkpoint ships its own modelling code through
# trust_remote_code, and that code does `import flash_attn` at module scope with
# no guard. if flash-attention isn't built against your local cuda, the import
# blows up before anything runs. we never take the code path that uses it, so we
# grab a local copy and wrap the import in a try. nothing else changes.
set -euo pipefail

DEST="${1:-$HOME/dsmod}"
CKPT="deepseek-ai/DeepSeek-V2-Lite"

if [ -d "$DEST" ]; then
  echo "$DEST already exists; remove it first to rebuild." >&2
  exit 1
fi

echo "fetching modelling code from $CKPT ..."
mkdir -p "$DEST"
python - "$CKPT" "$DEST" <<'PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

ckpt, dest = sys.argv[1], Path(sys.argv[2])
for fn in ("modeling_deepseek.py", "configuration_deepseek.py"):
    src = hf_hub_download(repo_id=ckpt, filename=fn)
    shutil.copy(src, dest / fn)
    print("  ", fn)
(dest / "__init__.py").write_text("")
PY

echo "patching the unconditional flash_attn import ..."
python - "$DEST" <<'PY'
import re, sys
from pathlib import Path

p = Path(sys.argv[1]) / "modeling_deepseek.py"
s = p.read_text(encoding="utf-8")
before = s

# Guard the module-scope import instead of deleting it, so the code path that
# genuinely wants flash-attention still works where it is available.
s = re.sub(
    r"^(\s*)(from flash_attn.*|import flash_attn.*)$",
    r"\1try:\n\1    \2\n\1except ImportError:  # patched: not required for the paths we take\n\1    pass",
    s, flags=re.M)

if s == before:
    print("  no flash_attn import found -- upstream may have changed; check manually")
else:
    p.write_text(s, encoding="utf-8")
    print("  patched", p)
PY

echo
echo "done: $DEST"
echo "run:  scripts/run_deepseek.sh --check"
