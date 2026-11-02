#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# BreakHis acquisition — run this on your own machine (it needs open internet).
#
#   bash get_breakhis.sh [TARGET_DIR]
#
# Default TARGET_DIR: ./data
# Tries mirrors in order and stops at the first that works.
# ---------------------------------------------------------------------------
set -u
TARGET="${1:-./data}"
mkdir -p "$TARGET"
cd "$TARGET" || exit 1
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

ok=0

echo "==> [1/3] Kaggle CLI (most reliable)"
if command -v kaggle >/dev/null 2>&1; then
  if kaggle datasets download -d ambarish/breakhis -p . --unzip; then
    ok=1
  else
    echo "    Kaggle CLI failed. Is ~/.kaggle/kaggle.json in place?"
    echo "    Get it from kaggle.com -> Settings -> API -> Create New Token"
  fi
else
  echo "    kaggle CLI not installed.  pip install kaggle"
fi

if [ "$ok" -eq 0 ]; then
  echo "==> [2/3] Official UFPR mirror"
  for U in "https://web.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz" \
           "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"; do
    echo "    trying $U"
    if curl -fL --retry 3 --retry-delay 5 -A "$UA" -o BreaKHis_v1.tar.gz "$U"; then
      tar -xzf BreaKHis_v1.tar.gz && rm -f BreaKHis_v1.tar.gz && ok=1 && break
    fi
  done
fi

if [ "$ok" -eq 0 ]; then
  echo "==> [3/3] All automated routes failed."
  cat <<'MSG'

    Download manually from either:
      https://www.kaggle.com/datasets/ambarish/breakhis
      https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

    Then extract into this directory so that the tree contains:
      BreaKHis_v1/histology_slides/breast/{benign,malignant}/SOB/...
MSG
  exit 1
fi

echo
echo "==> Verifying"
found=$(find . -name "SOB_*.png" | wc -l | tr -d ' ')
echo "    PNG files matching the BreakHis grammar: $found  (expected 7909)"
[ "$found" -eq 7909 ] && echo "    COMPLETE" || echo "    WARNING: count differs — this must be reported in the paper's dataset section."
echo "    Location: $(pwd)"
