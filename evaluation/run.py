#!/usr/bin/env python3
"""Single evaluation entry point: offline preparation and real decoding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octopus.main import main

if __name__ == "__main__":
    main()
