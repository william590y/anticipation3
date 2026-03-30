"""
Backward-compatible entrypoint for the old autoregressive evaluator name.

Current main uses packed ASAP-normalized defaults through `inference.py`.
"""

from inference import main


if __name__ == "__main__":
    main()
