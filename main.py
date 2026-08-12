"""Repo entrypoint: runs the full smoke/test array.

    python main.py

For training, use the package API (see README / notebooks):
    from geolip.alephllm import prepare
    run = prepare("mini-beatrix-1", hf_token=...)
    run.train()
"""
from geolip.alephllm.tests.smoke import main

if __name__ == "__main__":
    main()
