"""After an agent writes candidates.json: pins, then the usage log.

    python asia1m/naming/finish.py <batch> <round> <agent transcript .jsonl>
"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
batch, rnd, transcript = sys.argv[1:4]
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
subprocess.run([sys.executable, os.path.join(HERE, "pins.py"), batch], check=True, env=env)
subprocess.run([sys.executable, os.path.join(HERE, "usage.py"), batch, rnd, transcript], check=True, env=env)
subprocess.run([sys.executable, os.path.join(HERE, "dupes.py")], check=True, env=env)
