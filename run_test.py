import signal
import sys
import os
import psutil
from src.test import main


def shutdown(signum, frame):
    print("\n⏹ STOP ALL PROCESSES", file=sys.stderr)
    pid = os.getpid()
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            print(f"  - Killing child PID {child.pid}", file=sys.stderr)
            try:
                child.kill()
            except Exception as e:
                print(f"    (Already terminated or cannot kill: {e})", file=sys.stderr)
        parent.kill()
    except Exception as e:
        print(f"Error killing processes: {e}", file=sys.stderr)
    sys.exit(1)


for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, shutdown)

if __name__ == "__main__":
    main()
