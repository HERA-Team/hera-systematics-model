"""Checked child-process execution with retained-output resource monitoring."""

import os
import signal
import subprocess
import time

from .production import require_storage, retained_bytes


def stop_process_group(process):
    """Stop descendants as well as the immediate wrapper after a failed limit."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute_checked(command, directory, environment, stdout, stderr, timeout, check_limits, interval=10.):
    """Poll limits and preserve all outputs when a command or resource check fails."""
    check_limits()
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=directory, env=environment, stdout=stdout,
                               stderr=stderr, start_new_session=True)
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                code = process.wait(timeout=min(interval, remaining))
                check_limits()
                return code
            except subprocess.TimeoutExpired:
                check_limits()
    except BaseException:
        stop_process_group(process)
        raise


def storage_monitor(root, directory, projected_bytes, evidence):
    """Enforce the measured task reservation and global cap while writing."""
    def check():
        storage = require_storage(root, 0)
        task_bytes = retained_bytes(directory)
        evidence["checks"] = evidence.get("checks", 0) + 1
        evidence["maximum_sampled_retained_bytes"] = max(evidence.get("maximum_sampled_retained_bytes", 0), storage["retained_bytes"])
        evidence["maximum_sampled_task_bytes"] = max(evidence.get("maximum_sampled_task_bytes", 0), task_bytes)
        evidence["task_reservation_with_contingency"] = int(projected_bytes * 1.2 + .5)
        if task_bytes > evidence["task_reservation_with_contingency"]:
            raise ValueError("task retained products exceed measured reservation with contingency")
    return check
