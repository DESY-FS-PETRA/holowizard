import uuid


def submit_and_handle(task_name, func, cluster, scan, *args, priority=100, key=None):
    try:
        future = cluster.client_scheduler.submit(
            func,
            scan,
            *args,
            priority=priority,
            key=key or f"{task_name}-{uuid.uuid4()}-{scan.key}",
            pure=False,
        )
        result = future.result()
        status = "cancelled" if scan.cancelled else "done"
        return result, status
    except Exception as e:
        print(f"{task_name} failed for scan {scan.key}: {e}")
        if task_name != "find_focus":
            scan.cancel()
        if scan.cancelled:
            status = "cancelled"
        else:
            status = "failed"
        return None, "status"
    finally:
        future.release()
