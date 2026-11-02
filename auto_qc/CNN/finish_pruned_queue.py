"""Record an intentional queue cutoff after a legacy worker reaches its guard.

This monitor never starts or stops training. A worker that cached its queue before
the user narrowed its scope rejects the first deferred config during preflight.
Only that specific, expected rejection is recorded as completion with skips.
Other failures remain failures.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

MARKER = 'USER_DEFERRED_EARLY_RERUN'


def finish_state(queue, state):
    deferred = {entry['config']: entry for entry in queue['entries']
                if entry['status'] == 'deferred_by_user'}
    if (state['status'] != 'failed' or MARKER not in state.get('error', '')
            or state.get('active') not in deferred
            or not state['remaining']
            or not set(state['remaining']).issubset(deferred)):
        raise ValueError('Not the expected deferred-only queue cutoff; leaving status unchanged')
    result = copy.deepcopy(state)
    result['skipped'] = result.get('skipped', []) + [
        {'config': path, 'reason': deferred[path]['reasons']} for path in result['remaining']]
    result['intentional_stop_detail'] = {
        'error': result.pop('error'), 'traceback': result.pop('traceback', None),
        'note': 'User excluded older exploratory reruns; no deferred experiment trained.'}
    result.update(status='completed', active=None, remaining=[],
                  scope_completed_at_utc=datetime.now(timezone.utc).isoformat())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--queue-status', type=Path, required=True)
    args = parser.parse_args()
    while True:
        state = json.loads(args.queue_status.read_text(encoding='utf-8'))
        if state['status'] != 'running':
            break
        time.sleep(30)
    if state['status'] == 'completed':
        return
    queue = json.loads(args.queue.read_text(encoding='utf-8'))
    result = finish_state(queue, state)
    temporary = args.queue_status.with_suffix('.pruned.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.queue_status)
    print(f"Selected scope completed; {len(result['skipped'])} older reruns skipped.", flush=True)


if __name__ == '__main__':
    main()
