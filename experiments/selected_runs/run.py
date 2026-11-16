"""Run only the reviewed shortlist, sequentially; keep queue bookkeeping here."""
import argparse
import csv
from pathlib import Path

from auto_qc.CNN.training import train
from auto_qc.CNN.utils import ROOT

MANIFEST = Path(__file__).with_name('selection.csv')


def save(rows, manifest):
    temporary = manifest.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--manifest', type=Path, default=MANIFEST,
                        help='CSV shortlist to execute; completed rows are skipped')
    args = parser.parse_args()
    with args.manifest.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if args.check_only:
            train(ROOT / row['config'], check_only=True)
            continue
        if row['status'] == 'completed':
            continue
        if row['status'] == 'running':
            raise RuntimeError('A shortlisted run is already marked running; check its process before restarting.')
        row.update(status='running', error='')
        save(rows, args.manifest)
        print(f"START {row['model']} #{row['rank']}: {row['config']}", flush=True)
        try:
            output = train(ROOT / row['config'])
            row.update(status='completed', output=str(output))
            print(f'FINISHED {output}', flush=True)
        except BaseException as error:
            row.update(status='failed', error=str(error))
            raise
        finally:
            save(rows, args.manifest)


if __name__ == '__main__':
    main()
