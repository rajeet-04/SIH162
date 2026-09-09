"""Isolated retraining commands; never overwrite the deployed model bundle."""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from thermis.retrain_audit import audit_sources
from thermis.retrain_data import MODEL_FEATURES, prepare_examples, read_observations
from thermis.retrain_sources import fetch_historical_facilities, recover_archive


def validate_training_output(path: Path) -> None:
    resolved = path.resolve()
    if 'runs' not in resolved.parts or resolved.name in ('runs', 'tabular', 'image'):
        raise ValueError('Training output must be a versioned child of models/runs or data/runs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit', help='checksum and inspect configured source files')
    audit.add_argument('--config', type=Path, required=True)
    audit.add_argument('--output', type=Path, required=True)
    osm = commands.add_parser('fetch-osm', help='fetch historical industrial study-area context')
    osm.add_argument('--output', type=Path, required=True)
    osm.add_argument('--as-of', default='2026-06-01T00:00:00Z')
    osm.add_argument('--reuse-from', type=Path,
                     help='reuse successful area responses from the same historical snapshot')
    recover = commands.add_parser('recover-flaresat', help='recover original-provider imagery')
    recover.add_argument('--kind', choices=['fire', 'flare'], required=True)
    recover.add_argument('--output', type=Path, required=True)
    prepare = commands.add_parser('prepare', help='build chronological, proxy-labeled examples')
    prepare.add_argument('--database', type=Path, required=True)
    prepare.add_argument('--flare-catalog', type=Path, required=True)
    prepare.add_argument('--flare-available-from', default='2025-11-15T00:00:00Z',
                         help='source publication/availability, not acquisition date')
    prepare.add_argument('--facilities', type=Path)
    prepare.add_argument('--output', type=Path, required=True)
    train = commands.add_parser('train', help='train isolated CatBoost and PyTorch challengers')
    train.add_argument('--examples', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='cuda')
    train.add_argument('--epochs', type=int, default=20)
    train.add_argument('--batch-size', type=int, default=256)
    train.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'audit':
        config = yaml.safe_load(args.config.read_text())
        report = audit_sources(config['sources'], args.output)
    elif args.command == 'fetch-osm':
        report = fetch_historical_facilities(args.output, args.as_of, reuse_from=args.reuse_from)
    elif args.command == 'recover-flaresat':
        md5 = {'fire': '3872c9e98561eda7c8b362cc18b14f65',
               'flare': '9ab78eba28fa5c87d2378ac52a32ed7a'}[args.kind]
        report = recover_archive(
            f'https://zenodo.org/records/17619196/files/{args.kind}_patches.zip',
            args.output / f'{args.kind}_patches.zip', md5)
    elif args.command == 'prepare':
        flares = pd.read_csv(args.flare_catalog)
        flares['available_from'] = args.flare_available_from
        facilities = pd.read_parquet(args.facilities) if args.facilities else pd.DataFrame()
        if 'kind' in facilities:
            facilities = facilities[facilities.kind == 'industrial']
        report = prepare_examples(read_observations(args.database), args.output, flares, facilities)
    else:
        validate_training_output(args.output)
        from thermis.retrain_models import train_challengers
        report = train_challengers(pd.read_parquet(args.examples), args.output,
                                   list(MODEL_FEATURES), epochs=args.epochs,
                                   batch_size=args.batch_size, device=args.device,
                                   resume=args.resume)
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
