import argparse


def main():
    parser = argparse.ArgumentParser(description='Environmental sound experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Extract features and log-mel spectrograms from .ogg audio')
    prep.add_argument('--root', default='.')
    prep.add_argument('--output', default='artifacts/audio.npz')
    bench = sub.add_parser('benchmark', help='Nested cross-validation of classical models')
    bench.add_argument('--models', nargs='+', choices=['dummy', 'logistic', 'svm', 'forest', 'extra_trees'])
    bench.add_argument('--jobs', type=int, default=1)
    cnn = sub.add_parser('cnn', help='Cross-validated spectrogram CNN')
    cnn.add_argument('--epochs', type=int, default=40)
    cnn.add_argument('--patience', type=int, default=8)
    cnn.add_argument('--threads', type=int, default=2)
    cnn.add_argument('--no-augment', action='store_true')
    for p, default in [(bench, 'reports/baselines.json'), (cnn, 'reports/cnn.json')]:
        p.add_argument('--cache', default='artifacts/audio.npz')
        p.add_argument('--output', default=default)
        p.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'prepare':
        from .data import prepare
        meta = prepare(args.root, args.output)
        print(f"Prepared {len(meta['clips'])} clips across {len(meta['classes'])} classes")
    elif args.command == 'benchmark':
        from .benchmark import run
        run(args.cache, args.output, args.models, args.seed, args.jobs)
    else:
        from .cnn import run
        run(args.cache, args.output, args.seed, args.epochs, args.patience, not args.no_augment, args.threads)


if __name__ == '__main__':
    main()
