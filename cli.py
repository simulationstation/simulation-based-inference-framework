#!/usr/bin/env python3
"""
Command-line interface for Analysis Pack Standard.

Provides commands for:
- Validating analysis packs
- Fitting models
- Computing limits/intervals
- Combining analyses
- Exporting results

Usage:
    aps validate <pack>
    aps fit <pack> --poi mu
    aps limit <pack> --poi mu --method asymptotic
    aps combine <packA> <packB> --correlations correlations.yaml
    aps export <pack> --format parquet/json
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np


def cmd_validate(args):
    """Validate an analysis pack."""
    import aps
    from aps.loaders import verify_pack_integrity

    pack_path = args.pack

    print(f"Validating pack: {pack_path}")
    print("=" * 60)

    try:
        pack = aps.load(pack_path, validate=True)
        print(f"Name: {pack.name}")
        print(f"Version: {pack.version}")
        print(f"Likelihood type: {pack.likelihood_type}")
        print(f"POIs: {pack.n_pois}")
        print(f"Nuisances: {pack.n_nuisances}")
        print(f"Channels: {list(pack.channels.keys())}")
        print()
        print("VALIDATION: PASSED")

        # Check integrity if hashes available
        valid, errors = verify_pack_integrity(pack)
        if errors:
            print("\nIntegrity warnings:")
            for e in errors:
                print(f"  - {e}")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except ValueError as e:
        print(f"VALIDATION: FAILED")
        print(f"Errors:\n{e}")
        return 1

    return 0


def cmd_fit(args):
    """Fit a model to find MLE."""
    import aps
    from runtime import compute_mle

    pack_path = args.pack

    print(f"Fitting pack: {pack_path}")
    print("=" * 60)

    try:
        pack = aps.load(pack_path)
        model = pack.model()

        print(f"Running MLE fit with {args.starts} random starts...")

        result = compute_mle(
            model,
            n_starts=args.starts,
            verbose=args.verbose
        )

        print("\nFit Results:")
        print("-" * 40)

        if result.converged:
            print(f"Status: CONVERGED")
            print(f"NLL: {result.nll:.4f}")
            print(f"Log-likelihood: {-result.nll:.4f}")

            print("\nPOI estimates:")
            for i, name in enumerate(model.poi_names):
                print(f"  {name}: {result.theta_hat[i]:.6f}")

            if model.n_nuisances > 0:
                print("\nNuisance estimates (showing first 5):")
                for i, name in enumerate(model.nuisance_names[:5]):
                    print(f"  {name}: {result.nu_hat[i]:.6f}")
                if model.n_nuisances > 5:
                    print(f"  ... and {model.n_nuisances - 5} more")

            if result.fit_health:
                print(f"\nFit health: {result.fit_health.status}")
                print(f"  chi2/dof: {result.fit_health.chi2_per_dof:.2f}")
        else:
            print("Status: FAILED")
            print(f"Error: {result.audit.get('error', 'Unknown')}")
            return 1

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0


def cmd_limit(args):
    """Compute upper limit on POI."""
    import aps
    from runtime import upper_limit, confidence_interval

    pack_path = args.pack

    print(f"Computing limit for pack: {pack_path}")
    print("=" * 60)

    try:
        pack = aps.load(pack_path)
        model = pack.model()

        # Find POI index
        poi_index = 0
        if args.poi:
            if args.poi in model.poi_names:
                poi_index = model.poi_names.index(args.poi)
            else:
                try:
                    poi_index = int(args.poi)
                except ValueError:
                    print(f"ERROR: Unknown POI '{args.poi}'")
                    return 1

        poi_name = model.poi_names[poi_index] if poi_index < len(model.poi_names) else f"poi_{poi_index}"

        print(f"POI: {poi_name} (index {poi_index})")
        print(f"Method: {args.method}")
        print(f"CL: {args.cl}")

        if args.method == "interval":
            lower, upper = confidence_interval(
                model,
                poi_index=poi_index,
                cl=args.cl,
                n_starts=args.starts
            )
            print(f"\n{args.cl*100:.0f}% confidence interval:")
            print(f"  [{lower:.4f}, {upper:.4f}]")
        else:
            result = upper_limit(
                model,
                poi_index=poi_index,
                cl=args.cl,
                method=args.method,
                n_starts=args.starts,
                verbose=args.verbose
            )

            print(f"\nObserved {args.cl*100:.0f}% CL upper limit:")
            print(f"  {poi_name} < {result.observed:.4f}")

            if result.pvalue is not None:
                print(f"\np-value (mu=0): {result.pvalue:.4f}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1

    return 0


def cmd_combine(args):
    """Combine multiple analysis packs."""
    import aps
    from runtime import combine_models, compute_mle

    print(f"Combining packs: {args.packs}")
    print("=" * 60)

    try:
        packs = [aps.load(p) for p in args.packs]
        models = [p.model() for p in packs]

        # Load correlation map if provided
        correlation_map = None
        if args.correlations:
            import yaml
            with open(args.correlations, 'r') as f:
                correlation_map = yaml.safe_load(f)

        combined = combine_models(
            models,
            correlation_map=correlation_map,
            conservative=not args.no_conservative
        )

        print(f"\nCombined model:")
        print(f"  Total POIs: {combined.n_pois}")
        print(f"  Total nuisances: {combined.n_nuisances}")

        if args.fit:
            print("\nFitting combined model...")
            result = compute_mle(combined, n_starts=args.starts)

            if result.converged:
                print(f"Combined NLL: {result.nll:.4f}")
                print(f"Fit health: {result.fit_health.status if result.fit_health else 'N/A'}")
            else:
                print("Fit FAILED")
                return 1

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0


def cmd_export(args):
    """Export pack data to different format."""
    import aps

    pack_path = args.pack
    output_path = args.output

    print(f"Exporting pack: {pack_path}")
    print(f"Format: {args.format}")
    print("=" * 60)

    try:
        pack = aps.load(pack_path)

        if args.format == "json":
            export_data = {
                'name': pack.name,
                'version': pack.version,
                'likelihood_type': pack.likelihood_type,
                'pois': [{'name': p.name, 'nominal': p.nominal} for p in pack.pois],
                'nuisances': [{'name': p.name, 'constraint': p.constraint} for p in pack.nuisances],
                'channels': {}
            }

            for name, ch in pack.channels.items():
                export_data['channels'][name] = {
                    'observable': ch.observable.tolist(),
                    'observed': ch.observed.tolist(),
                    'errors': ch.errors.tolist()
                }

            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                print(f"Exported to: {output_path}")
            else:
                print(json.dumps(export_data, indent=2))

        elif args.format == "parquet":
            try:
                import pandas as pd
            except ImportError:
                print("ERROR: pandas required for parquet export")
                return 1

            if not output_path:
                output_path = f"{pack.name}_export"

            os.makedirs(output_path, exist_ok=True)

            for name, ch in pack.channels.items():
                df = pd.DataFrame({
                    'observable': ch.observable,
                    'observed': ch.observed,
                    'errors': ch.errors
                })
                if ch.errors_up is not None:
                    df['errors_up'] = ch.errors_up
                if ch.errors_down is not None:
                    df['errors_down'] = ch.errors_down

                parquet_path = os.path.join(output_path, f"{name}.parquet")
                df.to_parquet(parquet_path)
                print(f"Exported channel {name} to: {parquet_path}")

        elif args.format == "csv":
            if not output_path:
                output_path = f"{pack.name}_export"

            os.makedirs(output_path, exist_ok=True)

            for name, ch in pack.channels.items():
                csv_path = os.path.join(output_path, f"{name}.csv")
                with open(csv_path, 'w') as f:
                    f.write("observable,observed,errors\n")
                    for obs, val, err in zip(ch.observable, ch.observed, ch.errors):
                        f.write(f"{obs},{val},{err}\n")
                print(f"Exported channel {name} to: {csv_path}")

        else:
            print(f"ERROR: Unknown format '{args.format}'")
            return 1

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0


def cmd_surrogate(args):
    """Build or evaluate surrogate model."""
    import aps
    from surrogates import SurrogateBuilder

    pack_path = args.pack

    if args.action == "build":
        print(f"Building surrogate for pack: {pack_path}")
        print("=" * 60)

        pack = aps.load(pack_path)
        model = pack.model()

        builder = SurrogateBuilder(
            model,
            surrogate_type=args.type
        )

        print(f"Generating {args.n_samples} training points...")
        builder.generate_training_data(
            n_samples=args.n_samples,
            method=args.sampling
        )

        print("Training surrogate...")
        surrogate = builder.build()

        print("\nValidation results:")
        if surrogate.validation:
            print(f"  MAE: {surrogate.validation.mae:.4f}")
            print(f"  RMSE: {surrogate.validation.rmse:.4f}")
            print(f"  Max error: {surrogate.validation.max_error:.4f}")
            print(f"  68% coverage: {surrogate.validation.coverage_68:.2f}")
            print(f"  95% coverage: {surrogate.validation.coverage_95:.2f}")
            print(f"  Passed: {surrogate.validation.passed}")

        if args.output:
            surrogate.save(args.output)
            print(f"\nSaved to: {args.output}")

    elif args.action == "validate":
        from surrogates.builder import load_surrogate
        from surrogates.calibration import validate_surrogate_thresholds

        print(f"Validating surrogate in: {pack_path}")

        pack = aps.load(pack_path)
        surrogate = load_surrogate(pack)

        print(f"Surrogate type: {type(surrogate).__name__}")
        print(f"Trained: {surrogate.is_trained}")

        if surrogate.validation:
            passed, failures = validate_surrogate_thresholds(
                surrogate,
                surrogate.training_data
            )

            if passed:
                print("Validation: PASSED")
            else:
                print("Validation: FAILED")
                for f in failures:
                    print(f"  - {f}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Analysis Pack Standard CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aps validate packs/atlas_xyz_v1
  aps fit packs/cms_x6900 --poi mu --starts 100
  aps limit packs/cms_x6900 --poi mu --cl 0.95
  aps combine packs/pack1 packs/pack2 --correlations corr.yaml
  aps export packs/cms_x6900 --format json -o output.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate an analysis pack')
    validate_parser.add_argument('pack', help='Path to analysis pack')

    # Fit command
    fit_parser = subparsers.add_parser('fit', help='Fit model to find MLE')
    fit_parser.add_argument('pack', help='Path to analysis pack')
    fit_parser.add_argument('--poi', help='POI name or index')
    fit_parser.add_argument('--starts', type=int, default=50, help='Number of optimizer starts')
    fit_parser.add_argument('-v', '--verbose', action='store_true')

    # Limit command
    limit_parser = subparsers.add_parser('limit', help='Compute upper limit')
    limit_parser.add_argument('pack', help='Path to analysis pack')
    limit_parser.add_argument('--poi', help='POI name or index')
    limit_parser.add_argument('--cl', type=float, default=0.95, help='Confidence level')
    limit_parser.add_argument('--method', default='asymptotic_cls',
                              choices=['asymptotic_cls', 'asymptotic_plr', 'interval'],
                              help='Limit method')
    limit_parser.add_argument('--starts', type=int, default=50)
    limit_parser.add_argument('-v', '--verbose', action='store_true')

    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Combine analysis packs')
    combine_parser.add_argument('packs', nargs='+', help='Paths to analysis packs')
    combine_parser.add_argument('--correlations', help='Correlation mapping YAML file')
    combine_parser.add_argument('--no-conservative', action='store_true',
                                help='Disable conservative combination for unknown correlations')
    combine_parser.add_argument('--fit', action='store_true', help='Fit combined model')
    combine_parser.add_argument('--starts', type=int, default=50)

    # Export command
    export_parser = subparsers.add_parser('export', help='Export pack data')
    export_parser.add_argument('pack', help='Path to analysis pack')
    export_parser.add_argument('--format', default='json',
                               choices=['json', 'parquet', 'csv'],
                               help='Export format')
    export_parser.add_argument('-o', '--output', help='Output path')

    # Surrogate command
    surrogate_parser = subparsers.add_parser('surrogate', help='Build or validate surrogate')
    surrogate_parser.add_argument('action', choices=['build', 'validate'])
    surrogate_parser.add_argument('pack', help='Path to analysis pack')
    surrogate_parser.add_argument('--type', default='gp',
                                  choices=['gp', 'neural', 'parametric'],
                                  help='Surrogate type')
    surrogate_parser.add_argument('--n-samples', type=int, default=500,
                                  help='Number of training samples')
    surrogate_parser.add_argument('--sampling', default='lhs',
                                  choices=['lhs', 'sobol', 'random'],
                                  help='Sampling method')
    surrogate_parser.add_argument('-o', '--output', help='Output path for surrogate')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Add package to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Dispatch to command handler
    if args.command == 'validate':
        return cmd_validate(args)
    elif args.command == 'fit':
        return cmd_fit(args)
    elif args.command == 'limit':
        return cmd_limit(args)
    elif args.command == 'combine':
        return cmd_combine(args)
    elif args.command == 'export':
        return cmd_export(args)
    elif args.command == 'surrogate':
        return cmd_surrogate(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
