"""
File: scripts/processing/run_pipeline.py

Script to run feature engineering pipelines from YAML configuration files.

Usage:
    python scripts/run_pipeline.py config/pipelines/basic_pipeline.yaml
    OR
    docker-compose build feature-pipeline 
        && docker-compose run --rm feature-pipeline config/pipelines/basic_pipeline.yaml

Options:
    --dry-run    Show what would be done without executing
    --list       List available pipeline configurations
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Override for local Postgres if needed (only when NOT in Docker)
if os.getenv("POSTGRES_PORT_LOCAL") and not os.path.exists("/.dockerenv"):
    os.environ["POSTGRES_PORT"] = os.getenv("POSTGRES_PORT_LOCAL", "5555")
    os.environ["POSTGRES_HOST"] = "localhost"

from src.processing.pipeline_runner import PipelineRunner
import yaml

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def list_pipelines():
    """List all available pipeline configurations"""
    config_dir = project_root / 'config' / 'pipelines'

    if not config_dir.exists():
        logger.info("No pipeline configurations found.")
        return

    logger.info("Available pipeline configurations:\n")

    for config_file in sorted(config_dir.glob('*.yaml')):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        name = config.get('name', config_file.stem)
        version = config.get('version', 'N/A')
        description = config.get('description', 'No description')

        logger.info(f"  {config_file.name}")
        logger.info(f"    Name: {name}")
        logger.info(f"    Version: {version}")
        logger.info(f"    Description: {description}")
        logger.info()


def dry_run(config_path: Path):
    """Show what the pipeline would do without executing"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("DRY RUN - Configuration Summary")
    logger.info("=" * 80)
    logger.info(f"\nPipeline: {config.get('name', 'unnamed')}")
    logger.info(f"Version: {config.get('version', 'N/A')}")
    logger.info(f"Description: {config.get('description', 'N/A')}")

    logger.info("\nCache Configuration:")
    cache = config.get('cache', {})
    logger.info(f"  Enabled: {cache.get('enabled', True)}")
    logger.info(f"  Directory: {cache.get('directory', './feature_cache')}")
    logger.info(f"  Clear before run: {cache.get('clear_before_run', False)}")

    logger.info("\nData Configuration:")
    data = config.get('data', {})
    logger.info(f"  Min comments: {data.get('min_comments', 5)}")
    logger.info(f"  User IDs: {data.get('user_ids', 'All')}")
    logger.info(f"  Sample size: {data.get('sample_size', 'All')}")

    logger.info("\nFeature Groups:")
    for group in config.get('feature_groups', []):
        logger.info(f"  - {group['name']}")
        logger.info(f"    Features: {', '.join(group['features'])}")
        preprocessing = group.get('preprocessing', {})
        logger.info(f"    Scaling: {preprocessing.get('scaling_method', 'None')}")
        logger.info(f"    Imputation: {preprocessing.get('impute_strategy', 'None')}")

    logger.info("\nTrain/Test Split:")
    split = config.get('split', {})
    if split.get('enabled', False):
        logger.info(f"  Enabled: Yes")
        logger.info(f"  Test size: {split.get('test_size', 0.2)}")
        logger.info(f"  Random state: {split.get('random_state', 42)}")
    else:
        logger.info(f"  Enabled: No")

    logger.info("\nOutput Configuration:")
    output = config.get('output', {})
    logger.info(f"  Directory: {output.get('directory', './data/processed')}")
    logger.info(f"  Formats: {', '.join(output.get('formats', ['csv']))}")
    logger.info(f"  Save metadata: {output.get('save_metadata', True)}")
    logger.info(f"  Save profile: {output.get('save_profile', True)}")

    logger.info("\n" + "=" * 80)
    logger.info("This was a dry run. Use without --dry-run to execute.")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run feature engineering pipeline from YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic pipeline
  python scripts/run_pipeline.py config/pipelines/basic_pipeline.yaml

  # Run with dry-run to preview
  python scripts/run_pipeline.py config/pipelines/full_pipeline.yaml --dry-run

  # List available pipelines
  python scripts/run_pipeline.py --list
        """
    )

    parser.add_argument(
        'config',
        nargs='?',
        help='Path to pipeline configuration YAML file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available pipeline configurations'
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_pipelines()
        return 0

    # Require config file if not listing
    if not args.config:
        parser.print_help()
        return 1

    config_path = Path(args.config)

    if not config_path.exists():
        logger.error(f"Error: Config file not found: {config_path}")
        return 1

    # Handle --dry-run
    if args.dry_run:
        dry_run(config_path)
        return 0

    # Run the pipeline
    try:
        runner = PipelineRunner(str(config_path))
        results = runner.run()

        logger.info("\n✓ Pipeline completed successfully!")
        logger.info(f"\nResults:")
        for key, df in results.items():
            if df is not None:
                logger.info(f"  {key}: {df.shape}")

        return 0

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
