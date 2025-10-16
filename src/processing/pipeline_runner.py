"""
File: src/processing/pipeline_runner.py

Pipeline runner that executes feature engineering from YAML configuration files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from src.data.psql import Psql
from src.processing.feature_store import FeatureStore
from src.processing.feature_pipeline import FeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Executes feature engineering pipelines from YAML configuration files.

    Example:
        runner = PipelineRunner('config/pipelines/basic_pipeline.yaml')
        results = runner.run()
    """

    def __init__(self, config_path: str):
        """
        Initialize pipeline runner with config file

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loaded configuration: {self.config.get('name', 'unnamed')}")
        logger.info(f"Description: {self.config.get('description', 'N/A')}")

        # Initialize components
        self.psql_client = None
        self.feature_store = None
        self.pipeline = None
        self.results = {}
        
        self.name = self.config["name"] + "_" + "_".join(self.config["version"].split("."))

    def _setup_database(self):
        """Setup database connection"""
        logger.info("Connecting to database...")
        db_config = self.config.get('database', {})

        if db_config.get('use_env', True):
            # Use environment variables
            self.psql_client = Psql()
        else:
            # Override with config (not recommended for security)
            # Would need to modify Psql class to accept params
            raise NotImplementedError("Direct database config not yet supported")

        logger.info("✓ Database connected")

    def _setup_feature_store(self):
        """Setup feature store"""
        cache_config = self.config.get('cache', {})
        cache_dir = cache_config.get('directory', './feature_cache')

        logger.info(f"Initializing FeatureStore (cache: {cache_dir})...")

        # Create versioned cache directory
        version = self.config.get('version', 'v1.0')
        data_version = f"{self.config.get('name', 'pipeline')}_{version}_{datetime.now().strftime('%Y%m%d')}"

        self.feature_store = FeatureStore(
            psql_client=self.psql_client,
            cache_dir=cache_dir,
            data_version=data_version
        )

        # Clear cache if requested
        if cache_config.get('clear_before_run', False):
            logger.info("Clearing cache before run...")
            self.feature_store.clear_cache()

        logger.info("✓ FeatureStore initialized")

    def _setup_pipeline(self):
        """Setup feature pipeline from config"""
        logger.info("Configuring FeaturePipeline...")

        self.pipeline = FeaturePipeline(feature_store=self.feature_store)

        # Add feature groups from config
        feature_groups = self.config.get('feature_groups', [])

        for group_config in feature_groups:
            name = group_config['name']
            features = group_config['features']
            params = group_config.get('params', {})
            preprocessing = group_config.get('preprocessing', {})

            self.pipeline.add_feature_group(
                name=name,
                features=features,
                params=params,
                log_transform=preprocessing.get('log_transform', False),
                scaling_method=preprocessing.get('scaling_method'),
                impute_strategy=preprocessing.get('impute_strategy'),
                drop_if_null_pct=preprocessing.get('drop_if_null_pct')
            )

            logger.info(f"  ✓ Added feature group '{name}' with {len(features)} features")

        logger.info(f"✓ Pipeline configured with {len(feature_groups)} feature groups")

    def _get_user_ids(self) -> Optional[list]:
        """Get user IDs to process based on config"""
        data_config = self.config.get('data', {})

        user_ids = data_config.get('user_ids')
        sample_size = data_config.get('sample_size')

        if user_ids:
            logger.info(f"Using {len(user_ids)} specified user IDs")
            return user_ids

        if sample_size:
            logger.info(f"Sampling {sample_size} random users...")
            # Query to get random users
            query = f"""
            SELECT DISTINCT userId
            FROM Yt.Comments
            GROUP BY userId
            HAVING COUNT(*) >= %s
            ORDER BY RANDOM()
            LIMIT %s;
            """
            min_comments = data_config.get('min_comments', 5)
            result = self.psql_client.query(query, (min_comments, sample_size))
            user_ids = [row[0] for row in result]
            logger.info(f"  ✓ Sampled {len(user_ids)} users")
            return user_ids

        return None  # Process all users

    def _run_pipeline(self) -> Dict[str, pd.DataFrame]:
        """Execute the pipeline"""
        logger.info("Running feature pipeline...")

        cache_enabled = self.config.get('cache', {}).get('enabled', True)
        user_ids = self._get_user_ids()

        # Check if train/test split is enabled
        split_config = self.config.get('split', {})
        if split_config.get('enabled', False):
            logger.info("Train/test split enabled")

            # Fit on all data first to get users
            X_all = self.pipeline.fit_transform(user_ids=user_ids, use_cache=cache_enabled)

            if len(X_all) == 0 or len(X_all.columns) == 1:
                logger.warning("No features produced! Check your configuration.")
                return {}

            # Split users
            from sklearn.model_selection import train_test_split

            all_user_ids = X_all['userId'].tolist()
            test_size = split_config.get('test_size', 0.2)
            random_state = split_config.get('random_state', 42)

            train_ids, test_ids = train_test_split(
                all_user_ids,
                test_size=test_size,
                random_state=random_state
            )

            logger.info(f"  Train users: {len(train_ids)}")
            logger.info(f"  Test users: {len(test_ids)}")

            # Refit on train only
            logger.info("Refitting pipeline on training data...")
            self._setup_pipeline()  # Recreate pipeline
            X_train = self.pipeline.fit_transform(user_ids=train_ids, use_cache=cache_enabled)
            X_test = self.pipeline.transform(user_ids=test_ids, use_cache=cache_enabled)

            logger.info(f"  ✓ X_train: {X_train.shape}")
            logger.info(f"  ✓ X_test: {X_test.shape}")

            return {
                'train': X_train,
                'test': X_test,
                'all': X_all
            }
        else:
            # No split - return all data
            X = self.pipeline.fit_transform(user_ids=user_ids, use_cache=cache_enabled)
            logger.info(f"  ✓ Features: {X.shape}")

            return {'all': X}

    def _save_outputs(self, results: Dict[str, pd.DataFrame]):
        """Save pipeline outputs"""
        output_config = self.config.get('output', {})
        output_dir = Path(output_config.get('directory', f'./data/{self.name}'))
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving outputs to {output_dir}...")

        formats = output_config.get('formats', ['csv'])
        pipeline_name = self.config.get('name', 'pipeline')

        for key, df in results.items():
            if df is None or len(df) == 0:
                continue

            base_name = f"{pipeline_name}_{key}"

            # Save in requested formats
            for fmt in formats:
                if fmt == 'csv':
                    output_path = output_dir / f"{base_name}.csv"
                    df.to_csv(output_path, index=False)
                    logger.info(f"  ✓ Saved: {output_path}")
                elif fmt == 'parquet':
                    output_path = output_dir / f"{base_name}.parquet"
                    df.to_parquet(output_path, index=False)
                    logger.info(f"  ✓ Saved: {output_path}")

        # Save metadata
        if output_config.get('save_metadata', True):
            metadata_path = output_dir / f"{pipeline_name}_metadata.yaml"
            metadata = {
                'pipeline_name': self.config.get('name'),
                'version': self.config.get('version'),
                'description': self.config.get('description'),
                'run_timestamp': datetime.now().isoformat(),
                'config_path': str(self.config_path),
                'feature_groups': [fg['name'] for fg in self.config.get('feature_groups', [])],
                'shapes': {k: list(v.shape) for k, v in results.items() if v is not None}
            }
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f)
            logger.info(f"  ✓ Saved metadata: {metadata_path}")

        # Save feature profile
        if output_config.get('save_profile', True):
            if 'all' in results:
                profile = self.feature_store.profile_features(results['all'])
                profile_path = output_dir / f"{pipeline_name}_profile.csv"
                profile.to_csv(profile_path, index=False)
                logger.info(f"  ✓ Saved profile: {profile_path}")

    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the full pipeline

        Returns:
            Dictionary of DataFrames (train, test, all)
        """
        try:
            logger.info("=" * 80)
            logger.info(f"STARTING PIPELINE: {self.config.get('name', 'unnamed')}")
            logger.info("=" * 80)

            # Setup
            self._setup_database()
            self._setup_feature_store()
            self._setup_pipeline()

            # Run
            results = self._run_pipeline()

            # Save
            if results:
                self._save_outputs(results, )

            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

        finally:
            # Cleanup
            if self.psql_client:
                self.psql_client.close_db()
                logger.info("Database connection closed")


def run_pipeline_from_config(config_path: str) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to run pipeline from config file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary of DataFrames
    """
    runner = PipelineRunner(config_path)
    return runner.run()
