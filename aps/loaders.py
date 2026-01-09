"""
Pack loading and validation utilities for Analysis Pack Standard.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

import yaml
import numpy as np

from .schemas import (
    METADATA_SCHEMA,
    LIKELIHOOD_SCHEMA,
    CONSTRAINTS_SCHEMA,
    PARAMETER_SCHEMA,
    validate_against_schema
)


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""
    name: str
    description: str = ""
    range: Tuple[float, float] = (-np.inf, np.inf)
    nominal: float = 0.0
    unit: str = ""
    constraint: str = "none"  # normal, lognormal, poisson, uniform, fixed
    sigma: float = 1.0
    transform: str = "none"  # none, log, logit


@dataclass
class ChannelData:
    """Data for a single analysis channel."""
    name: str
    observable: np.ndarray  # e.g., mass bins
    observed: np.ndarray    # observed counts/cross-sections
    errors: np.ndarray      # statistical errors
    errors_up: Optional[np.ndarray] = None   # asymmetric upper errors
    errors_down: Optional[np.ndarray] = None # asymmetric lower errors
    systematic_errors: Optional[np.ndarray] = None

    @property
    def nbins(self) -> int:
        return len(self.observed)

    @property
    def has_asymmetric_errors(self) -> bool:
        return self.errors_up is not None and self.errors_down is not None


@dataclass
class AnalysisPack:
    """
    Loaded analysis pack with all components.

    This is the main object for interacting with an analysis.
    """
    path: Path
    metadata: Dict[str, Any]
    likelihood_spec: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    parameter_schema: Optional[Dict[str, Any]] = None

    # Loaded data
    channels: Dict[str, ChannelData] = field(default_factory=dict)
    pois: List[ParameterSpec] = field(default_factory=list)
    nuisances: List[ParameterSpec] = field(default_factory=list)

    # Validation state
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.metadata.get("name", "unknown")

    @property
    def version(self) -> str:
        return self.metadata.get("version", "0.0.0")

    @property
    def likelihood_type(self) -> str:
        return self.likelihood_spec.get("type", "unknown")

    @property
    def n_pois(self) -> int:
        return len(self.pois)

    @property
    def n_nuisances(self) -> int:
        return len(self.nuisances)

    @property
    def n_params(self) -> int:
        return self.n_pois + self.n_nuisances

    def get_poi_names(self) -> List[str]:
        return [p.name for p in self.pois]

    def get_nuisance_names(self) -> List[str]:
        return [p.name for p in self.nuisances]

    def model(self):
        """
        Create an executable likelihood model from this pack.

        Returns a LikelihoodModel that can evaluate logpdf, profile, etc.
        """
        try:
            from runtime import create_model
        except ImportError:
            from ..runtime import create_model
        return create_model(self)


def load(pack_path: str, validate: bool = True) -> AnalysisPack:
    """
    Load an analysis pack from a directory.

    Args:
        pack_path: Path to the pack directory
        validate: Whether to run validation checks

    Returns:
        AnalysisPack object

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If validation fails
    """
    path = Path(pack_path)

    if not path.exists():
        raise FileNotFoundError(f"Pack directory not found: {pack_path}")

    # Load metadata.yaml (required)
    metadata_path = path / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing required file: metadata.yaml in {pack_path}")

    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)

    # Load model/likelihood.json (required)
    likelihood_path = path / "model" / "likelihood.json"
    if not likelihood_path.exists():
        raise FileNotFoundError(f"Missing required file: model/likelihood.json in {pack_path}")

    with open(likelihood_path, 'r') as f:
        likelihood_spec = json.load(f)

    # Load optional files
    constraints = None
    constraints_path = path / "model" / "constraints.json"
    if constraints_path.exists():
        with open(constraints_path, 'r') as f:
            constraints = json.load(f)

    parameter_schema = None
    schema_path = path / "model" / "schema.json"
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            parameter_schema = json.load(f)

    # Create pack object
    pack = AnalysisPack(
        path=path,
        metadata=metadata,
        likelihood_spec=likelihood_spec,
        constraints=constraints,
        parameter_schema=parameter_schema
    )

    # Parse parameters
    pack.pois = _parse_parameters(metadata.get("parameters", {}).get("pois", []))
    pack.nuisances = _parse_parameters(metadata.get("parameters", {}).get("nuisances", []))

    # Load channel data
    pack.channels = _load_channels(path, likelihood_spec)

    # Validate if requested
    if validate:
        is_valid, errors = validate_pack(pack)
        pack.validated = is_valid
        pack.validation_errors = errors
        if not is_valid:
            raise ValueError(f"Pack validation failed:\n" + "\n".join(errors))

    return pack


def _parse_parameters(param_list: List[Dict]) -> List[ParameterSpec]:
    """Parse parameter specifications from metadata."""
    params = []
    for p in param_list:
        range_vals = p.get("range", [-np.inf, np.inf])
        params.append(ParameterSpec(
            name=p["name"],
            description=p.get("description", ""),
            range=(range_vals[0], range_vals[1]),
            nominal=p.get("nominal", 0.0),
            unit=p.get("unit", ""),
            constraint=p.get("constraint", "none"),
            sigma=p.get("sigma", 1.0),
            transform=p.get("transform", "none")
        ))
    return params


def _load_channels(pack_path: Path, likelihood_spec: Dict) -> Dict[str, ChannelData]:
    """Load data for all channels specified in likelihood."""
    channels = {}

    for ch_spec in likelihood_spec.get("channels", []):
        name = ch_spec["name"]
        data_spec = ch_spec.get("data", {})

        # Determine file path
        data_file = data_spec.get("file")
        if data_file:
            data_path = pack_path / "data" / data_file
            if data_path.exists():
                channels[name] = _load_channel_data(name, data_path, data_spec)

    return channels


def _load_channel_data(name: str, path: Path, spec: Dict) -> ChannelData:
    """Load data for a single channel from CSV."""
    data_format = spec.get("format", "csv")

    if data_format == "csv":
        return _load_csv_channel(name, path, spec)
    elif data_format == "parquet":
        return _load_parquet_channel(name, path, spec)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")


def _load_csv_channel(name: str, path: Path, spec: Dict) -> ChannelData:
    """Load channel data from CSV file."""
    # Read CSV header to determine format
    with open(path, 'r') as f:
        header = f.readline().strip().split(',')

    data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=float)

    # Handle different CSV formats
    if len(header) >= 3:
        if 'mass_GeV' in header[0] or 'sqrt_s' in header[0].lower():
            # CMS/BESIII format
            observable = data[:, 0]

            # Find counts/sigma column
            if 'counts' in header or len(header) == 3:
                observed = data[:, 1]
                errors = data[:, 2]
                errors_up = None
                errors_down = None
                syst = None
            elif 'sigma_pb' in header:
                # BESIII cross-section format
                sigma_col = header.index('sigma_pb')
                observed = data[:, sigma_col]

                # Look for stat errors
                if 'sigma_stat_up_pb' in header:
                    up_col = header.index('sigma_stat_up_pb')
                    down_col = header.index('sigma_stat_down_pb')
                    errors_up = data[:, up_col]
                    errors_down = data[:, down_col]
                    errors = (errors_up + errors_down) / 2  # Average for symmetric
                else:
                    errors = np.sqrt(np.abs(observed))  # Fallback
                    errors_up = None
                    errors_down = None

                # Look for syst errors
                if 'sigma_syst_pb' in header:
                    syst_col = header.index('sigma_syst_pb')
                    syst = data[:, syst_col]
                else:
                    syst = None
            else:
                observed = data[:, 1]
                errors = data[:, 2] if data.shape[1] > 2 else np.sqrt(np.abs(observed))
                errors_up = None
                errors_down = None
                syst = None
        else:
            # Generic format
            observable = data[:, 0]
            observed = data[:, 1]
            errors = data[:, 2] if data.shape[1] > 2 else np.sqrt(np.abs(observed))
            errors_up = None
            errors_down = None
            syst = None
    else:
        raise ValueError(f"CSV must have at least 3 columns, got {len(header)}")

    # Filter out NaN/inf values
    valid = np.isfinite(observed) & np.isfinite(errors) & (errors > 0)

    return ChannelData(
        name=name,
        observable=observable[valid],
        observed=observed[valid],
        errors=errors[valid],
        errors_up=errors_up[valid] if errors_up is not None else None,
        errors_down=errors_down[valid] if errors_down is not None else None,
        systematic_errors=syst[valid] if syst is not None else None
    )


def _load_parquet_channel(name: str, path: Path, spec: Dict) -> ChannelData:
    """Load channel data from Parquet file."""
    try:
        import pandas as pd
        df = pd.read_parquet(path)

        # Extract columns based on spec or defaults
        obs_col = spec.get("observable_column", df.columns[0])
        data_col = spec.get("data_column", df.columns[1])
        err_col = spec.get("error_column", df.columns[2] if len(df.columns) > 2 else None)

        observable = df[obs_col].values
        observed = df[data_col].values
        errors = df[err_col].values if err_col else np.sqrt(np.abs(observed))

        return ChannelData(
            name=name,
            observable=observable,
            observed=observed,
            errors=errors
        )
    except ImportError:
        raise ImportError("pandas and pyarrow required for parquet support")


def validate_pack(pack: AnalysisPack) -> Tuple[bool, List[str]]:
    """
    Validate an analysis pack for completeness and correctness.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # 1. Validate metadata against schema
    valid, schema_errors = validate_against_schema(pack.metadata, METADATA_SCHEMA)
    if not valid:
        errors.extend([f"metadata.yaml: {e}" for e in schema_errors])

    # 2. Validate likelihood spec against schema
    valid, schema_errors = validate_against_schema(pack.likelihood_spec, LIKELIHOOD_SCHEMA)
    if not valid:
        errors.extend([f"likelihood.json: {e}" for e in schema_errors])

    # 3. Check that all referenced data files exist
    for ch_spec in pack.likelihood_spec.get("channels", []):
        data_file = ch_spec.get("data", {}).get("file")
        if data_file:
            data_path = pack.path / "data" / data_file
            if not data_path.exists():
                errors.append(f"Missing data file: data/{data_file}")

    # 4. Check that channels have data
    for ch_name, ch_data in pack.channels.items():
        if ch_data.nbins == 0:
            errors.append(f"Channel {ch_name} has no valid data bins")

    # 5. Check parameter consistency
    poi_names = set(p.name for p in pack.pois)
    nuisance_names = set(p.name for p in pack.nuisances)

    if poi_names & nuisance_names:
        overlap = poi_names & nuisance_names
        errors.append(f"Parameter names appear in both POIs and nuisances: {overlap}")

    # 6. Check for LICENSE file
    if not (pack.path / "LICENSE").exists():
        errors.append("Missing LICENSE file (required)")

    # 7. Validate constraints if present
    if pack.constraints:
        valid, schema_errors = validate_against_schema(pack.constraints, CONSTRAINTS_SCHEMA)
        if not valid:
            errors.extend([f"constraints.json: {e}" for e in schema_errors])

    return len(errors) == 0, errors


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file for provenance tracking."""
    h = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_pack_integrity(pack: AnalysisPack) -> Tuple[bool, List[str]]:
    """
    Verify integrity of pack files against recorded hashes.

    Returns:
        (is_valid, list_of_mismatches)
    """
    errors = []

    provenance = pack.metadata.get("provenance", {})
    sources = provenance.get("sources", [])

    for source in sources:
        if "hash" in source and "file" in source:
            file_path = pack.path / source["file"]
            if file_path.exists():
                computed = compute_file_hash(file_path)
                if computed != source["hash"]:
                    errors.append(f"Hash mismatch for {source['file']}: expected {source['hash']}, got {computed}")
            else:
                errors.append(f"File not found for hash verification: {source['file']}")

    return len(errors) == 0, errors
