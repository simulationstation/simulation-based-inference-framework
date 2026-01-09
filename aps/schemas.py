"""
JSON Schemas for Analysis Pack Standard validation.

These schemas define the required structure for metadata, likelihood specifications,
and constraint models.
"""

import json
from typing import Dict, Any

# Metadata schema for metadata.yaml
METADATA_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["name", "version", "description", "provenance", "parameters"],
    "properties": {
        "name": {
            "type": "string",
            "description": "Unique identifier for the analysis pack"
        },
        "version": {
            "type": "string",
            "pattern": r"^\d+\.\d+\.\d+$",
            "description": "Semantic version (e.g., 1.0.0)"
        },
        "description": {
            "type": "string",
            "description": "Short description of the analysis"
        },
        "collaboration": {
            "type": "string",
            "description": "Source experiment/collaboration (e.g., CMS, ATLAS, BESIII)"
        },
        "provenance": {
            "type": "object",
            "required": ["sources"],
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["hepdata", "arxiv", "doi", "url"]},
                            "id": {"type": "string"},
                            "hash": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["type", "id"]
                    }
                },
                "interpretation_limits": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What was approximated or not included"
                }
            }
        },
        "parameters": {
            "type": "object",
            "required": ["pois", "nuisances"],
            "properties": {
                "pois": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "range"],
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "range": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "unit": {"type": "string"},
                            "nominal": {"type": "number"}
                        }
                    }
                },
                "nuisances": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "constraint"],
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "constraint": {
                                "type": "string",
                                "enum": ["normal", "lognormal", "poisson", "uniform", "fixed"]
                            },
                            "nominal": {"type": "number"},
                            "sigma": {"type": "number"},
                            "range": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2
                            }
                        }
                    }
                }
            }
        },
        "statistical_contract": {
            "type": "object",
            "properties": {
                "likelihood_type": {
                    "type": "string",
                    "enum": ["poisson", "gaussian", "hybrid", "surrogate", "sbi"]
                },
                "systematics_treatment": {"type": "string"},
                "asymptotic_valid": {"type": "boolean"},
                "toy_mc_recommended": {"type": "boolean"}
            }
        },
        "reproduction_targets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "source": {"type": "string"},
                    "value": {"type": "number"},
                    "tolerance_rel": {"type": "number"},
                    "tolerance_abs": {"type": "number"}
                }
            }
        },
        "license": {"type": "string"},
        "contact": {"type": "string"},
        "doi": {"type": "string"},
        "references": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

# Likelihood specification schema
LIKELIHOOD_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["type", "channels"],
    "properties": {
        "type": {
            "type": "string",
            "enum": ["poisson", "gaussian", "hybrid", "surrogate"],
            "description": "Likelihood type"
        },
        "channels": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "data", "model"],
                "properties": {
                    "name": {"type": "string"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "format": {"type": "string"},
                            "bins": {"type": "integer"},
                            "observable": {"type": "string"}
                        }
                    },
                    "model": {
                        "type": "object",
                        "properties": {
                            "signal": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string"},
                                    "parametric": {"type": "boolean"},
                                    "formula": {"type": "string"}
                                }
                            },
                            "background": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string"},
                                    "parametric": {"type": "boolean"},
                                    "formula": {"type": "string"}
                                }
                            },
                            "efficiency": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "number"},
                                    "uncertainty": {"type": "number"},
                                    "nuisance": {"type": "string"}
                                }
                            }
                        }
                    },
                    "systematics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string", "enum": ["shape", "norm", "both"]},
                                "up": {"type": "string"},
                                "down": {"type": "string"},
                                "constraint": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },
        "correlations": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["independent", "full", "block_diagonal", "unknown"]},
                "matrix_file": {"type": "string"},
                "groups": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
            }
        }
    }
}

# Constraints schema for nuisance parameter correlations
CONSTRAINTS_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "nuisance_correlations": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["independent", "correlated", "unknown"]},
                "matrix": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                },
                "groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "members": {"type": "array", "items": {"type": "string"}},
                            "correlation": {"type": "number"}
                        }
                    }
                }
            }
        },
        "external_constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "parameters": {"type": "object"}
                }
            }
        }
    }
}

# Parameter schema for model/schema.json
PARAMETER_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["pois", "nuisances"],
    "properties": {
        "pois": {
            "type": "array",
            "description": "Ordered parameters of interest",
            "items": {
                "type": "object",
                "required": ["name", "index"],
                "properties": {
                    "name": {"type": "string"},
                    "index": {"type": "integer"},
                    "transform": {"type": "string", "enum": ["none", "log", "logit"]}
                }
            }
        },
        "nuisances": {
            "type": "array",
            "description": "Ordered nuisance parameters",
            "items": {
                "type": "object",
                "required": ["name", "index"],
                "properties": {
                    "name": {"type": "string"},
                    "index": {"type": "integer"},
                    "transform": {"type": "string", "enum": ["none", "log", "logit"]}
                }
            }
        },
        "derived": {
            "type": "array",
            "description": "Derived parameters",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "formula": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    }
}


def validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate data against a JSON schema.

    Returns:
        (is_valid, list_of_errors)
    """
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(schema)
        errors = list(validator.iter_errors(data))
        if errors:
            return False, [f"{e.json_path}: {e.message}" for e in errors]
        return True, []
    except ImportError:
        # Fallback to basic validation without jsonschema
        errors = []

        # Check required fields
        if "required" in schema:
            for field in schema["required"]:
                if field not in data:
                    errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors
        return True, []
