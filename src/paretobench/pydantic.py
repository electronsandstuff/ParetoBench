import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from typing import Any


class NumpyArray(np.ndarray):
    """NumPy array subclass with Pydantic support"""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        def validate_array(v: Any) -> np.ndarray:
            """Convert input to numpy array, handling nested lists"""
            if isinstance(v, np.ndarray):
                return v.astype(float)
            return np.asarray(v, dtype=float)

        # Use any_schema to accept any type, then validate it
        return core_schema.no_info_after_validator_function(
            validate_array,
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda v: v.tolist()),
        )
