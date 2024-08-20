from typing import Optional
from pydantic import BaseModel, validator, model_validator, Field, field_validator
import numpy as np


# TODO: change this so that they are all arrays, but can have second dimension be zero length if nothing is defined. This
# is more consistent with existing library
class Population(BaseModel):
    x: Optional[np.ndarray] = Field(default=None, arbitrary_types_allowed=True)
    f: np.ndarray = Field(..., arbitrary_types_allowed=True)
    g: Optional[np.ndarray] = Field(default=None, arbitrary_types_allowed=True)
    fevals: int

    class Config:
        arbitrary_types_allowed = True
        
    # Validator for numpy arrays to ensure they are 2-dimensional and floating-point
    @field_validator('x', 'f', 'g')
    def check_array(cls, v):
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise TypeError('The input must be a numpy.ndarray')
            if v.ndim != 2:
                raise ValueError('The array must be 2-dimensional')
            if not np.issubdtype(v.dtype, np.floating):
                raise TypeError('The array elements must be of floating-point type')
        return v

    # Model validator to ensure that 'f' is always provided and correct
    @model_validator(mode='before')
    @classmethod
    def check_mandatory_fields(cls, values):
        f = values.get('f')
        if f is None or not isinstance(f, np.ndarray):
            raise ValueError('Objectives array "f" must be provided and must be a numpy.ndarray')
        return values
    

# Example usage
try:
    population = Population(
        x=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        f=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        g=np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64),
        fevals=100
    )
    print("Population model is valid:", population)
except Exception as e:
    print("Validation error:", e)
