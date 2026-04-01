"""Output schema infrastructure for workflow implementations.

This module provides the `OutputFile` descriptor and `get_outputs_schema()` function
to declare and document workflow outputs a-priori. Workflows can use an inner `Outputs`
class to specify output files with their types and optional schemas.

Example
-------
>>> from muflow import OutputFile, register_workflow
>>> import pydantic
>>>
>>> class ResultSchema(pydantic.BaseModel):
...     accuracy: float
...     loss: float
>>>
>>> class MyOutputs:
...     files = {
...         "result.json": OutputFile(
...             file_type="json",
...             description="Training results",
...             schema=ResultSchema,
...         ),
...     }
>>>
>>> @register_workflow(name="myapp.my_workflow", outputs=MyOutputs)
... def my_workflow(context):
...     pass
"""

from dataclasses import dataclass
from typing import Literal, Type

import pydantic


@dataclass
class OutputFile:
    """Descriptor for an output file produced by a workflow.

    Attributes
    ----------
    file_type : Literal["json", "netcdf", "text", "binary"]
        The type of the output file.
    description : str
        Human-readable description of the file's contents.
    schema : Type[pydantic.BaseModel] | None
        For JSON files, an optional Pydantic model describing the structure.
    optional : bool
        Whether this file is always produced or only under certain conditions.
    """

    file_type: Literal["json", "netcdf", "text", "binary"]
    description: str = ""
    schema: Type[pydantic.BaseModel] | None = None
    optional: bool = False


def get_outputs_schema(outputs_class) -> list:
    """Generate JSON schema from an Outputs class.

    Parameters
    ----------
    outputs_class : class or None
        The Outputs inner class from a WorkflowImplementation, or None.

    Returns
    -------
    list
        List of file descriptors with their schemas.
    """
    schema = []

    if outputs_class is None:
        return schema

    # File descriptors for output files
    if hasattr(outputs_class, "files") and outputs_class.files:
        for filename, descriptor in outputs_class.files.items():
            file_info = {
                "filename": filename,
                "file_type": descriptor.file_type,
                "description": descriptor.description,
                "optional": descriptor.optional,
                "schema": None,
            }
            if descriptor.schema is not None:
                file_info["schema"] = descriptor.schema.model_json_schema()
            schema.append(file_info)

    return schema
