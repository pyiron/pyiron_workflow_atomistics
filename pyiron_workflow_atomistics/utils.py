import pyiron_workflow as pwf

@pwf.as_function_node
def update_dataclass(dataclass_instance, key, value):
    from dataclasses import replace, asdict

    if key not in asdict(dataclass_instance):
        raise KeyError(f"Field '{key}' not in dataclass {type(dataclass_instance).__name__}")

    updated_dataclass = replace(dataclass_instance, **{key: value})
    return updated_dataclass