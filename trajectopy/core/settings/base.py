"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import json
from abc import ABC
from dataclasses import dataclass
from typing import Any


@dataclass
class Settings(ABC):
    """Base Class for Settings"""

    def to_dict(self) -> dict:
        output = {}
        for key, value in self.__dict__.items():
            if issubclass(type(value), Settings):
                output[key] = value.to_dict()
            else:
                output[key] = self.encoder(key, value)

        return output

    @staticmethod
    def encoder(name: str, value: Any) -> Any:
        """Encoder for json serialization of dataclasses"""
        return value

    @staticmethod
    def decoder(name: str, value: Any) -> Any:
        """Decoder for json deserialization of dataclasses"""
        return value

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, dct: dict) -> "Settings":
        settings = cls()
        for attribute_name, attribute_type in cls.__annotations__.items():
            if attribute_name not in dct:
                raise ValueError(f"Attribute {attribute_name} not found in input data")

            attribute_data = dct[attribute_name]
            if isinstance(attribute_data, dict) and issubclass(attribute_type, Settings):
                setattr(settings, attribute_name, attribute_type.from_dict(attribute_data))
            else:
                setattr(settings, attribute_name, settings.decoder(attribute_name, attribute_data))

        return settings

    @classmethod
    def from_file(cls, path: str) -> "Settings":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def update_from_dict(self, dct: dict):
        for attribute_name, attribute_type in self.__annotations__.items():
            if attribute_name not in dct:
                raise ValueError(f"Attribute {attribute_name} not found in input data")

            attribute_data = dct[attribute_name]
            if isinstance(attribute_data, dict) and issubclass(attribute_type, Settings):
                getattr(self, attribute_name).update_from_dict(attribute_data)
            else:
                setattr(self, attribute_name, self.decoder(attribute_name, attribute_data))
