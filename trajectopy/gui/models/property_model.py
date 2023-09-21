"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2023
mail@gtombrink.de
"""
import numpy as np
import pandas as pd
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from trajectopy_core.util.entries import PropertyEntry

from trajectopy.gui.managers.requests import (
    PropertyModelRequest,
    PropertyModelRequestType,
    UIRequest,
    generic_request_handler,
)
from trajectopy.gui.models.table_model import BaseTableModel


class PropertyTableModel(BaseTableModel):
    """
    Class for the property table model.
    """

    ui_request = pyqtSignal(UIRequest)
    operation_finished = pyqtSignal()

    def __init__(self, num_cols: int = 2):
        self.REQUEST_MAPPING = {
            PropertyModelRequestType.EXPORT: self.export,
        }

        header_list = ["Name"]
        header_list.extend(["Value"] * (num_cols - 1))
        super().__init__(headers=header_list)
        self.items: list[PropertyEntry] = []

    @pyqtSlot(PropertyModelRequest)
    def handle_request(self, request: PropertyModelRequest) -> None:
        generic_request_handler(self, request, passthrough_request=True)

    def export(self, request: PropertyModelRequest) -> None:
        columns = [item.name for item in self.items]
        data = np.array([list(item.values) for item in self.items]).T
        dataframe = pd.DataFrame(data=data, columns=columns)
        dataframe.to_csv(request.file_path, index=False, sep=",", float_format="%.6f")
