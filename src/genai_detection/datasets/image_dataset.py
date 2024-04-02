from typing import Any, Dict, Callable, Optional

from torchvision.datasets import ImageFolder

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class ImageFolderDataset(AbstractDataset[ImageFolder, ImageFolder]):
    def __init__(self, filepath: str, transform: Optional[Callable] = None):
        """Creates a new instance of ImageDataset to load / save image data for given filepath.

        Args:
            filepath: The location of the image folder to load.
        """
        self.filepath = filepath
        self.transfortm = transform
        self.dataset = ImageFolder(root=self.filepath)


    def _load(self) -> ImageFolder:
        """Loads data from the image file.

        Returns:
            Data from the image file as a Pytorch ImageFolder dataset
        """
        return self.dataset

    def _save(self, data: ImageFolder) -> None:
            """Saves image data to the specified filepath."""
            raise NotImplementedError('Not implemented!')

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self.filepath, classes=self.dataset.classes)