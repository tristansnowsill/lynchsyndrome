import datetime
from os import getenv

from injector import Module

from ..config.metadata import Metadata

class MetadataProvider(Module):
    def configure(self, binder):
        metadata = Metadata(
            NAME=getenv('NAME', 'lynchsyndrome'),
            RUN_DATE=datetime.date.today(),
            RUN_DATETIME=datetime.datetime.now()
        )
        binder.bind(Metadata, metadata)
