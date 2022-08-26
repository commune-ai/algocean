
from dataclasses import dataclass,field
from typing import List,Optional
from enforce_typing import enforce_types
from datetime import datetime
import json

@dataclass
class DatasetMetadata():
    title:str
    authors:List
    file_link:str
    sample_link:str
    tags:List[str]
    description:str
    license:str
    links:Optional[str]=None
    otherinfo:Optional[str]=None
    copyrightHolder:Optional[str]=None
    content_langs:Optional[str]=None
    app_algorithm:Optional[str]=None
    dtype:str="Dataset"

    @property
    def nbytes(self):
        return len(self.tobytes())


    def __prettydict__(self):
        return {k: str(v) for k, v in ds_class.__dict__.items() if v is not None and not ""}

    def __templatestr__(self):

        metadata = f'''
        Dtype: {self.dtype}

        Owner/License: {self.title},{self.authors},{self.copyrightHolder}

        Files: {self.file_link},{self.sample_link}

        About Files: {self.tags}

        Description: {self.description}

        License: {self.license}

        Links: {self.links}

        Language: {self.content_langs}

        Other: {self.otherinfo}

        Approved Algorithms: {self.app_algorithm}

            '''
        return metadata
