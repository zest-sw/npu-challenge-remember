from dataclasses import dataclass, asdict

import datetime, time
from time import strftime, localtime
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
import numpy as np

from remembr.memory.memory import Memory, MemoryItem

from langchain_huggingface import HuggingFaceEmbeddings

from pymilvus import MilvusClient, DataType


FIXED_SUBTRACT=1721761000 # this is just a large value that brings us close to 1970

# Vector fields and their index settings. `position` and `time` are searched as
# vectors so that "what was near (x,y,z)" and "what happened around t" are ordinary
# nearest-neighbour lookups; nlist is small because both are low-dimensional.
_INDEX_SPECS = (
    ("text_embedding", 1024),
    ("position", 2),
    ("time", 2),
)

# Fields returned by a search; everything except the embedding, which is large and
# unused downstream. memory_to_string() reads `position` and `time`, and the caption
# becomes the Document's page_content.
_OUTPUT_FIELDS = ["id", "position", "theta", "time", "caption"]


def _to_documents(results):
    """Convert a MilvusClient search result into the langchain Documents the rest of
    ReMEmbR expects (page_content = caption, metadata = position/time/theta/id)."""
    docs = []
    for hit in (results[0] if results else []):
        entity = hit.get("entity", hit)
        docs.append(
            Document(
                page_content=entity.get("caption", ""),
                metadata={k: entity[k] for k in _OUTPUT_FIELDS if k in entity},
            )
        )
    return docs


class MilvusWrapper:
    """Thin MilvusClient wrapper.

    Uses the MilvusClient API rather than the ORM-style Collection/connections/utility
    API, which PyMilvus deprecated and removes in 3.1.
    """

    def __init__(self, collection_name='test', ip_address='127.0.0.1', port=19530, drop_collection=False):
        self.collection_name = collection_name
        self.client = MilvusClient(uri=f"http://{ip_address}:{port}")
        self.connect_to_milvus_collection(collection_name, 1024, drop_collection=drop_collection)

    def drop_collection(self):
        self.client.drop_collection(self.collection_name)

    def connect_to_milvus_collection(self, collection_name, dim, drop_collection=False):
        if drop_collection and self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

        # Attach to the existing collection when there is one -- the eval loads a
        # database that preprocessing already populated, so creating unconditionally
        # would either fail or discard those memories.
        if not self.client.has_collection(collection_name):
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field('id', DataType.VARCHAR, is_primary=True, max_length=1000)
            schema.add_field('text_embedding', DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field('position', DataType.FLOAT_VECTOR, dim=3)
            schema.add_field('theta', DataType.FLOAT)
            schema.add_field('time', DataType.FLOAT_VECTOR, dim=2)
            schema.add_field('caption', DataType.VARCHAR, max_length=3000)

            index_params = self.client.prepare_index_params()
            for field_name, nlist in _INDEX_SPECS:
                index_params.add_index(
                    field_name=field_name,
                    index_type='IVF_FLAT',
                    metric_type='L2',
                    params={'nlist': nlist},
                )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

        self.client.load_collection(collection_name)
        return collection_name

    def insert(self, data_list):
        self.client.insert(collection_name=self.collection_name, data=data_list)

    def search(self, data, anns_field='text_embedding', limit=10, nprobe=1024):
        """Nearest-neighbour search over one of the vector fields, as Documents."""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[list(data)],
            anns_field=anns_field,
            search_params={'metric_type': 'L2', 'params': {'nprobe': nprobe}},
            limit=limit,
            output_fields=_OUTPUT_FIELDS,
        )
        return _to_documents(results)




class MilvusMemory(Memory):


    def __init__(self, db_collection_name: str, db_ip='127.0.0.1', db_port=19530, time_offset=FIXED_SUBTRACT):

        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port
        self.time_offset = time_offset

        self.embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

        self.working_memory = []

        self.reset(drop_collection=False)


    def insert(self, item: MemoryItem, text_embedding=None):

        memory_dict = asdict(item)
        memory_dict['id'] = str(time.time())

        if text_embedding is None:
            text_embedding = self.embedder.embed_query(memory_dict['caption'])

        memory_dict['time'] =  [(memory_dict['time'] - self.time_offset), 0.0]

        memory_dict['text_embedding'] = text_embedding

        self.milv_wrapper.insert([memory_dict])

    def get_working_memory(self) -> list[MemoryItem]:
        return self.working_memory

    def reset(self, drop_collection=True):

        if drop_collection:
            print("Resetting memory. We are dropping the current collection")

        self.milv_wrapper = MilvusWrapper(self.db_collection_name, self.db_ip, self.db_port, drop_collection=drop_collection)


    def search_by_position(self, query: tuple) -> str:
        docs = self.milv_wrapper.search(
            np.array(query).astype(float), anns_field='position', limit=4
        )

        self.working_memory += docs

        docs = self.memory_to_string(docs)

        """Look up things online."""
        return docs

    def search_by_time(self, hms_time: str) -> str:

        # Input is time like 08:20:30
        # need to convert to searchable time
        t = localtime(self.time_offset)
        mdy_date = strftime('%m/%d/%Y', t)
        template = "%m/%d/%Y %H:%M:%S"

        # if the hms_time is already in the mdy hms format without me doing anything, let's just use that.
        # bad llms don't listen :(
        try:
            res = bool(datetime.datetime.strptime(hms_time, template))
        except ValueError:
            res = False

        hms_time = hms_time.strip()
        if not res: # convert to the right format then
            hms_time = mdy_date + ' ' + hms_time

        query = time.mktime(datetime.datetime.strptime(hms_time,template).timetuple()) - self.time_offset
        # convert from hms_time to something searchable


        docs = self.milv_wrapper.search(
            np.array([query, 0]), anns_field='time', limit=4
        )

        self.working_memory += docs

        docs = self.memory_to_string(docs)
        # np.unique([doc.metadata['time'][0] for doc in docs])
        """Look up things online."""
        return docs



    def search_by_text(self, query: str) -> str:

        docs = self.milv_wrapper.search(
            self.embedder.embed_query(query), anns_field='text_embedding', limit=5
        )

        self.working_memory += docs

        docs = self.memory_to_string(docs)

        """Look up things online."""
        return docs


    ### Doc formatting for the last LLM
    def memory_to_string(self, memory_list: list[MemoryItem], ref_time: float=None):
        if ref_time == None:
            ref_time = self.time_offset

        out_string = ""
        for doc in memory_list:
            if len(doc.metadata['time']) == 2:
                t = doc.metadata['time'][0]
            else:
                t = doc.metadata['time']

            if ref_time:
                t += ref_time
            t = localtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            s = f"At time={t}, the robot was at an average position of {np.array(doc.metadata['position']).round(3).tolist()}."
            s += f"The robot saw the following: {doc.page_content}\n\n"
            out_string += s
        return out_string
