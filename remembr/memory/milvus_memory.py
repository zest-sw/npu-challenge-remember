from dataclasses import dataclass, asdict

import datetime, time
from time import strftime, localtime
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
import numpy as np

from remembr.memory.memory import Memory, MemoryItem
from remembr.memory.sentence_chunking import dedupe_parent_documents
from remembr.utils import trace

from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


FIXED_SUBTRACT=1721761000 # this is just a large value that brings us close to 1970


class MilvusWrapper:

    def __init__(self, collection_name='test', ip_address='127.0.0.1', port=19530, drop_collection=False):
        self.collection_name = collection_name
        self.collection = self.connect_to_milvus_collection(collection_name, 1024, address=ip_address, port=port, drop_collection=drop_collection)


    def drop_collection(self):
        utility.drop_collection(self.collection_name)

    def connect_to_milvus_collection(self, collection_name, dim, address='127.0.0.1', port=19530, drop_collection=False):
        connections.connect(host=address, port=port)
        
        if drop_collection:
            utility.drop_collection(collection_name)
        
        fields = [
            FieldSchema(name='id', dtype=DataType.VARCHAR, description='ids', is_primary=True, auto_id=False, max_length=1000),
            FieldSchema(name='text_embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
            FieldSchema(name='position', dtype=DataType.FLOAT_VECTOR, description='position of robot', dim=3),
            FieldSchema(name='theta', dtype=DataType.FLOAT, description='rotation of robot', dim=1),
            FieldSchema(name='time', dtype=DataType.FLOAT_VECTOR, description='time', dim=2),
            FieldSchema(name='caption', dtype=DataType.VARCHAR, description='caption string', max_length=3000),

        ]
        schema = CollectionSchema(fields=fields, description='text image search')
        collection = Collection(name=collection_name, schema=schema)

        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":1024}
        }
        collection.create_index(field_name="text_embedding", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="position", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="time", index_params=index_params)

        return collection
    
    def insert(self, data_list):
        res = self.collection.insert(data_list)

    def search(self, data):

        self.collection.load()

        BATCH_SIZE = 2
        LIMIT = 10

        param = {
            "metric_type": "L2",
            "params": {
                "nprobe": 1024,
            }
        }

        res = self.collection.search(
            data=[data],
            anns_field="text_embedding",
            param=param,
            batch_size=BATCH_SIZE,
            limit=LIMIT,
            # expr="id > 3",
            output_fields=["id", "text_embedding"]
        )

        return res


class MilvusMemory(Memory):


    def __init__(self, db_collection_name: str, db_ip='127.0.0.1', db_port=19530,
                 time_offset=FIXED_SUBTRACT, text_search_k=5,
                 dedupe_text_parents=False):

        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port
        self.time_offset = time_offset
        self.text_search_k = text_search_k
        self.dedupe_text_parents = dedupe_text_parents
        # A parent has about seven sentences in the measured caption set. Fetch
        # beyond the final k so duplicate sentences cannot leave fewer than k poses.
        self.text_fetch_k = text_search_k * 10 if dedupe_text_parents else text_search_k

        self.embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

        self.working_memory = []

        self.reset(drop_collection=False)


    def insert(self, item: MemoryItem, text_embedding=None, record_id=None):

        memory_dict = asdict(item)
        memory_dict['id'] = record_id or str(time.time())

        if text_embedding is None:
            text_embedding = self.embedder.embed_query(memory_dict['caption'])

        memory_dict['time'] =  [(memory_dict['time'] - self.time_offset), 0.0]

        memory_dict['text_embedding'] = text_embedding

        self.milv_wrapper.insert([memory_dict])
        self._unflushed_inserts = True

    def _flush_if_needed(self):
        """Make inserts visible to search.

        Milvus buffers inserts; until they are flushed a search sees only part of them
        and `num_entities` reads 0. eval drops the collection, inserts a question's whole
        time window, then searches immediately -- so without this it searched a partially
        visible memory, nondeterministically, degrading answers without ever failing.

        Flushed on search rather than in `insert` so a bulk load costs one flush instead
        of one per row. That cost lands in the first search after a load, which
        `eval.instrument_memory_search` then records as db search latency.
        """
        if not self._unflushed_inserts:
            return
        self.milv_wrapper.collection.flush()
        self._unflushed_inserts = False

    def get_working_memory(self) -> list[MemoryItem]:
        return self.working_memory

    def reset(self, drop_collection=True):

        # New wrapper below, so nothing of ours is pending on the old collection.
        self._unflushed_inserts = False

        if drop_collection:
            trace.plain("Resetting memory. We are dropping the current collection")

        self.milv_wrapper = MilvusWrapper(self.db_collection_name, self.db_ip, self.db_port, drop_collection=drop_collection)

        text_vector_db = Milvus(
            self.embedder,
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='text_embedding',
            text_field='caption',
        )
        self.text_retriever = text_vector_db.as_retriever(
            search_kwargs={"k": self.text_fetch_k}
        )


        self.position_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='position',
            text_field='caption',
        )

        self.time_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='time',
            text_field='caption',
        )


    def _doc_epoch(self, doc, ref_time=None):
        """A document's timestamp as an epoch on the recording's clock.

        Milvus hands `time` back as either a scalar or a [start, end] pair depending on
        which index answered, and it is stored relative to `time_offset`. Both readers of
        that field -- the string the model sees and the trace line a human reads -- need
        the same normalisation, and they each had their own copy of it.
        """
        stamp = doc.metadata['time']
        if hasattr(stamp, '__len__') and len(stamp) == 2:
            stamp = stamp[0]
        ref_time = self.time_offset if ref_time is None else ref_time
        return stamp + ref_time if ref_time else stamp

    def _trace_hits(self, kind, query, docs):
        """Show what came back, not just how many.

        Which memories were retrieved is the difference between "search missed it" and
        "search found it and the model picked another one" -- the open question in #43.
        Position is on every line because that is what a position answer must match.
        """
        if not trace.ENABLED:
            return
        trace.step("검색", f"{kind}({query!r})  {len(docs)}개", arrow="←")
        for number, doc in enumerate(docs, 1):
            shown = strftime('%H:%M:%S', localtime(self._doc_epoch(doc)))
            where = np.array(doc.metadata['position']).round(1).tolist()
            trace.note(f"{number}  {shown}  {str(where):28s} {trace.clip(doc.page_content, 64)}")

    def _select_documents(self, docs, limit):
        if self.dedupe_text_parents:
            return dedupe_parent_documents(docs, limit=limit)
        return docs[:limit]

    def search_by_position(self, query: tuple) -> str:
        self._flush_if_needed()
        # docs = pos_db.similarity_search_by_vector(np.array(query))
        fetch_k = self.text_fetch_k if self.dedupe_text_parents else 4
        docs = similarity_search_with_score_by_vector(
            self.position_vector_db, np.array(query).astype(float), k=fetch_k
        )
        docs = self._select_documents(
            docs, self.text_search_k if self.dedupe_text_parents else 4
        )
        self._trace_hits("position", query, docs)

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


        self._flush_if_needed()
        fetch_k = self.text_fetch_k if self.dedupe_text_parents else 4
        docs = similarity_search_with_score_by_vector(
            self.time_vector_db, np.array([query, 0]), k=fetch_k
        )
        docs = self._select_documents(
            docs, self.text_search_k if self.dedupe_text_parents else 4
        )
        self._trace_hits("time", hms_time, docs)

        self.working_memory += docs

        docs = self.memory_to_string(docs)
        # np.unique([doc.metadata['time'][0] for doc in docs])
        """Look up things online."""
        return docs


    def search_by_text(self, query: str) -> str:

        self._flush_if_needed()
        docs = self.text_retriever.invoke(query)
        docs = self._select_documents(docs, self.text_search_k)
        self._trace_hits("text", query, docs)

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
            t = strftime('%Y-%m-%d %H:%M:%S', localtime(self._doc_epoch(doc, ref_time)))

            s = f"At time={t}, the robot was at an average position of {np.array(doc.metadata['position']).round(3).tolist()}."
            s += f"The robot saw the following: {doc.page_content}\n\n"
            out_string += s
        return out_string


# NOTE: This version of the code returns the vector
def similarity_search_with_score_by_vector(
        pos_db,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        if pos_db.col is None:
            print("No existing collection to search.")
            return []

        if param is None:
            param = pos_db.search_params

        # Determine result metadata fields with PK.
        output_fields = pos_db.fields[:]
        # output_fields.remove(pos_db._vector_field) # NOTE: Only thing removed
        timeout = pos_db.timeout or timeout
        # Perform the search.
        res = pos_db.col.search(
            data=[embedding],
            anns_field=pos_db._vector_field,
            param=param,
            limit=k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ret = []
        for result in res[0]:
            data = {x: result.entity.get(x) for x in output_fields}
            doc = pos_db._parse_document(data)
            pair = (doc, result.score)
            ret.append(pair)

        return [doc for doc, _ in ret]

