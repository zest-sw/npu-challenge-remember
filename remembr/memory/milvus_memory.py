from dataclasses import dataclass, asdict

import datetime, time
import warnings
from time import strftime, localtime
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
import numpy as np

from remembr.memory.memory import Memory, MemoryItem
from remembr.utils import trace

from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

# Both LangChain and pymilvus install their own warning filters, so `-W ignore` and
# PYTHONWARNINGS do not silence their deprecation notices. They fire on every
# MilvusMemory.reset(), which happens once per eval question, and each notice prints two
# lines (the message and the offending source line) -- they buried the pipeline trace
# they sit next to.
#
# Filtered by category with no module restriction on purpose: the warnings are raised
# inside pymilvus and langchain internals, not here, so a module pattern misses them.
# Migrating off the deprecated `Milvus` class is the real fix but pulls a new dependency
# into a pinned stack (AGENTS.md, venv dependency trap) -- tracked separately.
for _warning_class in ("langchain_core._api.deprecation:LangChainDeprecationWarning",
                       "pymilvus:PyMilvusDeprecationWarning"):
    try:
        _module, _name = _warning_class.split(":")
        warnings.filterwarnings(
            "ignore", category=getattr(__import__(_module, fromlist=[_name]), _name))
    except Exception:  # noqa: BLE001  -- never let a log tweak break imports
        pass
    pass

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


FIXED_SUBTRACT=1721761000 # this is just a large value that brings us close to 1970




def _caption_gist(caption: str, width: int = 64) -> str:
    """A readable slice of a caption, for the trace only.

    Captions from this VLM open with a fixed assistant preamble ending in a colon
    ("Certainly! Let's break down the image description step by step:", see #50), so
    every line would look identical without skipping it. Display heuristic, not a fix
    -- the underlying captions still carry the preamble.
    """
    flat = " ".join(str(caption).split())
    head, _, rest = flat[:120].partition(":")
    if rest and len(head) < 100:
        flat = (rest + flat[120:]).strip()
    return flat[:width]


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
                 time_offset=FIXED_SUBTRACT, now_clock=None, now_epoch=None):

        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port
        self.time_offset = time_offset
        # An anchor pairing a clock string with the epoch it names, used to convert any
        # clock time the caller asks about into this recording's axis.
        #
        # Needed because the clock times in a question are written in the dataset's
        # timezone, not this machine's. `search_by_time` used to build the date from
        # `localtime(time_offset)` and take the time-of-day from its argument; on this
        # dataset those frames differ by 17 hours, so the composed instant landed 7.0
        # hours outside a 450-second recording and Milvus returned the window boundary
        # for every query (#54). With an anchor from the question itself, both the date
        # and the time-of-day come from the same frame and the offset cancels.
        #
        # A live robot leaves these None: there the machine clock and the clock the robot
        # talks about are the same, so no conversion is needed.
        self.now_clock = now_clock
        self.now_epoch = now_epoch

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
            if not trace.ENABLED:
                print("Resetting memory. We are dropping the current collection")

        self.milv_wrapper = MilvusWrapper(self.db_collection_name, self.db_ip, self.db_port, drop_collection=drop_collection)

        text_vector_db = Milvus(
            self.embedder,
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='text_embedding',
            text_field='caption',
        )
        self.text_retriever = text_vector_db.as_retriever(search_kwargs={"k": 5})


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



    def _trace_hits(self, kind, query, docs):
        """Show what came back, not just how many.

        Which memories were retrieved is the difference between "search missed it" and
        "search found it and the model picked another one" -- the open question in #43.
        Position is on every line because that is what a position answer must match.
        """
        if not trace.ENABLED:
            return
        rows = []
        for number, doc in enumerate(docs, 1):
            stamp = doc.metadata['time']
            stamp = stamp[0] if hasattr(stamp, '__len__') and len(stamp) == 2 else stamp
            if self.time_offset:
                stamp += self.time_offset
            shown = strftime('%H:%M:%S', localtime(stamp))
            where = np.array(doc.metadata['position']).round(1).tolist()
            rows.append(f"{number}  {shown}  {str(where):28s} {_caption_gist(doc.page_content)}")
        # One call, because ToolNode runs a turn's tool calls concurrently and separate
        # prints interleave mid-line.
        trace.step("검색", f"{kind}({query!r})  {len(docs)}개", arrow="←", extra=rows)

    def search_by_position(self, query: tuple) -> str:
        # docs = pos_db.similarity_search_by_vector(np.array(query))
        docs = similarity_search_with_score_by_vector(self.position_vector_db, np.array(query).astype(float))
        self._trace_hits("position", query, docs)

        self.working_memory += docs 

        docs = self.memory_to_string(docs)

        """Look up things online."""
        return docs

    CLOCK_TEMPLATE = "%m/%d/%Y %H:%M:%S"

    def _timezone_shift(self) -> float:
        """Seconds this machine's clock reading differs from the question's clock.

        Zero when there is no anchor, which is the live-robot case.
        """
        if not (self.now_clock and self.now_epoch):
            return 0.0
        stated = datetime.datetime.strptime(self.now_clock, "%Y-%m-%d %H:%M:%S")
        return time.mktime(stated.timetuple()) - float(self.now_epoch)

    def _clock_to_relative_seconds(self, hms_time: str) -> float:
        """A clock time from the question, as seconds since this recording started."""
        hms_time = str(hms_time).strip()
        # Take the date from the anchor, not from localtime -- that mismatch was the bug.
        if self.now_clock:
            date = self.now_clock.split()[0].split('-')
            mdy = f"{date[1]}/{date[2]}/{date[0]}"
        else:
            mdy = strftime('%m/%d/%Y', localtime(self.time_offset))

        try:
            # The model may already have sent a full date, or just HH:MM:SS.
            asked = datetime.datetime.strptime(hms_time, self.CLOCK_TEMPLATE)
        except ValueError:
            asked = datetime.datetime.strptime(f"{mdy} {hms_time}", self.CLOCK_TEMPLATE)

        machine_epoch = time.mktime(asked.timetuple())
        return machine_epoch - self._timezone_shift() - self.time_offset

    def search_by_time(self, hms_time: str) -> str:
        """Memories from around the clock time `hms_time`, as the question states times.

        A clock time on purpose: the questions state times that way, so the model copies
        one rather than computing anything. When this took minutes-ago instead, the model
        had to do the arithmetic and got it wrong -- it asked for 27, 48 and 40 minutes
        back on a 7.5-minute recording, every one of them outside the window.
        """
        try:
            target = self._clock_to_relative_seconds(hms_time)
        except (TypeError, ValueError):
            trace.warn(f"search_by_time: {hms_time!r}를 시각으로 읽을 수 없다 — 지금으로 간주")
            target = self._clock_to_relative_seconds(
                self.now_clock.split()[1]) if self.now_clock else 0.0

        docs = similarity_search_with_score_by_vector(self.time_vector_db, np.array([target, 0]))
        self._trace_hits("time", f"{hms_time} = t{target:+.0f}s", docs)

        self.working_memory += docs

        docs = self.memory_to_string(docs)
        """Look up things online."""
        return docs



    def search_by_text(self, query: str) -> str:

        docs = self.text_retriever.invoke(query)
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


