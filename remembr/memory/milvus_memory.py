from dataclasses import dataclass, asdict

import datetime, re, time
from time import strftime, localtime
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
import numpy as np

from remembr.memory.memory import Memory, MemoryItem
from remembr.utils import trace

from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


FIXED_SUBTRACT=1721761000 # this is just a large value that brings us close to 1970

TEXT_SEARCH_K = 5

# What similarity_search_with_score_by_vector defaults to, named so position and time
# search can over-fetch and still come back with as many memories as they used to.
POSITION_TIME_K = 4

INDEX_UNITS = ('caption', 'sentence')

# Shortest run of words worth its own vector. Below this a fragment ("Ok.", "Yes.")
# embeds to noise while still competing for one of the k slots.
MIN_SENTENCE_WORDS = 3

# Sentence rows repeat their parent caption, so a search must over-fetch to come back
# with k distinct captions. Captions run about 7 sentences, so 8x covers the case where
# one caption owns most of the top hits.
SENTENCE_OVERFETCH = 8


def sentences_of(caption: str) -> list[str]:
    """The units that get their own embedding.

    A caption is a whole scene -- sidewalk, wall, bench, utility box, building, person --
    averaged into one vector, so a question about the bench matches one seventh of what it
    is compared against. Splitting undoes that dilution.
    """
    caption = caption.strip()
    if not caption:
        return []
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+', caption)]
    kept = [s for s in sentences if len(s.split()) >= MIN_SENTENCE_WORDS]
    # A caption of nothing but fragments still has to be findable, so index it whole.
    return kept or [caption]


def first_per_memory(docs, limit: int):
    """Keep each memory once, at its best-matching sentence.

    Sentence rows repeat their parent's caption, position and time, so without this one
    memory takes several of the slots the model chooses between. Two memories can
    legitimately share a caption -- a stationary robot repeats itself -- so the timestamp
    is part of what tells them apart.
    """
    kept, seen = [], set()
    for doc in docs:
        key = (doc.page_content, str(doc.metadata.get('time')))
        if key in seen:
            continue
        seen.add(key)
        kept.append(doc)
        if len(kept) == limit:
            break
    return kept


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


    def __init__(self, db_collection_name: str, db_ip='127.0.0.1', db_port=19530, time_offset=FIXED_SUBTRACT,
                 index_unit='caption'):
        """`index_unit` picks what a stored vector describes.

        'caption' is the original: one vector per caption, reusing the embeddings the
        caption files ship, which are unnormalised.

        'sentence' embeds each sentence separately and normalises, since it computes both
        sides itself and mxbai is a cosine model. Measured offline over the 12 position
        questions, the closest memory among the retrieved five went from 7.1 m to 1.6 m,
        and the caption naming the asked-about object rose from rank 33 to 5 on the
        question where retrieval had been missing it outright.
        """
        if index_unit not in INDEX_UNITS:
            raise ValueError(f"index_unit must be one of {INDEX_UNITS}, got {index_unit!r}")

        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port
        self.time_offset = time_offset
        self.index_unit = index_unit

        self.embedder = HuggingFaceEmbeddings(
            model_name='mixedbread-ai/mxbai-embed-large-v1',
            encode_kwargs={'normalize_embeddings': index_unit == 'sentence'})

        self.working_memory = []

        self.reset(drop_collection=False)


    def fetch_count(self, limit: int) -> int:
        """Rows to ask Milvus for so that `limit` distinct memories come back."""
        return limit * SENTENCE_OVERFETCH if self.index_unit == 'sentence' else limit

    def distinct_memories(self, docs, limit: int):
        """Collapse the sentence rows of one memory. A no-op on caption rows, which are
        already one per memory and may share a caption with each other."""
        return first_per_memory(docs, limit) if self.index_unit == 'sentence' else docs

    def rows_for(self, memory_dict: dict, text_embedding=None) -> list[dict]:
        """The Milvus rows one memory becomes, one per indexed unit.

        Every row carries the whole caption whatever the unit, so a sentence decides
        which memory is retrieved while the model still reads it in context.
        """
        if self.index_unit == 'caption':
            if text_embedding is None:
                text_embedding = self.embedder.embed_query(memory_dict['caption'])
            return [dict(memory_dict, id=str(time.time()), text_embedding=text_embedding)]

        # A precomputed embedding describes the whole caption, which is the averaging
        # this mode exists to avoid, so it is not reused here. A memory with no caption
        # still gets its row, since position and time search go looking for it.
        sentences = sentences_of(memory_dict['caption']) or [memory_dict['caption']]
        embeddings = self.embedder.embed_documents(sentences)
        stamp = time.time()
        return [dict(memory_dict, id=f"{stamp}-{n}", text_embedding=embedding)
                for n, embedding in enumerate(embeddings)]

    def insert(self, item: MemoryItem, text_embedding=None):

        memory_dict = asdict(item)
        memory_dict['time'] =  [(memory_dict['time'] - self.time_offset), 0.0]

        self.milv_wrapper.insert(self.rows_for(memory_dict, text_embedding))
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
            search_kwargs={"k": self.fetch_count(TEXT_SEARCH_K)})


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

    def search_by_position(self, query: tuple) -> str:
        self._flush_if_needed()
        # docs = pos_db.similarity_search_by_vector(np.array(query))
        docs = similarity_search_with_score_by_vector(
            self.position_vector_db, np.array(query).astype(float),
            k=self.fetch_count(POSITION_TIME_K))
        docs = self.distinct_memories(docs, POSITION_TIME_K)
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
        docs = similarity_search_with_score_by_vector(
            self.time_vector_db, np.array([query, 0]), k=self.fetch_count(POSITION_TIME_K))
        docs = self.distinct_memories(docs, POSITION_TIME_K)
        self._trace_hits("time", hms_time, docs)

        self.working_memory += docs

        docs = self.memory_to_string(docs)
        # np.unique([doc.metadata['time'][0] for doc in docs])
        """Look up things online."""
        return docs


    def search_by_text(self, query: str) -> str:

        self._flush_if_needed()
        docs = self.distinct_memories(self.text_retriever.invoke(query), TEXT_SEARCH_K)
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


