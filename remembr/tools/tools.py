# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
import numpy as np
from time import strftime, localtime
import time, datetime

from typing import Any, Iterable, List, Optional, Tuple, Union
from langchain_core.documents import Document

### Doc formatting for the last LLM
def format_document(docs, ref_time=None):
    out_string = ""
    for doc in docs:
        if len(doc.metadata['time']) == 2:
            t = doc.metadata['time'][0]
        else:
            t = doc.metadata['time']
        
        if ref_time:
            t += ref_time
        t = localtime(t)
        t = strftime('%Y-%m-%d %H:%M:%S', t)

        s = f"At time={t}, the robot was at an average position of {np.array(doc.metadata['position']).round(3).tolist()}. "
        s += f"The robot saw the following: {doc.page_content}\n\n"
        out_string += s
    return out_string

def format_docs(docs):
    out_string = ""
    for doc in docs:
        if len(doc['time']) == 2:
            t = doc['time'][0]
        else:
            t = doc['time']

        t = localtime(t)
        t = strftime('%Y-%m-%d %H:%M:%S', t)

        s = f"At time={t}, the robot was at an average position of {np.array(doc['position']).round(3).tolist()}. "
        s += f"The robot saw the following: {doc['caption']}\n\n"
        out_string += s
    return out_string




def search_by_position(pos_db, ref_time, query: tuple) -> str:
    # docs = pos_db.similarity_search_by_vector(np.array(query))
    docs = similarity_search_with_score_by_vector(pos_db, np.array(query).astype(float))

    docs = format_document(docs, ref_time=ref_time)

    """Look up things online."""
    return docs

# `search_by_time` used to live here, taking an HH:MM:SS clock time. It was dead code
# (only Memory.search_by_time is ever called) and it carried the same timezone bug that
# #54 fixed, so it is gone rather than left as a wrong second copy.


def search_by_text(retriever, ref_time, query: str) -> str:

    docs = retriever.invoke(query)

    docs = format_document(docs, ref_time=ref_time)

    """Look up things online."""
    return docs


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