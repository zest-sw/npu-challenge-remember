# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
import numpy as np
from time import strftime, localtime
import time, datetime
import json, os, re

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

def search_by_time(time_db, ref_time, hms_time: str) -> str:

    # Input is time like 08:20:30
    # need to convert to searchable time
    t = localtime(ref_time)
    mdy_date = strftime('%m/%d/%Y', t)
    template = "%m/%d/%Y %H:%M:%S"

    # if the hms_time is already in the mdy hms format without me doing anything, let's just use that.
    # bad llms don't listen :(
    try:
        res = bool(datetime.strptime(query, template))
    except ValueError:
        res = False

    hms_time = hms_time.strip()
    if not res: # convert to the right format then
        hms_time = mdy_date + ' ' + hms_time

    query = time.mktime(datetime.datetime.strptime(hms_time,template).timetuple()) - ref_time
    # convert from hms_time to something searchable


    docs = similarity_search_with_score_by_vector(time_db, np.array([query, 0]))

    docs = format_document(docs, ref_time=ref_time)
    # np.unique([doc.metadata['time'][0] for doc in docs])
    """Look up things online."""
    return docs



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


# ===========================================================================
# Space/time calculators — the bodies behind ReMEmbRAgent's calculator tools
# (compute_minutes_ago / compute_duration_minutes / find_nearest_position).
#
# Everything the agent reads is ABSOLUTE: the question preamble states "The
# current time is 2023-01-16 08:03:02 and you are located at [12.3, 56.7, 0.1]",
# and memory_to_string() renders each memory as "At time=2023-01-16 07:55:32, the
# robot was at an average position of [...]". But NaVQA wants RELATIVE answers —
# `time` in minutes-ago, `duration` in minutes, "nearest X" as an argmin over
# distances — and eval scores those numbers directly (position = L2 norm,
# time/duration = absolute difference), so a slip in the model's mental arithmetic
# becomes error.
#
# So the agent delegates: it decides a calculation is needed, calls the tool, and
# passes every operand explicitly (including "now" and its own position, both of
# which it reads off the question). These functions are pure — no hidden state, no
# inspecting the question or the answer to work out what to do. The regex inside
# parse_timestamp only turns an argument string into a datetime; it decides
# nothing.
# ===========================================================================

_TS_FULL = "%Y-%m-%d %H:%M:%S"
_TS_HMS = "%H:%M:%S"


def parse_timestamp(value, reference=None):
    """Parse a timestamp the model copied out of memory/preamble text.

    Accepts a full '%Y-%m-%d %H:%M:%S' or a bare '%H:%M:%S' (which borrows the
    date from `reference`, since memory captions sometimes echo only the clock
    time). Also accepts a datetime unchanged. Returns None when there is nothing
    usable, so callers can fall back instead of raising mid-eval.
    """
    if isinstance(value, datetime.datetime):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    match = re.search(r"(\d{4}-\d{2}-\d{2})[ T]+(\d{2}:\d{2}:\d{2})", text)
    if match:
        try:
            return datetime.datetime.strptime(f"{match.group(1)} {match.group(2)}", _TS_FULL)
        except ValueError:
            return None

    match = re.search(r"\b(\d{1,2}:\d{2}:\d{2})\b", text)
    if not match:
        return None
    try:
        clock = datetime.datetime.strptime(match.group(1).zfill(8), _TS_HMS).time()
    except ValueError:
        return None
    ref = reference if isinstance(reference, datetime.datetime) else None
    if ref is None:
        return None
    return datetime.datetime.combine(ref.date(), clock)


def minutes_ago(event, now):
    """Minutes elapsed from `event` until `now` (the `time` answer field).

    Returns None if either side is unparseable, or if `event` is in the future:
    a negative minutes-ago is never a valid answer here, and silently returning
    one would be scored as a large error rather than surfacing as unanswered.
    """
    now_dt = parse_timestamp(now)
    event_dt = parse_timestamp(event, reference=now_dt)
    if now_dt is None or event_dt is None:
        return None
    delta = (now_dt - event_dt).total_seconds() / 60.0
    return None if delta < 0 else float(delta)


def duration_minutes(start, end, reference=None):
    """Minutes between two timestamps (the `duration` answer field).

    Order-independent: the model routinely reports the pair backwards, and a
    duration is a magnitude either way.
    """
    ref = parse_timestamp(reference) if reference is not None else None
    start_dt = parse_timestamp(start, reference=ref)
    end_dt = parse_timestamp(end, reference=ref if ref is not None else start)
    if start_dt is None or end_dt is None:
        return None
    return float(abs((end_dt - start_dt).total_seconds()) / 60.0)


def _coords(value):
    """Coerce '[1, 2, 3]' / (1,2,3) / [1,2,3] to a float list, else None."""
    if isinstance(value, str):
        try:
            value = json.loads(value.replace("(", "[").replace(")", "]"))
        except (ValueError, TypeError):
            return None
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def distance(a, b):
    """Euclidean distance between two coordinates. None if either is unusable.

    Full 3-D on purpose: the benchmark asks about elevators and stairs, so the
    z axis genuinely separates candidates on different floors. Shorter vectors
    are compared on the axes they share.
    """
    pa, pb = _coords(a), _coords(b)
    if pa is None or pb is None:
        return None
    width = min(len(pa), len(pb))
    return float(np.linalg.norm(np.array(pa[:width]) - np.array(pb[:width])))


def _coord_list(value):
    """Normalize to a list of coordinates.

    Tolerates what the model actually emits: a list of coordinates, a single flat
    coordinate, or either of those as a JSON string.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value.replace("(", "[").replace(")", "]"))
        except (ValueError, TypeError):
            return []
    if not isinstance(value, (list, tuple)) or not value:
        return []
    if all(isinstance(v, (int, float)) for v in value):
        single = _coords(value)  # flat [x, y, z], not a list of them
        return [single] if single else []
    out = []
    for item in value:
        coordinate = _coords(item)
        if coordinate:
            out.append(coordinate)
    return out


def nearest(candidates, origin):
    """Pick the candidate closest to `origin` -> (index, coordinate, distance).

    Answers the "where is the closest/nearest X" questions without the model
    having to compare distances itself. Returns (None, None, None) when nothing
    is usable.
    """
    best = (None, None, None)
    for index, candidate in enumerate(_coord_list(candidates)):
        d = distance(candidate, origin)
        if d is None:
            continue
        if best[2] is None or d < best[2]:
            best = (index, candidate, d)
    return best

