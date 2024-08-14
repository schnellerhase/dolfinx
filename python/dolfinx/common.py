# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""General tools for timing and configuration."""

import functools
import typing
from types import TracebackType

from mpi4py import MPI as _MPI

from dolfinx.cpp import Reduction as _Reduction
from dolfinx.cpp import Timer as _Timer
from dolfinx.cpp import TimingTyp as _TimingType
from dolfinx.cpp.common import (
    IndexMap,
    git_commit_hash,
    has_adios2,
    has_debug,
    has_kahip,
    has_parmetis,
    has_petsc,
)
from dolfinx.cpp.common import list_timings as _list_timings
from dolfinx.cpp.common import timing as _timing

__all__ = [
    "IndexMap",
    "Timer",
    "timed",
    "git_commit_hash",
    "has_adios2",
    "has_debug",
    "has_kahip",
    "has_petsc",
    "has_parmetis",
]

TimingType = _TimingType
Reduction = _Reduction


def timing(task: str) -> tuple[int, float, float, float]:
    return _timing(task)


def list_timings(
    comm: _MPI.Comm, timing_types: list, reduction: _Reduction = Reduction.max
) -> None:
    """Print out a summary of all Timer measurements, with a choice of
    wall time, system time or user time. When used in parallel, a
    reduction is applied across all processes. By default, the maximum
    time is shown."""
    _list_timings(comm, timing_types, reduction)


class Timer:
    """A timer can be used for timing tasks. The basic usage is::

        with Timer(\"Some costly operation\"):
            costly_call_1()
            costly_call_2()

    or::

        with Timer() as t:
            costly_call_1()
            costly_call_2()
            print(\"Elapsed time so far: %s\" % t.elapsed()[0])

    The timer is started when entering context manager and timing
    ends when exiting it. It is also possible to start and stop a
    timer explicitly by::

        t = Timer(\"Some costly operation\")
        t.start()
        costly_call()
        t.stop()

    and retrieve timing data using::

        t.elapsed()

    Timings are stored globally (if task name is given) and
    may be printed using functions ``timing``, ``timings``,
    ``list_timings``, ``dump_timings_to_xml``, e.g.::

        list_timings(comm, [TimingType.wall, TimingType.user])
    """

    _cpp_object: _Timer

    def __init__(self, name: typing.Optional[str] = None) -> None:
        self._cpp_object = _Timer(None if name is None else name)

    def __enter__(self) -> typing.Self:
        self._cpp_object.start()
        return self

    def __exit__(
        self,
        exception_type: typing.Optional[BaseException],
        exception_value: typing.Optional[BaseException],
        traceback: typing.Optional[TracebackType],
    ) -> None:
        self._cpp_object.stop()

    def start(self) -> None:
        self._cpp_object.start()

    def stop(self) -> float:
        return self._cpp_object.stop()

    def resume(self) -> None:
        self._cpp_object.resume()

    def elapsed(self) -> float:
        return self._cpp_object.elapsed()


def timed(task: str) -> typing.Callable[..., typing.Any]:
    """Decorator for timing functions."""

    def decorator(func: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
        @functools.wraps(func)
        def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            with Timer(task):
                return func(*args, **kwargs)

        return wrapper

    return decorator
