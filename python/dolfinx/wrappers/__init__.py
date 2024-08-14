# Copyright (C) 2020 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path


def get_include_path() -> Path:
    """Return path to nanobind wrapper header files"""
    return Path(__file__).parent
