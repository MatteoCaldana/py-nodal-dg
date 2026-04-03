# -*- coding: utf-8 -*-
import numpy as np
import typing

EPS = np.finfo(float).eps
TOL = 2e-5


def check_type(sself):
    for (name, field_type) in sself.__annotations__.items():
        if typing.get_origin(field_type) is typing.Union:
            any_true = False
            for utype in typing.get_args(field_type):
                if isinstance(sself.__dict__[name], utype):
                    any_true = True
                    break
            if not any_true:
                current_type = type(sself.__dict__[name])
                raise TypeError(
                    f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`"
                )
        else:
            if isinstance(field_type, type):
                if not isinstance(sself.__dict__[name], field_type):
                    current_type = type(sself.__dict__[name])
                    raise TypeError(
                        f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`"
                    )
