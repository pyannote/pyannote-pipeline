from pyannote.pipeline.parameter import ParamDict, Uniform, Integer, ParamList, Categorical
from tests.utils import FakeTrial


def test_params_dict():
    params_dict = ParamDict({
        "param_a": Uniform(0, 1),
        "param_b": Integer(0, 10)
    })

    assert params_dict("params_dict", FakeTrial()) == {'param_a': 0.0,
                                                       'param_b': 0}


def test_params_list():
    params_list = ParamList([
        Integer(i, 10) for i in range(5)
    ])
    assert params_list("params_list", FakeTrial()) == list(range(5))


def test_nested_params():
    nested = ParamList([
        ParamDict({
            "param_a": Uniform(0, 1),
            "param_b": Integer(0, 10)
        }),
        Categorical(["a", "b", "c"]),
        Uniform(0, 100)
    ])
    assert nested("nested_params", FakeTrial()) == [{'param_a': 0.0,
                                                     'param_b': 0},
                                                    'a', 0.0]

    nested = ParamDict({
        "param_list": ParamList([Uniform(i, 10) for i in range(3)]),
        "param_cat": Categorical(["a", "b", "c"])
    })

    assert nested("nested_params", FakeTrial()) == {'param_list': [0.0, 1.0, 2.0],
                                                    'param_cat': 'a'}
