from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform, Integer, ParamDict, ParamList
from tests.utils import FakeTrial


def test_pipeline_params_simple():
    class TestPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.param_a = Uniform(0, 1)
            self.param_b = Integer(3, 10)

    pl = TestPipeline()
    assert pl.parameters(FakeTrial()) == {"param_a": 0, "param_b": 3}


def test_pipeline_params_structured():
    class TestPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.params_dict = ParamDict({
                "param_a": Uniform(0, 1),
                "param_b": Integer(5, 10)
            })
            self.params_list = ParamList([
                Integer(i, 10) for i in range(5)
            ])

    pl = TestPipeline()
    fake_params = {'params_dict': {'param_a': 0.0,
                                   'param_b': 5},
                   'params_list': [0, 1, 2, 3, 4]}

    assert pl.parameters(FakeTrial()) == fake_params
    pl.instantiate(pl.parameters(FakeTrial()))
    assert (pl._unflatten(pl._flattened_parameters(instantiated=True))
            ==
            fake_params
            )


def test_pipeline_with_subpipeline():
    class SubPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.param = Uniform(0, 1)

    class TestPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.params_dict = ParamDict({
                "param_a": Uniform(0, 1),
                "param_b": Integer(5, 10)
            })
            self.subpl = SubPipeline()

    pl = TestPipeline()
    fake_params = {'subpl': {'param': 0.0},
                   'params_dict': {'param_a': 0.0,
                                   'param_b': 5}
                   }

    assert pl.parameters(FakeTrial()) == fake_params

    pl.instantiate(pl.parameters(FakeTrial()))
    assert pl._unflatten(pl._flattened_parameters(instantiated=True)) == fake_params
