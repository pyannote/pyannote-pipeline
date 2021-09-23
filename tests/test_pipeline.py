from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform, Integer
from tests.utils import FakeTrial


def test_pipeline_params_simple():
    class SimplePipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.param_a = Uniform(0, 1)
            self.param_b = Integer(3, 10)

    pl = SimplePipeline()
    assert pl.parameters(FakeTrial()) == {"param_a": 0, "param_b": 3}
