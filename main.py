import kfp
import kfp.components as comp
from kfp import dsl
import kfp.compiler as compiler

@dsl.pipeline(
    name='plant-00040002',
    description='PV ADS AI model train'
)

def plant_pipeline():
    print("test")
    ml = dsl.ContainerOp(
        name="plant 00040002 pipeline",
        image="bellk/p00040002:1.1"
    )

compiler.Compiler().compile(plant_pipeline, "p00040002_pipeline.tar.gz")
