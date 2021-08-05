import kfp
import kfp.components as comp
from kfp import dsl
import kfp.compiler as compiler

@dsl.pipeline(
    name='plant-00040002',
    description='PV ADS AI model pipeline'
)

def plant_pipeline():
    add_data = dsl.ContainerOp(
        name="plant sid:00040002",
        image="bellk/data_load:1.0",
        arguments=["sid:00040002 data load"]
    )

    add_preprocessing = dsl.ContainerOp(
        name="load data preprocessing(missing data & feature engineering",
        image="bellk/preprocessing:1.0",
        arguments=["feature engineering"]

    )

    add_preprocessing.after(add_data)

    ml = dsl.ContainerOp(
        name="model train applied h2o library",
        image="bellk/model_train:1.0",
        arguments=["model train using h2o library"]
    )

    ml.after(add_preprocessing)

compiler.Compiler().compile(plant_pipeline, "p00040002_pipeline.tar.gz")
