import kfp.dsl as dsl
import kfp.compiler as compiler
from kfp.components import InputPath, OutputPath
import argparse
import os

parser = argparse.ArgumentParser()
args = parser.parse_args("")

args.sid = '0004_0002'
args.rtu_id_inv = '0004_0002_I_0001'

PVC = "pvmodel-vol-1/"
data_path = PVC + "data/" + args.sid
model_path = PVC + "model/{}/{}".format(args.sid, args.rtu_id_inv)
original_data = "{}_original.csv".format(args.rtu_id_inv)
cleaning_data = "{}_Preprocessing.csv".format(args.rtu_id_inv)

@dsl.pipeline(
    name="data load pipeline",
    description="apply volume"
)

def pipeline():
    data_vop=dsl.VolumeOp(
        name="pipeline-volume",
        resource_name="topinfra-pvc",
        modes=dsl.VOLUME_MODE_RWO,
        size="1Gi"
    )

    data_op=dsl.ContainerOp(
        name="load data from database",
        image="bellk/data_load:0.1",
        # file_outputs={"PVC": data_path, "original data": original_data},
        pvolumes={data_path: data_vop.volume}
    )

    preprocessing_op=dsl.ContainerOp(
        name="loaded data preprocessing",
        image="bellk/data_preprocessing:0.1",
        # arguments=[data_op.output["load_data"]],
        # file_outputs={"PVC": data_path, "preprocessing data": cleaning_data},
        pvolumes={data_path: data_vop.volume}
    )

    model_op=dsl.ContainerOp(
        name="model train based h2o library",
        image="bellk/model_train:0.1",
        # file_outputs={"model": model_path},
        pvolumes={model_path: data_vop.volume}
    )

    data_op.after(data_vop)
    preprocessing_op.after(data_op)
    model_op.after(preprocessing_op)


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, __file__ + ".tar.gz")