import kfp
import kfp.dsl as dsl
from kfp import components
import kfp.compiler as compiler


@dsl.pipeline(
    name="data load pipeline",
    description="apply volume"
)

def pipeline():
    data_vop=dsl.VolumeOp(
        name="data-volume",
        resource_name="data-pvc",
        modes=dsl.VOLUME_MODE_RWO,
        size="1Gi"
    )

    data_op=dsl.ContainerOp(
        name="data load container",
        image="bellk/load_data:1.2",
        # command=["sh","-c"],
        # arguments=["/data/0004_0002_I_0001_plant.csv"],
        file_outputs={"load_data":"/0004_0002_I_0001_plant.csv"},
        pvolumes={"/data": data_vop.volume}
    )

    # read_op=dsl.ContainerOp(
    #     name="read data container",
    #     image="bellk/read_data:1.2",
    #     # command=["cat"],
    #     arguments=["/data/0004_0002_I_0001_plant.csv"],
    #     pvolumes={"/data": data_op.pvolumes}
    # )

    data_op.after(data_vop)
    # read_op.after(data_op)

if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, __file__ + ".tar.gz")
