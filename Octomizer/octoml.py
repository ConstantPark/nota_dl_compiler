#!/usr/bin/env python3

"""
This is a simple example of use of the Octomizer API. It uploads an ONNX version of
the MNIST model, benchmarks it on ONNX-RT, Octomizes it, and returns both the Octomized
model benchmark results and a Python package for the optimized model.

After installing the Octomizer SDK, run as:
  % export OCTOMIZER_API_TOKEN=<your API token>
  % ./octomizer_example.py

Windows-based users should set their API token as an environment variable rather than use the export command, which is not supported in some versions of Windows.

Typical output will look like this:

Uploading model tests/testdata/mnist.onnx...
Benchmarking using ONNX-RT...
ONNX-RT benchmark metrics:
runtime_ms_mean: 0.1281123161315918
runtime_ms_std: 0.3681384027004242

Waiting for Octomization to complete...
Octomized TVM benchmark metrics:
runtime_ms_mean: 0.07081685215234756
runtime_ms_std: 0.019525768235325813
compile_ms: 1084.7218017578125
full_metrics_dataref_uuid: "7df384b3-1d3d-4ef0-ae25-e57588090876"

Saved packaged model to ./mnist_octomized-0.1.0-py3-none-any.whl
"""

from __future__ import annotations

import octomizer.client
import octomizer.models.onnx_model as onnx_model


def main():
    # Specify the model file and input layer parameters.
    onnx_model_file = "tests/testdata/mnist.onnx"

    # Specify the Python package name for the resulting model.
    model_package_name = "mnist_octomized"

    # The input layer name, type, and shape are model-dependent. These happen
    # to be the correct values for the MNIST model we are testing here.
    input_layer_name = "Input3"
    input_layer_dtype = "float32"
    input_layer_shape = [1, 1, 28, 28]

    # Specify the platform to target.
    platform = "broadwell"

    # Create the Octomizer Client instance.
    client = octomizer.client.OctomizerClient()

    # Upload the ONNX model file.
    print(f"Uploading model {onnx_model_file}...")
    input_layer_shapes = {input_layer_name: input_layer_shape}
    input_layer_dtypes = {input_layer_name: input_layer_dtype}
    model = onnx_model.ONNXModel(
        client,
        name=model_package_name,
        model=onnx_model_file,
        input_shapes=input_layer_shapes,
        input_dtypes=input_layer_dtypes,
        description="Created by octomizer_example.py",
    )
    model_variant = model.get_uploaded_model_variant()

    # Benchmark the model and get results. MNIST is a small and simple model,
    # so this should return quickly.
    benchmark_workflow = model_variant.benchmark(platform)
    print("Benchmarking using ONNX-RT...")
    benchmark_workflow.wait()
    if not benchmark_workflow.completed():
        raise RuntimeError(
            f"Workflow did not complete, status is {benchmark_workflow.status()}"
        )
    metrics = benchmark_workflow.metrics()
    print(f"ONNX-RT benchmark metrics:\n{metrics}")

    # Octomize the model. Since this is a small model and this is only an example,
    # we set the `kernel_trials` and `early_stopping_threshold` ridiculously low,
    # so it does not take too long to complete. Normally these values would be
    # much higher (the defaults are 500 and 100, respectively). See the python API
    # documentation for the full list of tuning options.
    octomize_workflow = model_variant.octomize(
        platform,
        kernel_trials=3,
        early_stopping_threshold=1,
    )
    print("Waiting for Octomization to complete...")
    octomize_workflow.wait()
    if not octomize_workflow.completed():
        raise RuntimeError(
            f"Workflow did not complete, status is {octomize_workflow.status()}"
        )
    metrics = octomize_workflow.metrics()
    # Don't be surprised if the TVM numbers are slower than ONNX-RT -- we didn't
    # run enough rounds of autotuning to ensure great performance!
    print(f"Octomized TVM benchmark metrics:\n{metrics}")

    # Download the packaged Python wheel for the optimized model.
    output_filename = octomize_workflow.save_package(out_dir=".")
    print(f"Saved packaged model to {output_filename}")


if __name__ == "__main__":
    main()
