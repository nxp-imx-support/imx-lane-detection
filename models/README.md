## ML Models

i.MX Lane detection application example uses two ML models in total to achieve lane lines detection and object detection. More details about each model is listed below.

### Lane Detection

| Information               | Value                                               |
| ------------------------- | --------------------------------------------------- |
| Input shape               | RGB image \[1, 288, 800, 3]                         |
| Input image preprocessing | mul:0.01735207,add:-2.017699                        |
| Output shape              | \[1, 101, 56, 4]                                    |
| Model size (INT8)         | 58.636 M                                            |
| Source framework          | Ultra-Fast-Lane-Detection(TensorFlow Lite)          |
| Origin                    | <https://github.com/cfzd/Ultra-Fast-Lane-Detection> |

### Object detection

| Information               | Value                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Input shape               | RGB image \[1, 300, 300, 3]                                                                                    |
| Input image preprocessing | None (Uint8 Input)                                                                                             |
| Output shape              | TFLite\_Detection\_PostProcess: <br /> mem\_boxes; mem\_detections; mem\_detections; mem\_num                  |
| MACs                      | 5.933 M                                                                                                        |
| Source framework          | mobilenet\_ssd\_v2\_coco\_quant\_postprocess.tflite                                                            |
| Origin                    | <https://github.com/google-coral/edgetpu/blob/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite> |

### Benchmarks

The quantized models have been tested on i.MX 8M Plus and i.MX 93 using `./benchmark_model` tool. For i.MX 93, quantized models need to be compiled by vela tool first before using NPU delegate.
(see [i.MX Machine Learning User's Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf) for more details).

> **NOTE:** Evaluated on BSP LF-6.6.23\_2.0.0.

#### Lane Detection&#x20;

| Platform     | Accelerator     | Avg. Inference Time | Command                                                                                                            |
| ------------ | --------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| i.MX 8M Plus | CPU (1 thread)  | 1815.70 ms          | ./benchmark\_model --graph=lane\_detection.tflite                                                                  |
| i.MX 8M Plus | CPU (4 threads) | 554.93 ms           | ./benchmark\_model --graph=lane\_detection.tflite --num\_threads=4                                                 |
| i.MX 8M Plus | NPU             | 34.91 ms            | ./benchmark\_model --graph=lane\_detection.tflite --external\_delegate\_path=/usr/lib/libvx\_delegate.so           |
| i.MX 93      | CPU (1 thread)  | 687.47 ms           | ./benchmark\_model --graph=lane\_detection.tflite                                                                  |
| i.MX 93      | CPU (2 threads) | 430.59 ms           | ./benchmark\_model --graph=lane\_detection.tflite --num\_threads=2                                                 |
| i.MX 93      | NPU             | 51.49 ms            | ./benchmark\_model --graph=lane\_detection\_vela.tflite --external\_delegate\_path=/usr/lib/libethosu\_delegate.so |

#### Object Detection model

| Platform     | Accelerator     | Avg. Inference Time | Command                                                                                                                                         |
| ------------ | --------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| i.MX 8M Plus | CPU (1 thread)  | 244.64 ms           | ./benchmark\_model --graph=mobilenet\_ssd\_v2\_coco\_quant\_postprocess.tflite                                                                  |
| i.MX 8M Plus | CPU (4 threads) | 74.25 ms            | ./benchmark\_model --graph=mobilenet\_ssd\_v2\_coco\_quant\_postprocess.tflite --num\_threads=4                                                 |
| i.MX 8M Plus | NPU             | 11.03 ms            | ./benchmark\_model --graph=mobilenet\_ssd\_v2\_coco\_quant\_postprocess.tflite --external\_delegate\_path=/usr/lib/libvx\_delegate.so           |
| i.MX 93      | CPU (1 thread)  | 111.35 ms           | ./benchmark\_model --graph=mobilenet\_ssd\_v2\_coco\_quant\_postprocess.tflite                                                                  |
| i.MX 93      | CPU (2 threads) | 65.68 ms            | ./benchmark\_model --graph=mobilenet\_ssd\_v2\_coco\_quant\_postprocess.tflite --num\_threads=2                                                 |
| i.MX 93      | NPU             | 12.77 ms            | ./benchmark\_model --graph=mobilenet\_ssd\_v2\_coco\_quant\_postprocess\_vela.tflite --external\_delegate\_path=/usr/lib/libethosu\_delegate.so |

Two models will release on the [nxp-demo-experience-assets](https://github.com/nxp-imx-support/nxp-demo-experience-assets) in the future.
