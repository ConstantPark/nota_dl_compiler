
## TVM: n Automated End-to-End Optimizing Compiler for Deep Learning
TVM과 관련된 자료들을 포함하고 있습니다. 다음과 같은 내용을 포함하고 있습니다.
- AutoTVM, Auto Scheduler
- MicroTVM
- Relay and Quantization

각각의 Python 파일은 별도의 옵션 없이 바로 실행가능합니다.

```cmd
python pth2onnx.py --pretrained --arch='mobilenet_v2' --opset=12 \
        --output='/home/workspace/model_optimization/pth/mobilenet_v2.onnx'
```
