
## PyTorch Mobile with Android/IOS, LibTorch
TVM과 관련된 자료들을 포함하고 있습니다. 다음과 같은 내용을 포함하고 있습니다.
- Pytorch Mobile (TorchScript 변환, torch_script_transform.py)
- NNAPI
- LibTorch (Libtorch Baseline/c++_code)

각각의 Python 파일은 별도의 옵션 없이 바로 실행가능합니다.   
MicroTVM은 별도의 세팅 과정이 필요하며, 세팅 과정에서 QEMU 문제로 실행이 불가능함을 확인하였습니다. 

```cmd
python xxx.py (opt_gemm.py) # 해당 Python 파일을 직접 실행
```
