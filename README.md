# Graduation project 2024_2

DL model inference time 단축하기 위한 프로젝트

Resnet34 , CIFAR10 dataset을 사용해 학습

Edge device : Jetson Orin Nano

Jetpack 5.1.2 , Tensorflow==2.12.0+nv23.06

but Original resnet으로 재학습해야됌

## Mid results // cpu
```
Model used: resnet34_cifar10_bs64_epochs10.keras
GPU used: No
CPU Threads used: 1
Average inference time: 1111.3610269228618 seconds
Max inference time: 1111.7148716449738 seconds
Min inference time: 1110.8259530067444 seconds
```

```
Model used: resnet34_cifar10_bs64_epochs10.keras
GPU used: No
CPU Threads used: 3
Average inference time: 400.5844699541728 seconds
Max inference time: 401.31527638435364 seconds
Min inference time: 400.1734607219696 seconds
```

```
Model used: resnet34_cifar10_bs64_epochs10.keras
GPU used: No
CPU Threads used: 6
Average inference time: 218.26513568560281 seconds
Max inference time: 220.73794174194336 seconds
Min inference time: 216.96071934700012 seconds
```

thread가 증가할수록 추론시간이 단축


## Mid results // gpu

```
Model used: resnet34_cifar10_bs64_epochs10.keras
GPU used: Yes
Average inference time: 18.36625822385152 seconds
Max inference time: 20.129164934158325 seconds
Min inference time: 17.39278745651245 seconds
--> thread 1
```

```
Model used: resnet34_cifar10_bs64_epochs10.keras
GPU used: Yes
Average inference time: 17.606494108835857 seconds
Max inference time: 18.5928955078125 seconds
Min inference time: 17.10651993751526 seconds
--> thread 3
```

```
Model used: resnet34_cifar10_bs64_epochs10.keras
GPU used: Yes
Average inference time: 17.755841970443726 seconds
Max inference time: 18.757434368133545 seconds
Min inference time: 17.233683824539185 seconds
-->thread 6
```

```
Model used: resnet34_cifar10_bs64_epochs10.keras
GPU used: Yes
Average inference time: 17.776134967803955 seconds
Max inference time: 18.74548363685608 seconds
Min inference time: 17.259194135665894 seconds
--> thread None
```

thread에 따른 효과 X



추후 진행사항
TensorRT 변환해서 추론 시간 측정

1) CPU 추론 시 추론시간 --> pidstat 등으로 cpu사용률 체크

2) GPU 추론 시 추론시간을 줄이는 방법에 대한 연구 (여기에 중점)
   







