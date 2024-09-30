# Graduation project 2024_2

DL model inference time 단축하기 위한 프로젝트

Resnet34 , CIFAR10 dataset을 사용해 학습

Edge device : Jetson Orin Nano

Jetpack 5.1.2 , Tensorflow==2.12.0+nv23.06


## First results // cpu
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


## First results // gpu

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


---

## Second

#### 위에 실험은 test.py로 돌린 것인데 처음에 MAX Pooling 없어서 더 오래걸렸음

#### mpstat 설명

```
mpstat -A == mpstat -n -u -I ALL

-u : CPU utilization

              %usr   Show the percentage of CPU utilization that
                     occurred while executing at the user level
                     (application).

              %nice  Show the percentage of CPU utilization that
                     occurred while executing at the user level with
                     nice priority.

              %sys   Show the percentage of CPU utilization that
                     occurred while executing at the system level
                     (kernel). Note that this does not include time
                     spent servicing hardware and software interrupts.

              %iowait
                     Show the percentage of time that the CPU or CPUs
                     were idle during which the system had an
                     outstanding disk I/O request.

              %irq   Show the percentage of time spent by the CPU or
                     CPUs to service hardware interrupts.

              %soft  Show the percentage of time spent by the CPU or
                     CPUs to service software interrupts.

              %steal Show the percentage of time spent in involuntary
                     wait by the virtual CPU or CPUs while the
                     hypervisor was servicing another virtual processor.

              %guest Show the percentage of time spent by the CPU or
                     CPUs to run a virtual processor.

              %gnice Show the percentage of time spent by the CPU or
                     CPUs to run a niced guest.

              %idle  Show the percentage of time that the CPU or CPUs
                     were idle and the system did not have an
                     outstanding disk I/O request.



-I {SUM, CPU, SCPU, ALL}

              CPU : /proc/interrupts 에 정의된 interrupts
              SCPU : /proc/softirqs 에 정의된 interrupts <-- 커널에 의한 interrupt(?)

```

mpstat -A 에 대한 분석해보고, interrupts 정리해보자






