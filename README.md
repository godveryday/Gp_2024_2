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

### 위에 실험은 test.py로 돌린 것인데 처음에 MAX Pooling 없어서 더 오래걸렸음

### mpstat 설명

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

### Orin interrupts 상황

### /proc/interrupts
```
           CPU0       CPU1       CPU2       CPU3       CPU4       CPU5       
  9:          0          0          0          0          0          0     GICv3  25 Level     vgic
 11:          0          0          0          0          0          0     GICv3  30 Level     kvm guest ptimer
 12:          0          0          0          0          0          0     GICv3  27 Level     kvm guest vtimer
 13:   90531155   96257032   67361432   65420324   64203809   66971399     GICv3  26 Level     arch_timer
 18:        170          0          0          0          0          0     GICv3 255 Level     mc_status
 20:          0          0          0          0          0          0     GICv3 202 Level     arm-smmu global fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault
 21:          0          0          0          0          0          0     GICv3 264 Level     arm-smmu global fault, arm-smmu-context-fault
 22:          0          0          0          0          0          0     GICv3 272 Level     arm-smmu global fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault
 23:         90          0          0          0          0          0     GICv3 270 Level     arm-smmu global fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault, arm-smmu-context-fault
 24:         80          0          0          0          0          0     GICv3 274 Level     arm-smmu global fault, arm-smmu-context-fault
 25:          0          0          0          0          0          0     GICv3 263 Level     13a00000.cbb-fabric
 26:          0          0          0          0          0          0     GICv3 204 Level     c600000.aon-fabric
 27:          0          0          0          0          0          0     GICv3 206 Level     d600000.bpmp-fabric
 28:          0          0          0          0          0          0     GICv3 413 Level     de00000.dce-fabric
 29:          0          0          0          0          0          0     GICv3 207 Level     be00000.rce-fabric
 30:          0          0          0          0          0          0     GICv3 205 Level     b600000.sce-fabric
 31:          0          0          0          0          0          0     GICv3 368 Level     tegra-p2u-intr
 32:          0          0          0          0          0          0     GICv3 369 Level     tegra-p2u-intr
 33:          0          0          0          0          0          0     GICv3 370 Level     tegra-p2u-intr
 34:          0          0          0          0          0          0     GICv3 371 Level     tegra-p2u-intr
 35:          0          0          0          0          0          0     GICv3 372 Level     tegra-p2u-intr
 36:          0          0          0          0          0          0     GICv3 373 Level     tegra-p2u-intr
 37:          0          0          0          0          0          0     GICv3 374 Level     tegra-p2u-intr
 38:          0          0          0          0          0          0     GICv3 375 Level     tegra-p2u-intr
 39:          0          0          0          0          0          0     GICv3 376 Level     tegra-p2u-intr
 40:          0          0          0          0          0          0     GICv3 377 Level     tegra-p2u-intr
 41:          0          0          0          0          0          0     GICv3 378 Level     tegra-p2u-intr
 42:          0          0          0          0          0          0     GICv3 379 Level     tegra-p2u-intr
 43:          0          0          0          0          0          0     GICv3 380 Level     tegra-p2u-intr
 44:          0          0          0          0          0          0     GICv3 381 Level     tegra-p2u-intr
 45:          0          0          0          0          0          0     GICv3 382 Level     tegra-p2u-intr
 46:          0          0          0          0          0          0     GICv3 383 Level     tegra-p2u-intr
 47:          0          0          0          0          0          0     GICv3 235 Level     tegra-p2u-intr
 48:          0          0          0          0          0          0     GICv3 252 Level     tegra-p2u-intr
 49:          0          0          0          0          0          0     GICv3 253 Level     tegra-p2u-intr
 50:          0          0          0          0          0          0     GICv3 254 Level     tegra-p2u-intr
 51:          0          0          0          0          0          0     GICv3 140 Level     tegra-p2u-intr
 52:          0          0          0          0          0          0     GICv3 141 Level     tegra-p2u-intr
 53:          0          0          0          0          0          0     GICv3 142 Level     tegra-p2u-intr
 54:          0          0          0          0          0          0     GICv3 143 Level     tegra-p2u-intr
 55:          0          0          0          0          0          0     GICv3  77 Level     tegra-pcie-intr, PCIe PME, aerdrv
 57:          0          0          0          0          0          0     GICv3  83 Level     tegra-pcie-intr
 59:          0          0          0          0          0          0     GICv3 386 Level     tegra-pcie-intr
 61:          0          0          0          0          0          0     GICv3 388 Level     tegra-pcie-intr, PCIe PME, aerdrv
 65:        632          0          0          0          0          0     GICv3 317 Level     uart-pl011
 66:          0          0          0          0          0          0     GICv3 152 Level     3c00000.tegra-hsp
 67:   54311800          0          0          0          0          0     GICv3  94 Level     mmc1
 68:          0          0          0          0          0          0     GICv3  68 Level     3210000.spi
 69:          0          0          0          0          0          0     GICv3  70 Level     3230000.spi
 70:        241          0          0          0          0          0     GICv3  57 Level     3160000.i2c
 71:      80493          0          0          0          0          0     GICv3  58 Level     c240000.i2c
 72:          2          0          0          0          0          0     GICv3  59 Level     3180000.i2c
 73:          0          0          0          0          0          0     GICv3  60 Level     3190000.i2c
 74:          0          0          0          0          0          0     GICv3  62 Level     31b0000.i2c
 75:          0          0          0          0          0          0     GICv3  63 Level     31c0000.i2c
 76:          0          0          0          0          0          0     GICv3  64 Level     c250000.i2c
 77:          0          0          0          0          0          0     GICv3  65 Level     31e0000.i2c
 79:          0          0          0          0          0          0     GICv3 108 Level     gpcdma.0
 80:          0          0          0          0          0          0     GICv3 109 Level     gpcdma.1
 81:          0          0          0          0          0          0     GICv3 110 Level     gpcdma.2
 82:          0          0          0          0          0          0     GICv3 111 Level     gpcdma.3
 83:          0          0          0          0          0          0     GICv3 112 Level     gpcdma.4
 84:          0          0          0          0          0          0     GICv3 113 Level     gpcdma.5
 85:          0          0          0          0          0          0     GICv3 114 Level     gpcdma.6
 86:          0          0          0          0          0          0     GICv3 115 Level     gpcdma.7
 87:          0          0          0          0          0          0     GICv3 116 Level     gpcdma.8
 88:          0          0          0          0          0          0     GICv3 117 Level     gpcdma.9
 89:          0          0          0          0          0          0     GICv3 118 Level     gpcdma.10
 90:          0          0          0          0          0          0     GICv3 119 Level     gpcdma.11
 91:          0          0          0          0          0          0     GICv3 120 Level     gpcdma.12
 92:          0          0          0          0          0          0     GICv3 121 Level     gpcdma.13
 93:          0          0          0          0          0          0     GICv3 122 Level     gpcdma.14
 94:          0          0          0          0          0          0     GICv3 123 Level     gpcdma.15
 95:          0          0          0          0          0          0     GICv3 124 Level     gpcdma.16
 96:          0          0          0          0          0          0     GICv3 125 Level     gpcdma.17
 97:          0          0          0          0          0          0     GICv3 126 Level     gpcdma.18
 98:          0          0          0          0          0          0     GICv3 127 Level     gpcdma.19
112:   28955693          0          0          0          0          0     GICv3  92 Level     snd_hda_tegra
113:          0          0          0          0          0          0     GICv3  51 Level     bc00000.rtcpu
114:      24335          0          0          0          0          0     GICv3 242 Level     d230000.actmon
115:          0          0          0          0          0          0     GICv3  23 Level     arm-pmu
116:          0          0          0          0          0          0     GICv3 579 Level     arm_dsu_0
117:          0          0          0          0          0          0     GICv3 580 Level     arm_dsu_1
118:          0          0          0          0          0          0     GICv3 581 Level     arm_dsu_2
119:          0          0          0          0          0          0     GICv3 583 Level     scf-pmu
120:          0          0          0          0          0          0     GICv3 398 Level     e860000.psc
121:          0          0          0          0          0          0     GICv3 399 Level     e860000.psc
122:          0          0          0          0          0          0     GICv3 400 Level     e860000.psc
123:          0          0          0          0          0          0     GICv3 401 Level     e860000.psc
124:          0          0          0          0          0          0     GICv3 402 Level     e860000.psc
125:          0          0          0          0          0          0     GICv3 403 Level     e860000.psc
126:          0          0          0          0          0          0     GICv3 404 Level     e860000.psc
127:          0          0          0          0          0          0     GICv3 405 Level     e860000.psc
128:     183586          0          0          0          0          0     GICv3 480 Level     host_syncpt
129:          0          0          0          0          0          0     GICv3 481 Level     host_syncpt
130:          0          0          0          0          0          0     GICv3 482 Level     host_syncpt
131:          0          0          0          0          0          0     GICv3 483 Level     host_syncpt
132:          0          0          0          0          0          0     GICv3 484 Level     host_syncpt
133:          0          0          0          0          0          0     GICv3 485 Level     host_syncpt
134:          0          0          0          0          0          0     GICv3 486 Level     host_syncpt
135:          0          0          0          0          0          0     GICv3 487 Level     host_syncpt
136:          0          0          0          0          0          0     GICv3 295 Level     host_status
137:          0          0          0          0          0          0     GICv3 238 Level     vic
138:          0          0          0          0          0          0     GICv3 260 Level     tsec_riscv_irq
150:       6059          0          0          0          0          0     GICv3 165 Level     c150000.tegra-hsp
154:   35693103          0          0          0          0          0     GICv3 208 Level     3c00000.tegra-hsp
162:          1          0          0          0          0          0     GICv3 160 Level     3d00000.tegra-hsp
167:        128          0          0          0          0          0     GICv3 214 Level     b950000.tegra-hsp
171:          0          0          0          0          0          0     GICv3 316 Level     tegra-se-nvrng
224:      75534          0          0          0          0          0     GICv3  39 Level     2190000.watchdog
228:          0          0          0          0          0          0     GICv3 199 Level     3610000.xhci
229:          0          0          0          0          0          0     GICv3 198 Level     3550000.xudc
230:       2782          0          0          0          0          0     GICv3 195 Level     xhci-hcd:usb1
231:          0          0          0          0          0          0     GICv3 196 Level     3610000.xhci
232:          0          0          0          0          0          0     GICv3 100 Level     gk20a_stall
233:       4658          0          0          0          0          0     GICv3 102 Level     gk20a_stall
234:        241          0          0          0          0          0     GICv3 103 Level     gk20a_stall
235:          0          0          0          0          0          0     GICv3  99 Level     gk20a_nonstall
236:          0          0          0          0          0          0     GICv3 408 Level     tegra_dce_isr
237:       1003          0          0          0          0          0     GICv3 409 Level     tegra_dce_isr
245:          0          0          0          0          0          0  c360000.pmc  73 Level     tegra_rtc
246:          0          0          0          0          0          0  c360000.pmc  24 Level     nvvrs-pseq-irq
247:          0          0          0          0          0          0  nvvrs-pseq-irq   3 Edge      rtc-alarm
249:        517          0          0          0          0          0  agic-controller  73 Edge    
250:        259          0          0          0          0          0  agic-controller  64 Edge    
252:        258          0          0          0          0          0  agic-controller  94 Edge    
253:        258          0          0          0          0          0  agic-controller  89 Level   
262:        342          0          0          0          0          0  agic-controller  32 Level   
263:        331          0          0          0          0          0  agic-controller  33 Level   
294:          0          0          0          0          0          0      gpio  42 Edge      3400000.sdhci cd
296:          0          0          0          0          0          0  c360000.pmc  83 Edge      sw-wake
297:          0          0          0          0          0          0      gpio  35 Edge      force-recovery
298:          0          0          0          0          0          0      gpio  27 Edge      power-key
299:          0          0          0          0          0          0      gpio 131 Level     1-0025
301:   42525629          0          0          0          0          0       MSI 1074266112 Edge      eth0
302:    4427681          0          0          0          0          0       MSI 134742016 Edge      rtl88x2ce
IPI0:  35679938   46384122   43490193   51915366   52719530   49081324       Rescheduling interrupts
IPI1:   4568883    3194939    2190373    1773975    1393612    1417165       Function call interrupts
IPI2:         0          0          0          0          0          0       CPU stop interrupts
IPI3:         0          0          0          0          0          0       CPU stop (for crash dump) interrupts
IPI4:         0          0          0          0          0          0       Timer broadcast interrupts
IPI5:   2540092    1642117    1455601    1413417    2661213    2751161       IRQ work interrupts
IPI6:         0          0          0          0          0          0       CPU wake-up interrupts
Err:          0


```

특정 interrupts의 경우 CPU0에서만 대부분을 처리함




### /proc/softirq 

```
                    CPU0       CPU1       CPU2       CPU3       CPU4       CPU5       
          HI:    4320918          0          0          0          0          0
       TIMER:   27125953   63321135   32739129   30630301   29605333   32752780
      NET_TX:    1198442          3          3          7          1          4
      NET_RX:   39139986        889        636        648        600       1208
       BLOCK:     801717     111323     105842     110657      93722     122628
    IRQ_POLL:          0          0          0          0          0          0
     TASKLET:      11744         32          5          3          0         17
       SCHED:   38621830   61625664   35374541   32986445   32767456   35976765
     HRTIMER:      10464       1572        538        312         10         16
         RCU:   13958365   13787975   12362731   12090094   11692892   12519382

```
  


이거 기반으로 mpstat -A 에 대한 분석해보고, interrupts 정리해보자

softriq 의 경우 HI, NET_TX/RX, TASKLET, HRTIMER 에서 CPU0가 월등히 높은 수치

SCHED 에서는 CPU1번이 약 2배정도 많은 수치를 보임 --> ResNet 추론에서도 동일한지 check

그런데 CPU에 대한 연구가 메리트가 있는지 의문







