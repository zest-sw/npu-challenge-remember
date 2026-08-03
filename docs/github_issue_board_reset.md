<!--
GitHub 이슈 초안. New Issue 화면에서 아래 "제목"을 Title에, "본문" 이하를 Body에 붙여넣으세요.
대상 레포 후보:
  - 내부 트래킹: zest-sw/npu-challenge-remember
  - 드라이버 upstream(해당되면): bos 드라이버(tenstorrent-kmd 포크) 저장소
-->

# 제목
`bos` 2.3.0: FW-init 먹통 보드를 복구할 ASIC reset ioctl이 없음 (PCIe link reset은 보드를 버스에서 제거함)

# 본문

## 요약
Blackhole 기반 BOS 보드가 **펌웨어 초기화 실패("failed to initialize FW! Try resetting the board")** 상태로 wedge되면, 로드된 `bos` 2.3.0 드라이버가 노출하는 `RESET_DEVICE` ioctl로는 이를 복구할 수 없다. ASIC 레벨 reset flag가 구현돼 있지 않고, 유일하게 "reset"에 해당하는 PCIe link reset(flag 1)은 이 보드에서 링크를 끊어 **디바이스가 PCI 버스에서 사라지게** 만든다(재부팅 외 복구 불가). 실질적으로 소프트웨어 복구 경로가 없어 매번 물리 재부팅이 필요하다.

## 환경
- KMD: `bos` **2.3.0** (`/lib/modules/6.8.0-124-generic/updates/dkms/bos.ko`, `modinfo bos` version 2.3.0)
- 드라이버 소스: `/usr/src/bos-2.3.0/`
- 보드: arch `BLACKHOLE`, PCI id `0x16c3:0xabcd`, 노드 `/dev/bos/0`, BDF `0000:01:00.0`
- 커널: 6.8.0-124-generic
- 참고: 시스템에 `/usr/src/tenstorrent-2.10.1-pre/`(더 최신 소스, `reset.c` 포함)도 존재하나 **로드된 것은 bos 2.3.0**.

## 증상 (재현)
`ttnn.open_mesh_device` 또는 최소 예제에서 FW init이 타임아웃:

```
Device 0: Timeout (10000 ms) waiting for physical cores to finish:
  (x=0,y=0), (x=2,y=1), (x=1,y=0), (x=3,y=1), (x=2,y=0), (x=0,y=1),
  (x=3,y=0), (x=1,y=1), (x=4,y=0), (x=0,y=2), (x=1,y=2), (x=4,y=1).
Device 0 init: failed to initialize FW! Try resetting the board.
RuntimeError: TT_THROW @ .../metal_context.cpp:1390
```

- 서버/사용자 코드와 무관하게 재현됨 — `python3 -m ttnn.examples.usage.run_op_on_device`(디바이스를 붙든 프로세스 없는 상태)에서도 동일하게 실패.
- 이 시점 PCIe는 정상: `lspci`에 보이고, `power_state=D0`, `/dev/bos/0` 존재.

## 근본 원인
`bos` 2.3.0의 `RESET_DEVICE` ioctl은 flag 3개만 처리한다 (`/usr/src/bos-2.3.0/ioctl.h`, `chardev.c`):

```c
// ioctl.h
#define BOS_RESET_DEVICE_RESTORE_STATE   0
#define BOS_RESET_DEVICE_RESET_PCIE_LINK 1
#define BOS_RESET_DEVICE_CONFIG_WRITE    2
```
```c
// chardev.c ioctl_reset_device()
if (in.flags == BOS_RESET_DEVICE_RESTORE_STATE) {
    safe_pci_restore_state(pdev);
    priv->device->dev_class->restore_reset_state(priv->device);
    ok = priv->device->dev_class->init_hardware(priv->device);   // 하드웨어 재init
} else if (in.flags == BOS_RESET_DEVICE_RESET_PCIE_LINK) {
    ok = pcie_hot_reset_and_restore_state(pdev);                  // 링크 hot reset
} else if (in.flags == BOS_RESET_DEVICE_CONFIG_WRITE) {
    ok = pcie_timer_interrupt(pdev);
} else {
    return -EINVAL;
}
```

- 더 최신 소스(`tenstorrent-2.10.1-pre/ioctl.h`)에는 존재하는 **`ASIC_RESET`(4) / `ASIC_DMC_RESET`(5)** flag가 bos 2.3.0엔 없음 → 해당 값으로 호출 시 `-EINVAL`.
- `flag 1`(PCIe link reset)은 이 보드에서 링크를 끊어 디바이스가 버스에서 사라짐(재부팅 필요). `/sys/.../reset`(PCIe FLR)도 동일하게 D3cold/링크 소실을 유발함.
- 즉 FW-init 실패를 고칠 **ASIC/보드 레벨 reset이 부재**.

## 기대 동작
FW init이 실패한 보드를, PCIe 링크를 끊지 않고 소프트웨어로 복구할 수 있어야 한다(예: `ASIC_RESET` flag, 또는 tt-smi류의 board reset 경로 backport).

## 현재 워크어라운드
1. **드라이버 재적재** — probe(`enumerate.c bos_pci_probe` → `init_device`→`init_hardware`→`post_hardware_init`)가 하드웨어 init 시퀀스를 다시 돌리므로, 디바이스를 붙든 프로세스가 없을 때 아래가 유효할 수 있음:
   ```bash
   sudo rmmod bos && sudo modprobe bos
   ```
2. 위가 실패하면 **물리 재부팅**.
3. **하지 말 것**: `/sys/bus/pci/devices/0000:01:00.0/reset`(FLR) 및 `RESET_DEVICE` flag 1 — 둘 다 링크를 끊어 디바이스를 버스에서 제거함.

## 요청
- bos 드라이버에 ASIC-level reset(또는 `flag 0`의 `init_hardware` 재init 경로를 사용자 공간에서 안전하게 트리거하는 수단) 제공/backport 여부 확인.
- FW-init 먹통 시 권장 복구 절차 문서화.
