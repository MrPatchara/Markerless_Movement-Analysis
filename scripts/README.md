# Scripts – เครื่องมือ CLI

สคริปต์ที่รันจาก command line (ไม่ผ่าน GUI)

| สคริปต์ | คำสั่ง | ใช้สำหรับ |
|---------|--------|-----------|
| **analyze_sls** | `python scripts/analyze_sls.py <file> [--side L\|R]` | วิเคราะห์ SLS/FPKPA จาก C3D หรือ JSON |
| **debug_force** | `python scripts/debug_force.py <c3d_path>` | ตรวจสอบ force/COP ใน C3D |
| **inspect_c3d** | `python scripts/inspect_c3d.py <c3d_path>` | ดูโครงสร้าง C3D (FORCE_PLATFORM, analog) |

## ตัวอย่าง

```bash
# วิเคราะห์ SLS
python scripts/analyze_sls.py Data/test.json --side R

# ตรวจสอบ force
python scripts/debug_force.py c3d/mytrial.c3d

# ดูโครงสร้าง C3D
python scripts/inspect_c3d.py c3d/mytrial.c3d
```
