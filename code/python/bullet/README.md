# README
## Data Collection

Use `vr_demo_collect.py` to record demo in VR and replay demo and generate trace file.

Use `bullet3-master\bin>test_vhacd_vs2010_x64_release.exe` to decompose comcave objects

### Trace file: data exchange format
First line will be the object id of kuka, object A, object B, etc.
Each following line is formatted as

```
	[Time] [ObjectID] [x] [y] [z] [qx] [qy] [qz] [qw]
```

for kuka:

```
	[Time] [ObjectID] [x] [y] [z] [qx] [qy] [qz] [qw] [left_finger_joint] [right_finger_joint]
```
