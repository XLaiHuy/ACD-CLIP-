# P25 Performance and Execution Contract

CPU exact AP is authoritative. GPU may be selected only after <=1e-12 real
fixture AP/V parity, identical signs/order/trajectory, and lower measured
end-to-end projected runtime. TF32 is off in any parity-critical CUDA path.

The frozen real engineering fixture is candle with 128 panel candidates chosen
by the panel hash before V inspection. It compares sparse fast target AP with
direct frozen deployment: AP/V error <=1e-12, zero sign and ordering mismatch.

Projection uses actual 12-class inventory and includes target generation, Q1,
conditional Q2, audit, and serialization. Preferred total <=120 minutes;
hard worst-case <=180 minutes. Target <=60, Q1 <=20, calibration <=10, Q2
<=30, audit <=15 minutes are planning budgets. Peak host RSS must stay below
14 GiB; only one active class target cache may retain full arrays.

Before the marker: all T01--T30 tests, py_compile, production-path smoke,
strict JSON, exactly-once rehearsal, target parity fixture, provenance,
firewall, memory benchmark, performance projection, and remote-clean execution
base must pass. Progress is atomic and monotonic. P25 creates no attempt marker
before those gates.
