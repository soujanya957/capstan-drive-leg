# Body stress test (fullbuild_v3): results

Linear-elastic FEA (`src/stress_test.py`; gmsh P2 tetrahedra, scikit-fem). Peak von Mises with the hottest 0.1 % of
the volume and the elements within 2.5 mm of the held faces left out. SF: PETG against 45 MPa along / 25 MPa across
the layers; 6061-T6 against 276 MPa yield. Displacements: PETG (E 2000 MPa) and 6061 (E 68.9 GPa).

Pod bearing reactions per foot at 585 N (continuous beam, foot under the hip at x 221.5): x 38.8: -307 N, x 103.5: 682 N, x 336.5: 211 N; motor C rope pull 450 N (half per end).

| Part | Case | von Mises MPa | across layers MPa | SF printed (PETG) | SF laser (6061) | max disp PETG / 6061 mm | target | pass (printed / laser) |
|---|---|---|---|---|---|---|---|---|
| Centre bulkhead | push / landing 585 N | 2.8 | 0.5 | 16.12 | 98.9 | 0.21 / 0.006 | 2.0 | ✓ / ✓ |
| Centre bulkhead | impact x2 1170 N | 5.0 | 0.7 | 9.02 | 55.3 | 0.22 / 0.006 | 1.0 | ✓ / ✓ |
| Centre bulkhead | one leg x2 1170 N | 4.1 | 0.6 | 11.0 | 67.5 | 0.22 / 0.006 | 1.0 | ✓ / ✓ |
| Inner bulkhead | push / landing 585 N | 3.0 | 0.6 | 14.89 | 91.3 | 0.08 / 0.002 | 2.0 | ✓ / ✓ |
| Inner bulkhead | impact x2 1170 N | 7.5 | 1.5 | 6.01 | 36.9 | 0.18 / 0.005 | 1.0 | ✓ / ✓ |
| Inner bulkhead | one leg x2 1170 N | 6.4 | 1.1 | 6.98 | 42.8 | 0.19 / 0.005 | 1.0 | ✓ / ✓ |
| End frame | push / landing 585 N | 2.4 | 0.3 | 18.92 | 116.0 | 0.07 / 0.002 | 2.0 | ✓ / ✓ |
| End frame | impact x2 1170 N | 4.8 | 0.6 | 9.46 | 58.0 | 0.14 / 0.004 | 1.0 | ✓ / ✓ |
| End frame | one leg x2 1170 N | 4.3 | 0.4 | 10.37 | 63.6 | 0.14 / 0.004 | 1.0 | ✓ / ✓ |
