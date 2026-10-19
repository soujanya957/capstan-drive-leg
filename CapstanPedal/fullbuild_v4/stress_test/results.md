# Body stress test (fullbuild_v4): results

Linear-elastic FEA (`src/stress_test.py`; gmsh P2 tetrahedra, scikit-fem). Peak von Mises with the hottest 0.1 % of
the volume and the elements within 2.5 mm of the held faces left out. SF: PETG against 45 MPa along / 25 MPa across
the layers; 6061-T6 against 276 MPa yield. Displacements: PETG (E 2000 MPa) and 6061 (E 68.9 GPa).

Pod bearing reactions per foot at 585 N (continuous beam, foot under the hip at x 221.5): x 38.8: -307 N, x 103.5: 682 N, x 336.5: 211 N; motor C rope pull 450 N (half per end).

| Part | Case | von Mises MPa | across layers MPa | SF printed (PETG) | SF laser (6061) | max disp PETG / 6061 mm | target | pass (printed / laser) |
|---|---|---|---|---|---|---|---|---|
| Centre bulkhead | push / landing 585 N | 4.2 | 0.7 | 10.68 | 65.5 | 0.21 / 0.006 | 2.0 | ✓ / ✓ |
| Centre bulkhead | impact x2 1170 N | 7.6 | 1.2 | 5.89 | 36.1 | 0.21 / 0.006 | 1.0 | ✓ / ✓ |
| Centre bulkhead | one leg x2 1170 N | 6.2 | 0.9 | 7.25 | 44.5 | 0.23 / 0.007 | 1.0 | ✓ / ✓ |
| Inner bulkhead | push / landing 585 N | 2.9 | 0.7 | 15.55 | 95.4 | 0.07 / 0.002 | 2.0 | ✓ / ✓ |
| Inner bulkhead | impact x2 1170 N | 7.2 | 1.6 | 6.28 | 38.5 | 0.17 / 0.005 | 1.0 | ✓ / ✓ |
| Inner bulkhead | one leg x2 1170 N | 6.0 | 1.2 | 7.53 | 46.2 | 0.17 / 0.005 | 1.0 | ✓ / ✓ |
| End frame | push / landing 585 N | 2.1 | 0.3 | 21.9 | 134.3 | 0.06 / 0.002 | 2.0 | ✓ / ✓ |
| End frame | impact x2 1170 N | 4.1 | 0.6 | 10.95 | 67.2 | 0.12 / 0.004 | 1.0 | ✓ / ✓ |
| End frame | one leg x2 1170 N | 3.8 | 0.5 | 11.82 | 72.5 | 0.13 / 0.004 | 1.0 | ✓ / ✓ |
