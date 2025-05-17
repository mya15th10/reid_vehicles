VeRi-776 Dataset Analysis
==================================================

[1] Train Dataset Analysis:
Total images: 37778
Unique vehicles: 576
Unique cameras: 20

[1.1] Vehicle Distribution:
Minimum images per vehicle: 11
Maximum images per vehicle: 289
Average images per vehicle: 65.59
Vehicles with fewer than 5 images: 0 (0.00%)

[1.2] Camera Distribution:

[1.3] Image Properties:
Resolution: 244x215 (average)
Aspect ratio: 1.21 (average)
Color channels: [3]

[1.4] Sample Images:

[1.5] Vehicle Examples:

==================================================
[2] Query Dataset Analysis:
Total images: 1678
Unique vehicles: 200
Unique cameras: 19

[2.1] Vehicle Distribution:
Minimum images per vehicle: 2
Maximum images per vehicle: 18
Average images per vehicle: 8.39
Vehicles with fewer than 5 images: 28 (14.00%)

[2.2] Camera Distribution:

[2.3] Sample Images:

==================================================
[3] Test Dataset Analysis:
Total images: 11579
Unique vehicles: 200
Unique cameras: 19

[3.1] Vehicle Distribution:
Minimum images per vehicle: 11
Maximum images per vehicle: 202
Average images per vehicle: 57.90
Vehicles with fewer than 5 images: 0 (0.00%)

[3.2] Camera Distribution:

[3.3] Sample Images:

==================================================
[4] Dataset Comparison:

Dataset Comparison:
            Metric Train Query  Test
      Total Images 37778  1678 11579
   Unique Vehicles   576   200   200
    Unique Cameras    20    19    19
Min Images/Vehicle    11     2    11
Max Images/Vehicle   289    18   202
Avg Images/Vehicle 65.59  8.39 57.90    

Vehicle ID Overlap:
Train-Query overlap: 0 vehicles
Train-Test overlap: 0 vehicles
Query-Test overlap: 200 vehicles 