# Autonomous vehicles: machine vision exercises

**Autonomna vozila: algoritmi strojnog vida**

> Course exercises – Autonomous Vehicles (Autonomna vozila), FSB, University of Zagreb, 2026

![C++](https://img.shields.io/badge/C%2B%2B-17-00599C)
![OpenCV](https://img.shields.io/badge/vision-OpenCV-5C3EE8)
![CMake](https://img.shields.io/badge/build-CMake-064F8C)

Lab exercise 2 of the course, on classic machine vision algorithms for a vehicle's perception.
Three small C++ programs built on OpenCV find traffic cones by colour, traffic signs by shape, and
the track in a sequence of camera frames. Each one shows its intermediate images and has sliders for tuning
the parameters while it runs.

## What's inside

All the work is in `vjezbe_02_algoritmi_srojnog_vida/`.

| Program | Source | What it does |
|---|---|---|
| `Cunjevi` | `src/test_color_cunjevi.cpp` | detects red and green cones with HSV colour thresholds and marks each one with a box and a label |
| `Znakovi` | `src/test_shape_znakovi.cpp` | finds traffic signs with adaptive thresholding and contours, filtered by area, aspect ratio and polygon approximation |
| `Auto` | `src/test_track_autonomous_vehicles.cpp` | goes through the frames in `data/` and keeps the contours that match the track by area and length |
| `OpenCV` | `src/test.cpp` | the starting contour detection example |

The test images are in `data/`.

## Build and run

The programs need OpenCV and a C++17 compiler. They read the images from `../data`, so they are
started from the `build/` folder.

```bash
cd vjezbe_02_algoritmi_srojnog_vida
cmake -S . -B build
cmake --build build
cd build
./Cunjevi
```
