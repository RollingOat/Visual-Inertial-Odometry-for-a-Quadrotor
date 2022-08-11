# Stereo Visual Inertial Odometry for a Quadrotor

## Task: Given IMU and stereo data during a flight, estimate the state of the quadrotor using error state Kalman filter

## Summary of ESKF algorithm (Penn MEAM 620)
1. Initialize the filter with state vector **x** = (p, v, q, a~b, a~w, g) and error state covariance matrix, **p**.
2. For each new imu reading or vision measurement
  - Compute time since last update, dt
  - Update error state covariance matrix
  - Update nominal state using using prevailing IMU measurement a_m, w_m
  - if there is a vision measurement, for each stereo correspondence
      - check if measurement is an inlier
      - compute relevant Jacobian, H
      - COmpute EFK update 
      - Update nominal state
      - Update error state covariance matrix
  
## 1. Nominal State Update
![image](https://user-images.githubusercontent.com/97129990/162635335-30e0c721-5266-4b68-bfd4-45c32c7bb854.png)
![image](https://user-images.githubusercontent.com/97129990/162635375-a1fd7cc1-99de-455f-b8e3-7162478c2dd3.png)

## 2. Error State Update
![image](https://user-images.githubusercontent.com/97129990/162636014-dbbc73fb-e908-4753-b0d0-b1a6ffe4b35e.png)
![image](https://user-images.githubusercontent.com/97129990/162637302-3d3cb5d4-741c-4912-82e9-e6390090b05e.png)

## 3. Incorporate observation to update nominal state
