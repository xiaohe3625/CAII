# CAII（Context-aware instance injection）

Description：Point cloud category imbalance handling for road scenes

This project was created by Xiao He.

We performed complete testing in the Paris-Lille-3D dataset, obtaining class-balanced road scene point cloud data through context-aware instance injection method, trained and tested using KPConv and RandLANet-pytorch deep learning models. The results of the Paris-Lille-3D test set (the following are the results returned by the test data submitted to the official website: https://npm3d.fr/paris-lille-3d), training and testing with RandLANet-pytorch resulted in an overall improvement of 0.9% in miou and an 8.6% improvement in the Bollard category iou. Training and testing with KPConv resulted in a 0.6% improvement in overall miou and a 4.4% improvement in Bollard category iou.
