#!/usr/bin/env bash
# First five patients (Argument before False, at 1)
source run_networks.sh TEMPONet 2 2 32 8 1 5 3 0 14 16 0 1 False True
# Second five patients (Argument before False, at 2)
source run_networks.sh TEMPONet 2 2 32 8 1 5 3 0 14 16 0 2 False True

source run_networks.sh ViT 1 1 8  8 1 10 0 0 14 0 0 1 False True
source run_networks.sh ViT 1 1 8  8 1 10 0 0 14 0 0 2 False True
source run_networks.sh ViT 1 1 16 8 1 10 0 0 14 0 0 1 False True
source run_networks.sh ViT 1 1 16 8 1 10 0 0 14 0 0 2 False True
source run_networks.sh ViT 2 1 32 8 1 10 0 0 14 0 0 1 False True
source run_networks.sh ViT 1 1 32 8 1 10 0 0 14 0 0 2 False True


source run_networks.sh ViT 1 1 8  8 1 30 0 0 14 0 0 1 False True
source run_networks.sh ViT 1 1 8  8 1 30 0 0 14 0 0 2 False True
source run_networks.sh ViT 1 1 16 8 1 30 0 0 14 0 0 1 False True
source run_networks.sh ViT 1 1 16 8 1 30 0 0 14 0 0 2 False True


source run_networks.sh ViT 1 1 8  8 1 60 0 0 14 0 0 1 False True
source run_networks.sh ViT 1 1 8  8 1 60 0 0 14 0 0 2 False True
source run_networks.sh ViT 1 1 16 8 1 60 0 0 14 0 0 1 False True
source run_networks.sh ViT 1 1 16 8 1 60 0 0 14 0 0 2 False True