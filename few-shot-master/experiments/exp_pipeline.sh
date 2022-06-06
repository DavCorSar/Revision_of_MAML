#!/bin/bash

echo Proto Nets 2-Way
echo 1-Shot
python3 proto_nets.py --dataset miniImageNet --k-train 2 --k-test 2 --q-train 15 --n-train 1 --n-test 1

echo 5-Shot
python3 proto_nets.py --dataset miniImageNet --k-train 2 --k-test 2 --q-train 15 --n-train 5 --n-test 5

echo 10-Shot
python3 proto_nets.py --dataset miniImageNet --k-train 2 --k-test 2 --q-train 15 --n-train 10 --n-test 10

echo 15-Shot
python3 proto_nets.py --dataset miniImageNet --k-train 2 --k-test 2 --q-train 15 --n-train 15 --n-test 15

echo 20-Shot
python3 proto_nets.py --dataset miniImageNet --k-train 2 --k-test 2 --q-train 15 --n-train 20 --n-test 20


echo Matching Nets 2-Way
echo 1-Shot
python3 matching_nets.py --dataset miniImageNet --fce True --k-train 2 --k-test 2 --q-train 15 --n-train 1 --n-test 1

echo 5-Shot
python3 matching_nets.py --dataset miniImageNet --fce True --k-train 2 --k-test 2 --q-train 15 --n-train 5 --n-test 5

echo 10-Shot
python3 matching_nets.py --dataset miniImageNet --fce True --k-train 2 --k-test 2 --q-train 15 --n-train 10 --n-test 10

echo 15-Shot
python3 matching_nets.py --dataset miniImageNet --fce True --k-train 2 --k-test 2 --q-train 15 --n-train 15 --n-test 15

echo 20-Shot
python3 matching_nets.py --dataset miniImageNet --fce True --k-train 2 --k-test 2 --q-train 15 --n-train 20 --n-test 20


echo Proto Nets 1-Shot
echo 10-Way
python3 proto_nets.py --dataset miniImageNet --k-train 10 --k-test 10 --q-train 15 --n-train 1 --n-test 1

echo 15-Way
python3 proto_nets.py --dataset miniImageNet --k-train 15 --k-test 15 --q-train 15 --n-train 1 --n-test 1

echo Matching Nets 1-Shot
echo 10-Way
python3 matching_nets.py --dataset miniImageNet --fce False --k-train 10 --k-test 10 --q-train 15 --n-train 1 --n-test 1

echo 15-Way
python3 matching_nets.py --dataset miniImageNet --fce False --k-train 15 --k-test 15 --q-train 15 --n-train 1 --n-test 1
