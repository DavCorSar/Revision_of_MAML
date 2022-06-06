#!/bash/bin

echo 2-Way 1-Shot
echo Training...
#python3 miniimagenet_train_few_shot.py -w 2 -s 1 -b 15
echo Testing...
#python3 miniimagenet_test_few_shot.py -w 2 -s 1 -b 15 >> relation_nets_2Way_1Shot_b15.txt


echo 2-Way 5-Shot
echo Training...
#python3 miniimagenet_train_few_shot.py -w 2 -s 5 -b 15
echo Testing...
#python3 miniimagenet_test_few_shot.py -w 2 -s 5 -b 15 >> relation_nets_2Way_5Shot_b15.txt


echo 2-Way 10-Shot
echo Training...
#python3 miniimagenet_train_few_shot.py -w 2 -s 10 -b 15
echo Testing...
#python3 miniimagenet_test_few_shot.py -w 2 -s 10 -b 15 >> relation_nets_2Way_10Shot_b15.txt


echo 2-Way 15-Shot
echo Training...
#python3 miniimagenet_train_few_shot.py -w 2 -s 15 -b 15
echo Testing...
#python3 miniimagenet_test_few_shot.py -w 2 -s 15 -b 15 >> relation_nets_2Way_15Shot_b15.txt


echo 2-Way 20-Shot
echo Training...
#python3 miniimagenet_train_few_shot.py -w 2 -s 20 -b 15
echo Testing...
#python3 miniimagenet_test_few_shot.py -w 2 -s 20 -b 15 >> relation_nets_2Way_20Shot_b15.txt


#echo 2-Way 25-Shot
#python3 miniimagenet_train_few_shot.py -w 2 -s 25 -b 15



echo 5-Way 1-Shot
echo Training...
python3 miniimagenet_train_few_shot.py -w 5 -s 1 -b 14
echo Testing...
python3 miniimagenet_test_few_shot.py -w 5 -s 1 -b 14 >> relation_nets_5Way_1Shot_b14.txt


#echo 10-Way 1-Shot
#python3 miniimagenet_train_few_shot.py -w 10 -s 1 -b 15

#echo 15-Way 1-Shot
#python3 miniimagenet_train_few_shot.py -w 15 -s 1 -b 15
