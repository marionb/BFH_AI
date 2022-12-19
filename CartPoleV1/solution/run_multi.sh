truncate -s 0 logs.txt
for i in {1..2}
do
  echo "------------run $i-----------" >> logs.txt
  python3 deep_Q_Network.py >> logs.txt
  echo -e "\n------------------------------" >> logs.txt
done