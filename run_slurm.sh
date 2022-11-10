case $1 in
0) python train_task.py --n-controlled 1 --env-ou --comment "1-vs-ou" ;;
1) python train_task.py --n-controlled 3 --env-ou --comment "3-vs-ou" ;;
*) echo "Opcao Invalida!" ;;
esac