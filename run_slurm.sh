case $1 in
0) python train_task.py --n-controlled 1 --comment "refactored-1-vs-stopped" ;;
1) python train_task.py --n-controlled 3 --comment "refactored-3-vs-stopped" ;;
2) python train_task.py --n-controlled 1 --env-ou --comment "refactored-1-vs-ou" ;;
3) python train_task.py --n-controlled 3 --env-ou --comment "refactored-3-vs-ou" ;;
*) echo "Opcao Invalida!" ;;
esac