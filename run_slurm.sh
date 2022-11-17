case $1 in
0) python train_task.py --n-controlled 3 --env-ou --comment "permutations" ;;
*) echo "Opcao Invalida!" ;;
esac