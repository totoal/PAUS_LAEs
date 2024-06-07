#!/bin/fish

function check_py_exception
    # Check the exit code of the Python script
    if test $status -ne 0
        echo $stderr
        exit 1
    end
end

# set -l nb_list  "0 2" "2 4" "4 6" "6 8" "8 10" "10 12" "12 14" "14 16" "16 18"
set -l nb_list "0 3" "2 5" "4 7" "6 9" "8 11" "10 13" "12 15" "14 18"
# for nb in $nb_list
#     /home/alberto/cosmos/PAUS_LAEs/PAUS_Lya_LF_corrections.py $nb
#     check_py_exception
# end

for nb in $nb_list
    /home/alberto/cosmos/PAUS_LAEs/Make_Lya_LF.py $nb
    check_py_exception
end
wait

for nb in $nb_list
    /home/alberto/cosmos/PAUS_LAEs/LF_bootstrap_err.py $nb
    check_py_exception
end
wait

py LF_bootstrap_err.py "combi"

# for i in (seq 0 18)
#     set nb "$i $i"
#     /home/alberto/cosmos/PAUS_LAEs/Make_Lya_LF.py $nb
#     check_py_exception
# end
# wait

# for i in (seq 0 18)
#     set nb "$i $i"
#     /home/alberto/cosmos/PAUS_LAEs/LF_bootstrap_err.py $nb
#     check_py_exception
# end