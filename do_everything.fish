#!/bin/fish

function check_py_exception
    # Check the exit code of the Python script
    if test $status -ne 0
        echo "Python exception occurred. Exiting..."
        exit 1
    end
end

# set -l nb_list "0 10"
set -l nb_list  "0 2" "2 4" "4 6" "6 8" "8 10" "10 12" "12 14" "14 16"
for nb in $nb_list
    /home/alberto/cosmos/PAUS_LAEs/PAUS_Lya_LF_corrections.py $nb
    check_py_exception
    /home/alberto/cosmos/PAUS_LAEs/Make_Lya_LF.py $nb
    check_py_exception
    /home/alberto/cosmos/PAUS_LAEs/LF_bootstrap_err.py $nb
    check_py_exception

end