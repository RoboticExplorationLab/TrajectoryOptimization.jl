exename="/home/odroid/Julia/julia-1.0.4/bin/julia"
blas=true
addprocs([("odroid@rexquad3",1)], exename=exename, exeflags="--project=$(@__DIR__)", enable_threaded_blas=blas)
addprocs([("odroid@quad3",1)], exename=exename, exeflags="--project=$(@__DIR__)", enable_threaded_blas=blas)
addprocs(1, exeflags="--project=$(@__DIR__)", enable_threaded_blas=blas)
